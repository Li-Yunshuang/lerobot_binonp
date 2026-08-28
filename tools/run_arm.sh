#!/bin/bash
# Train and evaluate one experiment arm.
#
# Every arm holds the input stack and the heads fixed and varies one thing, so results across the
# campaign stay comparable. Usage:
#
#   tools/run_arm.sh <name> <backbone> <dataset_root> <repo_id> [extra --policy.* flags...]
#
# oom_score_adj is set so the kernel kills training rather than the desktop session; children
# inherit it. Training and evaluation are sequential because one GPU serves both.
set -euo pipefail

NAME=$1; BACKBONE=$2; DS_ROOT=$3; REPO=$4; shift 4
EXTRA=("$@")

RUNS=/home/samsung/data/runs
LEROBOT_PY=/home/samsung/miniforge3/envs/lerobot/bin/python
ISAAC_PY=/home/samsung/miniforge3/envs/isaaclab/bin/python
OUT=$RUNS/$NAME
LOG=${ARM_LOG_DIR:-$OUT}
SOCK=/tmp/pc_arm_$NAME.sock
STEPS=${STEPS:-100000}
BATCH=${BATCH:-64}
WORKERS=${WORKERS:-6}
SEED=${SEED:-1000}
# Primitive name; becomes observation.task_onehot for the multi-task policies.
TASK=${TASK:-push}
# Rollouts per object. The curated evaluation set trades objects for a higher ceiling, so the
# rollout count has to rise to keep the comparison resolvable: 14 objects x 6 resolves only
# +/-15 pp, which is worse than the uncurated set it replaced. 12 -> +/-10 pp for screening,
# 20 -> +/-8 pp for confirming finalists.
EPISODES_PER_OBJECT=${EPISODES_PER_OBJECT:-12}
# Which evaluation object lists to use. "_curated" keeps only objects the scripted expert solves
# at least 75% of the time; failures elsewhere score the demonstrator, not the policy.
# `-` not `:-`: an explicitly empty value must mean "no suffix", whereas `:-` would
# silently substitute the default for an empty string and evaluate the wrong lists.
EVAL_VARIANT=${EVAL_VARIANT-_curated}
# Envs per object during evaluation. The harness maps env i to object (i % n_objects), so a
# multiple of the object count gives every object that many rollouts per batch and cuts the number
# of sequential batches proportionally. 2x turns 12 batches into 6 at roughly 1.5x the VRAM.
EVAL_ENV_MULT=${EVAL_ENV_MULT:-2}
# Rollout videos per split. Cheap: the goal ghost is rasterised once per batch (~6 s) and reused
# across frames, and only the first N envs are captured. 0 disables.
RECORD_VIDEO=${RECORD_VIDEO:-6}
VIDEO_EVERY=${VIDEO_EVERY:-10}

# Only the log dir: lerobot-train refuses to start if its output directory already exists.
mkdir -p "$LOG"
echo 400 > /proc/self/oom_score_adj

# ---- train -----------------------------------------------------------------------------
if [ -d "$OUT/checkpoints/$(printf %06d "$STEPS")" ]; then
  echo "[arm] $NAME already trained, skipping to eval"
else
  echo "[arm] training $NAME (backbone=$BACKBONE)"
  EPS=$(cat "$DS_ROOT/splits/episodes_train.json")
  "$LEROBOT_PY" -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$REPO" --dataset.root="$DS_ROOT" --dataset.episodes="$EPS" \
    --policy.type=pc_diffusion --policy.device=cuda --policy.push_to_hub=false \
    --policy.backbone="$BACKBONE" \
    --policy.goal_conditioning=both --policy.use_task_onehot=true \
    --policy.num_objects=1 --policy.aux_residual_weight=0.1 --policy.aux_predict_rotation=true \
    "${EXTRA[@]}" \
    --batch_size=$BATCH --steps=$STEPS --num_workers=$WORKERS --seed=$SEED \
    --log_freq=200 --save_freq=10000 \
    --output_dir="$OUT" --job_name="$NAME" \
    --wandb.enable=true --wandb.project=primitives_campaign > "$LOG/train_$NAME.log" 2>&1
  echo "[arm] training done"
fi

# ---- evaluate --------------------------------------------------------------------------
CKPT=$OUT/checkpoints/$(printf %06d "$STEPS")/pretrained_model
[ -d "$CKPT" ] || { echo "[arm] NO CHECKPOINT at $CKPT"; exit 1; }
mkdir -p "$OUT"

rm -f "$SOCK"
"$LEROBOT_PY" /home/samsung/lerobot_binonp/tools/pc_policy_server.py \
  --checkpoint "$CKPT" --dataset_root "$DS_ROOT" --repo_id "$REPO" \
  --socket "$SOCK" > "$LOG/server_$NAME.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; rm -f "$SOCK"' EXIT
for _ in $(seq 1 180); do
  grep -q PC_POLICY_SERVER_READY "$LOG/server_$NAME.log" 2>/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "[arm] server died"; tail -20 "$LOG/server_$NAME.log"; exit 1; }
  sleep 1
done
echo "[arm] policy server ready"

cd /home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/eval
for SPLIT in test indomain; do
  case $SPLIT in
    test)     OUTF=$OUT/eval_heldout.jsonl;  TAG=held-out ;;
    indomain) OUTF=$OUT/eval_indomain.jsonl; TAG=in-domain ;;
  esac
  LIST=$DS_ROOT/splits/objects_${SPLIT}${EVAL_VARIANT}_paths.txt
  [ -f "$LIST" ] || { echo "[arm] missing $LIST, skipping $TAG"; continue; }
  N=$(grep -c . "$LIST")
  ENVS=$((N * EVAL_ENV_MULT))
  echo "[arm] eval $TAG: $N objects x $EPISODES_PER_OBJECT rollouts, $ENVS envs"
  VIDEO_ARGS=()
  if [ "$RECORD_VIDEO" -gt 0 ]; then
    VIDEO_ARGS=(--record_video "$RECORD_VIDEO" --video_every "$VIDEO_EVERY"
                --video_dir "$OUT/videos_${SPLIT}")
  fi
  OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" eval_push_policy.py \
    --object_list "$LIST" --num_envs "$ENVS" --episodes_per_object "$EPISODES_PER_OBJECT" \
    --task "$TASK" "${VIDEO_ARGS[@]}" \
    --socket "$SOCK" --output "$OUTF" >> "$LOG/eval_${NAME}_${SPLIT}.log" 2>&1
  echo "[arm] $TAG -> $OUTF"
done
echo "ARM_DONE $NAME"

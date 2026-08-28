#!/bin/bash
# In-distribution arm: train on EVERY episode, evaluate on objects the demonstrator can do,
# with fresh goals. There is no held-out object set here by design.
#
# This is valid because the simulator samples a NEW random goal per episode: the object's initial
# pose is fixed, so the goal *is* the task variation, and at evaluation it is drawn fresh from the
# same distribution. The policy saw ~35 goals per mesh in training and is scored on goals it has
# never seen. That measures interpolation within the task distribution -- which is what matters
# for deployment on known objects -- and it is NOT evidence of generalisation to new geometry.
#
#   tools/run_arm_indist.sh <name> <backbone> <dataset_root> <repo_id> [extra --policy.* flags]
set -euo pipefail
NAME=$1; BACKBONE=$2; DS_ROOT=$3; REPO=$4; shift 4
EXTRA=("$@")
RUNS=/home/samsung/data/runs
LEROBOT_PY=/home/samsung/miniforge3/envs/lerobot/bin/python
ISAAC_PY=/home/samsung/miniforge3/envs/isaaclab/bin/python
OUT=$RUNS/$NAME
LOG=${ARM_LOG_DIR:-$OUT}
SOCK=/tmp/pc_indist_$NAME.sock
STEPS=${STEPS:-100000}; BATCH=${BATCH:-64}; WORKERS=${WORKERS:-6}; SEED=${SEED:-1000}
ROLLOUTS=${ROLLOUTS:-12}
EVAL_LIST=${EVAL_LIST:-$DS_ROOT/splits/objects_indist_screen_paths.txt}
TASK=${TASK:-push}
RECORD_VIDEO=${RECORD_VIDEO:-6}
mkdir -p "$LOG"
echo 400 > /proc/self/oom_score_adj

if [ ! -d "$OUT/checkpoints/$(printf %06d "$STEPS")" ]; then
  echo "[indist] training $NAME on ALL episodes (backbone=$BACKBONE)"
  EPS=$(cat "$DS_ROOT/splits/episodes_all.json")
  "$LEROBOT_PY" -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$REPO" --dataset.root="$DS_ROOT" --dataset.episodes="$EPS" \
    --policy.type=pc_diffusion --policy.device=cuda --policy.push_to_hub=false \
    --policy.backbone="$BACKBONE" \
    --policy.goal_conditioning=both --policy.use_task_onehot=true \
    --policy.num_objects=1 --policy.aux_residual_weight=0.1 --policy.aux_predict_rotation=true \
    "${EXTRA[@]}" \
    --batch_size=$BATCH --steps=$STEPS --num_workers=$WORKERS --seed=$SEED \
    --log_freq=200 --save_freq=10000 --output_dir="$OUT" --job_name="$NAME" \
    --wandb.enable=true --wandb.project=primitives_campaign > "$LOG/train_$NAME.log" 2>&1
  echo "[indist] training done"
fi

CKPT=$OUT/checkpoints/$(printf %06d "$STEPS")/pretrained_model
[ -d "$CKPT" ] || { echo "[indist] NO CHECKPOINT"; exit 1; }
rm -f "$SOCK"
"$LEROBOT_PY" /home/samsung/lerobot_binonp/tools/pc_policy_server.py \
  --checkpoint "$CKPT" --dataset_root "$DS_ROOT" --repo_id "$REPO" --socket "$SOCK" \
  > "$LOG/server_$NAME.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; rm -f "$SOCK"' EXIT
for _ in $(seq 1 180); do
  grep -q PC_POLICY_SERVER_READY "$LOG/server_$NAME.log" 2>/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "[indist] server died"; tail -20 "$LOG/server_$NAME.log"; exit 1; }
  sleep 1
done
N=$(grep -c . "$EVAL_LIST")
echo "[indist] eval: $N objects x $ROLLOUTS rollouts, $N envs"
cd /home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/eval
VID=()
[ "$RECORD_VIDEO" -gt 0 ] && VID=(--record_video "$RECORD_VIDEO" --video_every 10 --video_dir "$OUT/videos_indist")
OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_PY" eval_push_policy.py \
  --object_list "$EVAL_LIST" --num_envs "$N" --episodes_per_object "$ROLLOUTS" \
  --task "$TASK" "${VID[@]}" --socket "$SOCK" --output "$OUT/eval_indist.jsonl" \
  > "$LOG/eval_${NAME}_indist.log" 2>&1
echo "INDIST_DONE $NAME -> $OUT/eval_indist.jsonl"

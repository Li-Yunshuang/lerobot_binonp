#!/bin/bash
# Train the pc_diffusion_aux_v1 recipe on current code, plus optional extra --policy.* flags.
#
# aux_v1 was trained 2026-08-25 from untracked code that no longer exists, and an
# identically-configured run two days later (push_unet_oldata) scored 14.3 pp lower
# (50.0% vs 35.7% paired, p=0.003). This runner pins every setting aux_v1 used that
# now has a different default, so a rerun differs from it only by the flags passed in:
#   num_objects=1        (default is now 0 -> aux head would be absent entirely)
#   use_task_onehot=false (the field postdates aux_v1; it now defaults to true)
# It also reuses aux_v1's exact 1619-episode list rather than the full dataset.
#
#   tools/train_aux_recipe.sh <name> [extra --policy.* / --ema.* flags]
set -euo pipefail
NAME=$1; shift
EXTRA=("$@")
RUNS=/home/samsung/data/runs
LEROBOT_PY=/home/samsung/miniforge3/envs/lerobot/bin/python
DS_ROOT=/home/samsung/data/push_pc1024_poses
REPO=local/push_pc1024_poses
EPS_FILE=${EPS_FILE:?set EPS_FILE to aux_v1_episodes.json}
OUT=$RUNS/$NAME
LOG=${ARM_LOG_DIR:-$OUT}
STEPS=${STEPS:-100000}; BATCH=${BATCH:-64}; WORKERS=${WORKERS:-6}; SEED=${SEED:-1000}
# Only the log dir: lerobot_train fails with FileExistsError if its output_dir already
# exists and resume is false, so $OUT must not be pre-created here.
mkdir -p "$LOG"
echo 400 > /proc/self/oom_score_adj

if [ -d "$OUT/checkpoints/$(printf %06d "$STEPS")" ]; then
  echo "[arm] $NAME already trained; skipping"; exit 0
fi
echo "[arm] training $NAME  ($(date +%H:%M))"
"$LEROBOT_PY" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$REPO" --dataset.root="$DS_ROOT" \
  --dataset.episodes="$(cat "$EPS_FILE")" \
  --policy.type=pc_diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.goal_conditioning=points \
  --policy.use_task_onehot=false \
  --policy.num_objects=1 \
  --policy.aux_residual_weight=0.1 \
  --policy.aux_predict_rotation=false \
  "${EXTRA[@]}" \
  --batch_size=$BATCH --steps=$STEPS --num_workers=$WORKERS --seed=$SEED \
  --log_freq=200 --save_freq=20000 --output_dir="$OUT" --job_name="$NAME" \
  --wandb.enable=true --wandb.project=primitives_campaign > "$LOG/train_$NAME.log" 2>&1
echo "[arm] training done ($(date +%H:%M))"

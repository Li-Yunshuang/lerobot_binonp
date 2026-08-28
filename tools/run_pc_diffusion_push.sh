#!/usr/bin/env bash
# Reproducible pipeline for the IsaacLab bimanual-push point-cloud diffusion policy.
#
#   ./tools/run_pc_diffusion_push.sh port     # v2.1 source -> v3.0 dataset + goal recovery
#   ./tools/run_pc_diffusion_push.sh split    # held-out object split
#   ./tools/run_pc_diffusion_push.sh train    # train on the train split
#   ./tools/run_pc_diffusion_push.sh serve <ckpt>          # policy server (lerobot env)
#   ./tools/run_pc_diffusion_push.sh eval  <train|test>    # IsaacLab rollouts (isaaclab env)
#
# Two interpreters are involved and they cannot be merged: LeRobot needs Python >= 3.12 /
# numpy >= 2, IsaacSim ships CPython 3.11 / numpy 1.26. `serve` runs under the first,
# `eval` under the second, and they talk over an AF_UNIX socket.
set -euo pipefail

LEROBOT_PY=/home/samsung/miniforge3/envs/lerobot/bin
ISAAC_PY=/home/samsung/miniforge3/envs/isaaclab/bin/python
REPO=/home/samsung/lerobot_binonp
BIMANUAL=/home/samsung/3D_Bimanual_repo
SRC_DATA=$BIMANUAL/scripts_isaaclab/irregular/dataset_collection/push_data
DATASET=/home/samsung/data/push_pc1024
REPO_ID=local/push_pc1024
RUN=/home/samsung/data/runs/pc_diffusion_v1
SOCKET=/tmp/pc_policy.sock

export HF_LEROBOT_HOME=/home/samsung/data/lerobot
export HF_DATASETS_CACHE=/home/samsung/data/hf_datasets
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Isaac Sim prompts for EULA acceptance on stdin, which deadlocks a non-interactive run.
# This records the acceptance already made for this install; unset it to be prompted instead.
export OMNI_KIT_ACCEPT_EULA=YES

case "${1:-}" in
port)
  # Reads $SRC_DATA strictly read-only. Never point the v2.1->v3.0 converter at it: that
  # converter shutil.move()s its source directory.
  "$LEROBOT_PY/python" "$REPO/examples/port_datasets/port_isaaclab_pointcloud_push.py" \
    --src_root "$SRC_DATA" --dst_root "$DATASET" --repo_id "$REPO_ID" \
    --num_points 1024 --goal_num_points 512 --num_workers 6
  ;;

split)
  "$LEROBOT_PY/python" "$REPO/examples/port_datasets/make_push_object_split.py" \
    --dataset_root "$DATASET" --src_root "$SRC_DATA" --test_frac 0.2
  ;;

train)
  # num_workers=6 is a measured budget, not a guess: ~1.5 GB RSS per spawned worker on a
  # 31 GiB host, plus ~4 GB parent and ~8 GB desktop reserve. See tools/README_pc_diffusion.md.
  EPS=$("$LEROBOT_PY/python" -c "import json;print(json.dumps(json.load(open('$DATASET/splits/episodes_train.json'))).replace(' ',''))")
  "$LEROBOT_PY/lerobot-train" \
    --policy.type=pc_diffusion \
    --policy.goal_conditioning=points \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET" \
    --dataset.episodes="$EPS" \
    --dataset.eval_split=0.0 \
    --batch_size=64 \
    --num_workers=6 \
    --prefetch_factor=2 \
    --steps=100000 \
    --log_freq=200 \
    --save_freq=10000 \
    --ema.enable=true \
    --wandb.enable=true \
    --wandb.project=pc_diffusion_push \
    --job_name=pc_diffusion_goalpoints_v1 \
    --output_dir="$RUN"
  ;;

serve)
  CKPT="${2:-$RUN/checkpoints/last/pretrained_model_ema}"
  [ -d "$CKPT" ] || CKPT="$RUN/checkpoints/last/pretrained_model"
  echo "serving $CKPT"
  "$LEROBOT_PY/python" "$REPO/tools/pc_policy_server.py" \
    --checkpoint "$CKPT" --dataset_root "$DATASET" --repo_id "$REPO_ID" \
    --socket "$SOCKET" --device cuda
  ;;

eval)
  WHICH="${2:-test}"
  "$ISAAC_PY" "$BIMANUAL/scripts_isaaclab/irregular/eval/eval_push_policy.py" \
    --object_list "$DATASET/splits/objects_${WHICH}_paths.txt" \
    --episodes_per_object 10 --num_envs 20 \
    --socket "$SOCKET" \
    --output "$RUN/eval_${WHICH}.jsonl"
  ;;

*)
  sed -n '2,12p' "$0"; exit 1;;
esac

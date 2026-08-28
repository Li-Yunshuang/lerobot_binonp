#!/bin/bash
# Evaluate ONE checkpoint on ONE object list. Unlike run_arm.sh this never trains and takes the
# object list explicitly, which is what a controlled A/B on a shared object set needs.
#
#   tools/eval_checkpoint.sh <ckpt> <dataset_root> <repo_id> <object_list> <out.jsonl> [rollouts]
set -euo pipefail
CKPT=$1; DS=$2; REPO=$3; LIST=$4; OUTF=$5; ROLL=${6:-12}
TAG=$(basename "$OUTF" .jsonl)
LOG=${ARM_LOG_DIR:-$(dirname "$OUTF")}
SOCK=/tmp/pc_ab_$TAG.sock
mkdir -p "$LOG" "$(dirname "$OUTF")"
echo 400 > /proc/self/oom_score_adj
[ -d "$CKPT" ] || { echo "[ab] no checkpoint at $CKPT"; exit 1; }
N=$(grep -c . "$LIST"); ENVS=$((N * 2))

rm -f "$SOCK"
/home/samsung/miniforge3/envs/lerobot/bin/python \
  /home/samsung/lerobot_binonp/tools/pc_policy_server.py \
  --checkpoint "$CKPT" --dataset_root "$DS" --repo_id "$REPO" --socket "$SOCK" \
  > "$LOG/server_$TAG.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; rm -f "$SOCK"' EXIT
for _ in $(seq 1 180); do
  grep -q PC_POLICY_SERVER_READY "$LOG/server_$TAG.log" 2>/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "[ab] server died"; tail -20 "$LOG/server_$TAG.log"; exit 1; }
  sleep 1
done
echo "[ab] $TAG: $N objects x $ROLL rollouts, $ENVS envs"
cd /home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/eval
OMNI_KIT_ACCEPT_EULA=YES /home/samsung/miniforge3/envs/isaaclab/bin/python eval_push_policy.py \
  --object_list "$LIST" --num_envs "$ENVS" --episodes_per_object "$ROLL" \
  --task push --socket "$SOCK" --output "$OUTF" > "$LOG/eval_$TAG.log" 2>&1
echo "AB_DONE $TAG -> $OUTF"

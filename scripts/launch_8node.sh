#!/usr/bin/env bash
# Launch 8-node cold rollout collection. Run from ANY node (it ssh-es to all 8,
# including itself). Each node: 1 local vllm (Qwen3.6-27B TP=8) + its 1/8 shard,
# real e2b sandbox, CONCURRENCY parallel sessions per node.
#
# Node list (rank -> IP). Rank 0 = this current node (use its own IP or 127.0.0.1).
# Edit NODES if the cluster changes.
set -euo pipefail
ROOT_DIR=/mnt/afs_toolcall/sunhao4/workspace/agentic_cl_research

# rank0 first; these are the 8 pod IPs (rank0 = the node you launch from).
NODES=(
  10.120.5.40    # NOTE: rank0 = launching node; if you launch from a node NOT in
  10.120.5.24    # this list, prepend its IP. Below are the 7 peers + we add self.
  10.120.5.3
  10.120.4.229
  10.120.4.215
  10.120.4.195
  10.120.4.155
)
# rank0 = this node (run worker locally, no ssh); peers = the array above.
SELF_IP="${SELF_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
NUM_NODES=8
export NUM_NODES

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"
WORKER="$ROOT_DIR/scripts/_node_worker.sh"
LOGD="$ROOT_DIR/logs/cold"; mkdir -p "$LOGD"

echo "[launch] self=$SELF_IP  num_nodes=$NUM_NODES"

# rank 0 -> run locally in a tmux session (independent of this shell / ssh)
echo "[launch] rank 0 (local $SELF_IP) -> tmux 'collect'"
tmux kill-session -t collect 2>/dev/null || true
tmux new-session -d -s collect "NODE_RANK=0 NUM_NODES=$NUM_NODES bash $WORKER > $LOGD/node_r0.out 2>&1"
echo "  tmux session 'collect' started"

# rank 1..7 -> ssh to peers, each starts its own tmux session
rank=1
for ip in "${NODES[@]}"; do
  echo "[launch] rank $rank -> $ip (tmux 'collect')"
  $SSH "$ip" "tmux kill-session -t collect 2>/dev/null; tmux new-session -d -s collect 'NODE_RANK=$rank NUM_NODES=$NUM_NODES bash $WORKER > $LOGD/node_r${rank}.out 2>&1'" \
    && echo "  dispatched" || echo "  SSH FAILED to $ip"
  rank=$((rank+1))
done

echo
echo "[launch] all 8 nodes dispatched (each in its own tmux 'collect' session)."
echo "[launch] monitor:"
echo "  per-node vllm:    tail -f $LOGD/vllm_r<RANK>.log"
echo "  per-node collect: tail -f $LOGD/rollout_r<RANK>.log"
echo "  outputs:          $ROOT_DIR/data/mock/rollouts/local/rollouts_local_r<RANK>.jsonl"

#!/bin/bash

#SBATCH --job-name=inkling2-swe-rl
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=<partition>
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=/tmp/prl-render/render/launcher/logs/job_%j.log
#SBATCH --error=/tmp/prl-render/render/launcher/logs/job_%j.log

# Exit immediately on any failed command so a crash anywhere releases the
# allocation instead of leaving zombie processes holding nodes.
set -e

# Configs
export NUM_TRAIN_NODES=6
export NUM_INFER_NODES=2
export NODES_PER_INFER_REPLICA=1
export NUM_INFER_REPLICAS=2
export GPUS_PER_NODE=8
export ROUTER_PORT=8000
export INFERENCE_TP=8
export INFERENCE_DP_LOCAL=$((GPUS_PER_NODE / INFERENCE_TP))
export INFERENCE_DATA_PARALLEL_RPC_PORT=13345
export BACKEND_PORT=8100
export INFERENCE_ENABLE_EXPERT_PARALLEL=1

# Paths
export PROJECT_DIR=/Users/mkgz/Code/prime/prime-rl
export CONFIG_DIR=/tmp/prl-render/render/configs/attempt_1/resolved
export OUTPUT_DIR=/tmp/prl-render/render
export LAUNCHER_DIR=/tmp/prl-render/render/launcher
export LOG_DIR=/tmp/prl-render/render/logs/attempt_1
export PRL_ATTEMPT_CONFIG_DIR=$CONFIG_DIR
export PRL_ATTEMPT_LOG_DIR=$LOG_DIR
mkdir -p $LOG_DIR/trainer $LOG_DIR/inference
ln -sfn trainer/node_0.log $LOG_DIR/trainer.log
ln -sfn inference/node_0.log $LOG_DIR/inference.log

# Clean exits do not trigger srun --kill-on-bad-exit. Each terminal component
# publishes its successful exit through the shared launcher directory. Every node
# uses the same completion condition before it tears down support processes.
export TRAINER_DONE=$LAUNCHER_DIR/.trainer.done
export ORCHESTRATOR_DONE=$LAUNCHER_DIR/.orchestrator.done
mkdir -p "$LAUNCHER_DIR"
rm -f "$TRAINER_DONE" "$ORCHESTRATOR_DONE"

export PRL_RUN_ID=${PRL_RUN_ID:-$(python3 -c "import uuid; print(uuid.uuid4().hex)")}
export PRL_RUN_NAME=render
export WANDB_SHARED_MODE=1

# Networking
HOSTNAMES=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
TRAIN_HOSTS=( ${HOSTNAMES[@]:$NUM_INFER_NODES:$SLURM_JOB_NUM_NODES} )
export HOSTNAMES_STR="${HOSTNAMES[*]}"

INFER_HOSTS=( ${HOSTNAMES[@]:0:$NUM_INFER_NODES} )

# Single global router on inference node 0 fronts every per-rank endpoint;
# admin URLs bypass it and hit each rank directly.
INFER_URLS="http://${INFER_HOSTS[0]}:${ROUTER_PORT}/v1"
ADMIN_URLS=""
ROUTER_ARGS=""
# External-LB: one endpoint per DP rank (BACKEND_PORT + dp_local) on each node.
for ((n=0; n<NUM_INFER_NODES; n++)); do
    host="${INFER_HOSTS[$n]}"
    for ((k=0; k<INFERENCE_DP_LOCAL; k++)); do
        ADMIN_URLS="${ADMIN_URLS:+$ADMIN_URLS }http://${host}:$((BACKEND_PORT + k))/v1"
        ROUTER_ARGS="$ROUTER_ARGS http://${host}:$((BACKEND_PORT + k))"
    done
done
export INFER_URLS
export ADMIN_URLS
export ROUTER_ARGS
echo "INFER_HOSTS=${INFER_HOSTS[*]}"
echo "INFER_URLS=${INFER_URLS}"
echo "ADMIN_URLS=${ADMIN_URLS}"

echo "HOSTNAMES=${HOSTNAMES[*]}"
echo "TRAIN_HOSTS=${TRAIN_HOSTS[*]}"

export MASTER_ADDR="${HOSTNAMES[$NUM_INFER_NODES]}"
export MASTER_PORT=29500
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"



# The orchestrator runs on trainer rank 0's node (SLURM_PROCID == NUM_INFER_NODES) by
# default; with orchestrator_on_inference it moves to the last inference node (a decode
# node) to keep its rollout buffer + env workers off host-RAM-saturated trainer nodes.
# The ZMQ rollout transport binds in the orchestrator process, so both sides must
# address the orchestrator's node.
ORCH_PROCID=$NUM_INFER_NODES
export ORCH_PROCID
export ORCH_ADDR="${HOSTNAMES[$ORCH_PROCID]}"
echo "ORCH_ADDR=${ORCH_ADDR}"

# Setup environment
cd $PROJECT_DIR
[ -f .env ] && source .env
source .venv/bin/activate

# Shared (NFS) venv: one sync on the batch node populates it for all nodes.
uv sync --all-extras --all-packages
# uv sync restores the locked NIXL wheel, whose bundled UCX breaks P/D KV transfer: reinstall the source build
# into the project environment (uv pip would otherwise pick the activated .venv, not UV_PROJECT_ENVIRONMENT).
if compgen -G "deps/nixl_cu13-*.whl" > /dev/null; then uv pip install -q --reinstall --no-deps --python "${UV_PROJECT_ENVIRONMENT:-.venv}" deps/nixl_cu13-*.whl; fi

# The environment is synced above. Every later `uv run` would sync it again: the per-process syncs race, and they
# replace the source-built NIXL with the locked wheel.
export UV_NO_SYNC=1



# The run's admin token, which guards the engines' admin routes and the control APIs.
# Generated here unless the submitting environment or .env provides one, so it is
# never written to this script.
export PRL_ADMIN_TOKEN=${PRIME_RL_ADMIN_TOKEN:-$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")}


# Cleanup stale node-local state from prior jobs. Orphan python/torchrun/vllm
# processes and vLLM/torch IPC files under /dev/shm and /tmp can survive SLURM
# termination and cause the next launch to hang (e.g. decode engines stuck at
# "Waiting for READY message from DP Coordinator"). Harmless on clean nodes.
#
# `bash -s` with a heredoc on stdin keeps the script body out of every bash
# / srun argv — otherwise `pkill -f "[v]llm"` self-matches the cleanup procs
# because the surviving literal `vllm` strings below appear in the cmdline.
srun bash -s <<'CLEANUP_SH'
    pkill -9 -f "[p]ython.*prime_rl" 2>/dev/null
    pkill -9 -f "[t]orchrun" 2>/dev/null
    pkill -9 -f "[v]llm-router" 2>/dev/null
    # llm-d router procs (EPP + Envoy + decode pd-sidecar) left over from a prior run.
    pkill -9 -f "[e]pp.*pool-name" 2>/dev/null
    pkill -9 -f "[e]nvoy.*envoy.yaml" 2>/dev/null
    pkill -9 -f "[p]d-sidecar" 2>/dev/null
    pkill -9 -f "[v]llm" 2>/dev/null
    pkill -9 -f "[p]rime_rl" 2>/dev/null
    # Also match prctl-set process names (kernel comm) like "vllm::router"
    # which pkill -f does not see because -f only matches the cmdline.
    pkill -9 "vllm" 2>/dev/null
    pkill -9 "vllm::.*" 2>/dev/null
    # Mooncake store (master + client) left over from a prior offload run. -x matches
    # the exact process name, so it never hits the worktree path (e.g. mooncake-baby).
    pkill -9 -x mooncake_master 2>/dev/null
    pkill -9 -x mooncake_client 2>/dev/null

    sleep 2
    rm -rf /dev/shm/vllm-* /dev/shm/vllm_* /tmp/vllm-* /tmp/vllm_* /tmp/torch-* /tmp/torchelastic_* 2>/dev/null
    procs=$(ps -eo comm,args | grep -E "python|torchrun|vllm|vllm::|mooncake_|[e]pp |envoy|pd-sidecar" | grep -v grep | wc -l)
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{s+=\$1} END {print s}")
    echo "[node-cleanup] $(hostname) procs=$procs gpu_mem=${gpu}MiB"
CLEANUP_SH

# Run RL
# --kill-on-bad-exit=1 propagates a non-zero exit on any node to all other
# tasks so a crashed component terminates the SLURM job instead of leaving the
# surviving ranks holding nodes.
#
# The body goes to bash on stdin through a quoted heredoc instead of a
# single-quoted `bash -c` argument: srun broadcasts stdin to every task, and the
# quoted delimiter passes the script through verbatim, so apostrophes and quotes
# (including in the shared _launch_*.sh.j2 includes) need no escaping. Nothing in
# the body reads stdin, which would otherwise eat the rest of the script.
#
# The run owner registers the Prime run around the step and fails it when the
# step fails; every component attaches through the $RUN_ID it exports.
python -m prime_rl.entrypoints.run_owner rl $CONFIG_DIR/rl.json srun --kill-on-bad-exit=1 bash -s <<'LAUNCH_SH'
    # pipefail ensures `cmd | tee log` propagates `cmd` failures, and the
    # wait/cleanup block at the end of this script surfaces background
    # process crashes (router, orchestrator) as task failures.
    set -o pipefail
    # The run owner set $PRL_RUN_ID to the Prime run's id.
    export WANDB_RUN_ID=$PRL_RUN_ID

    # Source environment
    cd $PROJECT_DIR
    [ -f .env ] && source .env
    source .venv/bin/activate
    # Do not sync to avoid conflicts with lockfile

    # Keep the admin token out of the environment: only the orchestrator, which calls
    # the engines' admin routes, is started with it. The engines and the trainer's
    # control API only check it, so they get its SHA-256 digest; the env worker pools
    # and the router get neither.
    ADMIN_TOKEN=$PRL_ADMIN_TOKEN
    ADMIN_TOKEN_DIGEST=$(printf '%s' "$ADMIN_TOKEN" | sha256sum | cut -d' ' -f1)
    unset PRL_ADMIN_TOKEN PRIME_RL_ADMIN_TOKEN PRIME_RL_ADMIN_TOKEN_SHA256



    # Higher ulimit
    ulimit -n 65536

    # Infiniband setup
    IB_HCA=$(ibv_devinfo | sed -n -e "/hca_id/p" -e "/link_layer:/p" | grep -B1 InfiniBand | grep hca_id | sed -e "s/^hca_id://g" | tr -d "[[:blank:]]" |paste -sd,)
    export NCCL_IB_HCA=$IB_HCA




if [ "$SLURM_PROCID" -lt "$NUM_INFER_NODES" ]; then
    INFER_NODE_RANK=$SLURM_PROCID
    # Inference env vars: launcher defaults + [env_vars] + [inference.env_vars]
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export PYTHONUNBUFFERED="1"
    export OMP_NUM_THREADS="1"
    export GIT_LFS_SKIP_SMUDGE="1"
    export TRITON_CACHE_DIR="/tmp/triton-502"
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
    export VLLM_ENGINE_READY_TIMEOUT_S="4200"
    export UCX_TLS="all"
    export VLLM_CACHE_ROOT="/tmp/.vllm-cache-502"
    export FLASHINFER_WORKSPACE_BASE="/tmp/.flashinfer-cache-502"
    export VLLM_HTTP_TIMEOUT_KEEP_ALIVE="3600"
    LOCAL_IP=$(hostname -I | awk "{print \$1}")

    # NIXL/RDMA registers pinned host memory (KV offload pools, the HiSparse host pool).
    ulimit -l unlimited

    # Every inference node runs a replica of every env worker pool, and the orchestrator spreads
    # each source's episodes over them, so env-worker-pool memory and CPU spread over all inference
    # nodes instead of piling up next to the orchestrator. Each binds this node so the
    # orchestrator connects across nodes. They take the orchestrator env vars.
    # NOTE: keep this block free of single quotes (see the orchestrator block).
    mkdir -p $LOG_DIR/envs/train
    (
        export CUDA_DEVICE_ORDER="PCI_BUS_ID"
        export PYTHONUNBUFFERED="1"
        export OMP_NUM_THREADS="1"
        export GIT_LFS_SKIP_SMUDGE="1"
        export TRITON_CACHE_DIR="/tmp/triton-502"
        exec uv run env-worker-pool @ $CONFIG_DIR/envs/train/scaleswe.json --serve.address tcp://$LOCAL_IP:0 \
            --address-file $CONFIG_DIR/envs/train/scaleswe.$INFER_NODE_RANK.address \
            > $LOG_DIR/envs/train/scaleswe.$INFER_NODE_RANK.log 2>&1
    ) &

    REPLICA_IDX=$((INFER_NODE_RANK / NODES_PER_INFER_REPLICA))
    RANK_IN_REPLICA=$((INFER_NODE_RANK % NODES_PER_INFER_REPLICA))

    echo "INFER_NODE_RANK=$INFER_NODE_RANK REPLICA=$REPLICA_IDX RANK_IN_REPLICA=$RANK_IN_REPLICA LOCAL_IP=$LOCAL_IP" \
        | tee $LOG_DIR/inference/node_${INFER_NODE_RANK}.log

    CONFIG_PATH="$CONFIG_DIR/inference.json"
    ADMIN_TOKEN_ENV="PRIME_RL_ADMIN_TOKEN_SHA256=$ADMIN_TOKEN_DIGEST"

# rank_log <node_rank> <k> — DP rank 0 logs to the node-level file (node_<rank>.log,
# the target of the top-level inference.log symlink) so it captures real server output;
# ranks 1+ get their own node_<rank>_rank<k>.log.
rank_log() {
    if [ "$2" -eq 0 ]; then
        echo "$LOG_DIR/inference/node_$1.log"
    else
        echo "$LOG_DIR/inference/node_$1_rank$2.log"
    fi
}

# launch_inference_rank <port> <cuda_visible_devices> <dp> <rpc_port> <vllm_extra_json|""> <nixl_side_channel_port|""> <log_file>
launch_inference_rank() {
    local port="$1" gpus="$2" dp="$3" rpc="$4" extra="$5" nixl="$6" log="$7"
    # Per-rank dir for the mooncake-store lookup ipc socket. It must be unique per process:
    # external-LB runs one engine per process, all with data_parallel_rank_local 0, so the
    # lookup socket name collides on bind if the dir is shared node-wide. $port is unique per
    # rank on the node. Node-local /tmp also avoids inheriting an absent submit-node TMPDIR.
    local rpcbase="/tmp/vllm-rpc-$USER-$SLURM_JOB_ID-$port"
    mkdir -p "$rpcbase"
    local -a cmd=(uv run inference @ "$CONFIG_PATH"
        --server.host 0.0.0.0 --server.port "$port"
        --vllm.data-parallel-size "$dp" --vllm.data-parallel-size-local 1 --vllm.api-server-count 1)
    [ -n "$rpc" ] && cmd+=(--vllm.data-parallel-rpc-port "$rpc")
    # Per-rank vLLM overrides go through a JSON file merged into [inference.vllm]
    # (dict-valued keys like compilation_config can't be inlined as CLI flags).
    if [ -n "$extra" ]; then
        printf '%s' "$extra" > "$rpcbase/vllm-overrides.json"
        cmd+=(--vllm @ "$rpcbase/vllm-overrides.json")
    fi
    # ADMIN_TOKEN_ENV (NAME=value: the admin token's digest) is set by launchers that
    # guard the engines' admin routes; the engine then refuses to start without it.
    local -a rank_env=(CUDA_VISIBLE_DEVICES="$gpus" VLLM_RPC_BASE_PATH="$rpcbase")
    if [ -n "$nixl" ]; then
        rank_env+=(VLLM_NIXL_SIDE_CHANNEL_PORT="$nixl")
        # NIXL_NIC_AFFINITY=1 (P/D): a single-GPU rank sends its NIXL traffic only through the RDMA port nearest its
        # GPU, on one rail. GPUDirect RDMA through a NIC on the other CPU socket can stall the GPU under load.
        if [ "${NIXL_NIC_AFFINITY:-0}" = 1 ] && [[ "$gpus" != *,* ]]; then
            local nic
            # Same environment as the engine (`uv run`), which need not be the activated .venv.
            if nic=$(uv run --no-sync python -m prime_rl.utils.rdma "$gpus" 2>>"$log") && [ -n "$nic" ]; then
                rank_env+=(UCX_NET_DEVICES="$nic" UCX_MAX_RMA_RAILS=1)
                echo "[nixl] GPU $gpus -> $nic" | tee -a "$log"
            else
                echo "[nixl] GPU $gpus: no nearest RDMA port found, NIXL uses every port" | tee -a "$log"
            fi
        fi
    fi
    if [ -n "${ADMIN_TOKEN_ENV:-}" ]; then
        rank_env+=("$ADMIN_TOKEN_ENV")
        cmd+=(--server.require-admin-token)
    fi
    env "${rank_env[@]}" "${cmd[@]}" 2>&1 | tee -a "$log" &
    INFERENCE_PORT_BY_PID[$!]="$port"
}


# launch_router <mode:pd|regular> <worker_args> <port> <policy> <log>
launch_router() {
    local mode="$1" worker_args="$2" port="$3" policy="$4" log="$5"
    local tag="vllm-router"; [ "$mode" = pd ] && tag="vllm-router (PD)"
    echo "Starting $tag on $LOCAL_IP:$port" | tee "$log"
    local -a cmd=(vllm-router --policy "$policy" --host 0.0.0.0 --port "$port"
        --request-id-headers x-session-id
        --request-timeout-secs 14400
        --prometheus-port "$((port + 21000))"
        --worker-startup-timeout-secs 4200 --worker-startup-check-interval 1)
    if [ "$mode" = pd ]; then
        cmd+=(--vllm-pd-disaggregation $worker_args --pd-turn-retries 2)
    else
        cmd+=(--worker-urls $worker_args)
    fi
    "${cmd[@]}" >> "$log" 2>&1 &
}


# Start the single global router on inference node 0 (external LB across all per-rank endpoints)
    if [ "$INFER_NODE_RANK" -eq 0 ]; then
        launch_router regular "$ROUTER_ARGS" "$ROUTER_PORT" "sticky_least_loaded" "$LOG_DIR/inference/router.log"
    fi

    # External-LB: one vLLM API server per DP rank on this node (BACKEND_PORT + d),
    # each pinned to its TP slice of GPUs; the router balances across the per-rank
    # endpoints. For MoE the ranks form one external-LB DP group (EP all2all spans
    # them) rooted on the replica-head DP coordinator. data_parallel_address is
    # per-node, not per-rank: every rank on the head reaches the coordinator via the
    # head LOCAL_IP, ranks on other nodes via the head hostname (matches the P/D path).
    read -ra HOSTNAMES <<< "$HOSTNAMES_STR"
    REPLICA_HEAD_HOST="${HOSTNAMES[$((REPLICA_IDX * NODES_PER_INFER_REPLICA))]}"
    INFER_GROUP_DP=$((NODES_PER_INFER_REPLICA * INFERENCE_DP_LOCAL))
    if [ "$RANK_IN_REPLICA" -eq 0 ]; then DP_ADDR="$LOCAL_IP"; else DP_ADDR="$REPLICA_HEAD_HOST"; fi
    for ((d=0; d<INFERENCE_DP_LOCAL; d++)); do
        RANK_GPUS=$(seq -s, $((d * INFERENCE_TP)) $((d * INFERENCE_TP + INFERENCE_TP - 1)))
        RANK_LOG=$(rank_log "$INFER_NODE_RANK" "$d")
        if [ "$INFERENCE_ENABLE_EXPERT_PARALLEL" -eq 1 ] && [ "$INFER_GROUP_DP" -gt 1 ]; then
            GLOBAL_DP_RANK=$((RANK_IN_REPLICA * INFERENCE_DP_LOCAL + d))
            RANK_EXTRA="{\"data_parallel_rank\": $GLOBAL_DP_RANK, \"data_parallel_address\": \"$DP_ADDR\"}"
            launch_inference_rank "$((BACKEND_PORT + d))" "$RANK_GPUS" "$INFER_GROUP_DP" "$INFERENCE_DATA_PARALLEL_RPC_PORT" "$RANK_EXTRA" "" "$RANK_LOG"
        else
            # Dense or single-DP-rank EP (dp=1, e.g. EP within one tp=8 group):
            # independent single-engine servers. vLLM rejects the external-LB DP
            # args when data_parallel_size == 1; enable_expert_parallel in the
            # inference config alone shards experts across the TP group.
            launch_inference_rank "$((BACKEND_PORT + d))" "$RANK_GPUS" 1 "" "" "" "$RANK_LOG"
        fi
    done

else
    TRAIN_NODE_RANK=$((SLURM_PROCID - NUM_INFER_NODES))

    # Trainer env vars: launcher defaults + [env_vars] + [trainer.env_vars]
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export PYTHONUNBUFFERED="1"
    export OMP_NUM_THREADS="1"
    export GIT_LFS_SKIP_SMUDGE="1"
    export TRITON_CACHE_DIR="/tmp/triton-502"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export TORCH_FR_BUFFER_SIZE="2000"
    export TORCH_NCCL_DUMP_ON_TIMEOUT="1"
    export TORCH_FR_DUMP_TEMP_FILE="/tmp/nccl_trace_502_rank_"

    echo $HOSTNAMES_STR | tee $LOG_DIR/trainer/node_${TRAIN_NODE_RANK}.log

    (
        WANDB_SHARED_LABEL=trainer uv run torchrun \
            --role=trainer \
            --nnodes=$NUM_TRAIN_NODES \
            --nproc-per-node=$GPUS_PER_NODE \
            --node-rank=$TRAIN_NODE_RANK \
            --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
            --rdzv-id=job_$SLURM_JOB_ID \
            --log-dir=$LOG_DIR/trainer/torchrun \
            --tee=3 \
            --redirects=3 \
            --local-ranks-filter=0 \
            -m prime_rl.trainer.rl.train \
            @ $CONFIG_DIR/trainer.json \
            --rollout_transport.host $ORCH_ADDR \
            2>&1 | sed -u 's/^\[[a-zA-Z]*[0-9]*\]://' | tee -a $LOG_DIR/trainer/node_${TRAIN_NODE_RANK}.log
        COMPONENT_CODE=$?
        if [ "$COMPONENT_CODE" -eq 0 ] && [ "$TRAIN_NODE_RANK" -eq 0 ]; then
            touch "$TRAINER_DONE" || exit $?
        fi
        exit "$COMPONENT_CODE"
    ) &
    fi

# Launch the orchestrator on the node ORCH_ADDR points at (see the ORCH_PROCID
# computation above). Top-level (after the role if/else) so the wait -n teardown
# covers it.
if [ "$SLURM_PROCID" -eq "$ORCH_PROCID" ]; then
    ORCHESTRATOR_ARGS="@ $CONFIG_DIR/orchestrator.json"
    ORCHESTRATOR_ARGS="$ORCHESTRATOR_ARGS --model.client.base-url $INFER_URLS"
    ORCHESTRATOR_ARGS="$ORCHESTRATOR_ARGS --model.client.admin-base-url $ADMIN_URLS"
    # Orchestrator env vars from [env_vars] + [orchestrator.env_vars]
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export PYTHONUNBUFFERED="1"
    export OMP_NUM_THREADS="1"
    export GIT_LFS_SKIP_SMUDGE="1"
    export TRITON_CACHE_DIR="/tmp/triton-502"

    (
        ORCHESTRATOR_ARGS="$ORCHESTRATOR_ARGS --weight_broadcast.host $MASTER_ADDR"
        ORCHESTRATOR_ARGS="$ORCHESTRATOR_ARGS --rollout_transport.host $ORCH_ADDR"
        PRIME_RL_ADMIN_TOKEN="$ADMIN_TOKEN" WANDB_SHARED_LABEL=orchestrator uv run orchestrator $ORCHESTRATOR_ARGS \
            2>&1 | tee $LOG_DIR/orchestrator.log
        COMPONENT_CODE=$?
        if [ "$COMPONENT_CODE" -eq 0 ]; then
            touch "$ORCHESTRATOR_DONE" || exit $?
        fi
        exit "$COMPONENT_CODE"
    ) &
fi


# Every node watches the terminal components for successful completion.
# Inference runs stop after both the trainer and orchestrator finish.
( until [ -f "$TRAINER_DONE" ] && [ -f "$ORCHESTRATOR_DONE" ]; do sleep 10; done ) &
COMPLETION_PID=$!

# A clean component exit is not enough to stop the run. Keep waiting for the
# shared completion condition. Propagate any failure through
# --kill-on-bad-exit=1 so every node tears down.
# Inference servers must outlive the run, so a clean server exit is a failure.
while true; do
    wait -n -p FINISHED_PID
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -ne 0 ] || [ "${FINISHED_PID:-}" = "$COMPLETION_PID" ]; then
        break
    fi
    if [ -n "${INFERENCE_PORT_BY_PID[$FINISHED_PID]:-}" ]; then
        echo "[$(hostname)] inference server on port ${INFERENCE_PORT_BY_PID[$FINISHED_PID]} exited with code 0 before the run completed"
        EXIT_CODE=1
        break
    fi
done

REMAINING_PIDS=$(jobs -p)
# Cross-node teardown is driven by `srun --kill-on-bad-exit=1`, which reaps peer
# tasks only once a task *exits* non-zero. So on a non-zero exit (crash, SIGTERM,
# wall-time) we stay alive and signal nothing while the newest trainer checkpoint
# is still being written (no DCP `.metadata` yet), up to the grace period: this lets
# it finish on this node, and — because the task has not exited yet — keeps
# --kill-on-bad-exit from reaping checkpointing trainers on peer nodes. We then
# propagate the exit code so the allocation tears down. A clean (zero) exit cannot
# trip --kill-on-bad-exit, so we tear down immediately.
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "[$(hostname)] component exited with code $EXIT_CODE - waiting up to 3600s for an in-flight trainer checkpoint to finish before teardown"
    GRACE_DEADLINE=$((SECONDS + 3600))
    while [ "$SECONDS" -lt "$GRACE_DEADLINE" ]; do
        LATEST_CKPT=$(ls -d /tmp/prl-render/render/checkpoints/step_*/trainer 2>/dev/null | sort -V | tail -n 1)
        if [ -z "$LATEST_CKPT" ] || [ -f "$LATEST_CKPT/.metadata" ]; then
            break
        fi
        sleep 10
    done
fi
echo "[$(hostname)] tearing down (exit code $EXIT_CODE) - terminating remaining processes"
for pid in $REMAINING_PIDS; do
    kill -TERM $pid 2>/dev/null || true
done
exit $EXIT_CODE
LAUNCH_SH

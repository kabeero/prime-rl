# Remote inference: running inference + envs in another datacenter

Status: revised after builder review · Branch: `exp/wan-weight-xfer` · Date: 2026-11-23

## 1. Intent

prime-rl runs trainer, orchestrator, inference (vLLM) and env worker pools inside one datacenter, joined by a
shared filesystem and (optionally) RDMA. We want to run **inference + env worker pools in a different
datacenter** (first provider: Modal), reached over ordinary TCP across the internet, while the **trainer and
orchestrator stay together** in the home datacenter (SLURM on Nebius `us-central1`).

Minimally invasive to existing launch and transport code, but shaped to grow: more providers, a direct-push
weight path instead of an object store, sparse weight deltas, later LoRA.

User experience target: take an existing example TOML (e.g. `examples/advanced/qwen3-30b-a3b/math.toml`), set
`inference.deployment.type = "remote"`, and `uv run rl @ ...` brings up the remote machines, prepares them, runs
training, and **always** tears them down.

**Demo target**: 1 SLURM training node (8×H200) + 1 Modal inference node (8×H200, `us-central`), Qwen3-30B-A3B,
the settings of `_demo/slurm/run.sh`. (The Modal workspace caps concurrent GPUs at 10, so 2 remote nodes are not
possible until Modal raises it.)

## 2. Goals, non-goals, anti-goals

Goals
- `inference.deployment.type = "remote"` is the single switch; everything that follows from it is inferred
  (weight transport, local inference node count) or rejected with a clear error.
- The launcher provisions the remote nodes (Modal clustered Function, RDMA-capable), prepares them (code,
  `uv sync`, Tailscale, model cache), starts every remote process, and streams their logs into the run's `$LOG_DIR`.
- A WAN weight transport: trainer ranks upload to **Nebius Object Storage** (S3-compatible) with parallel
  multipart uploads; each remote node pulls in parallel into local staging; vLLM loads from the local path.
- **Asynchronous weight delivery**: the trainer never waits for the upload, the remote download, or the remote
  load; inference switches to the newest staged version with only a short local load pause (§10.2).
- A startup network check (tailnet reachability, direct vs relayed, iperf3, object-store throughput).
- Remote GPUs always shut down: success, training failure, `scancel`/time limit, and SLURM side dying silently.
- Phase 3: sparse deltas computed on the trainer, applied at remote staging, full sends as the safe default.

Non-goals (this design)
- LoRA over the remote transport (must stay possible, §12); speculative-decoding drafts; disaggregated P/D;
  replicas spanning >1 node; `orchestrator_on_inference`; `rollout_transport.retain`; non-SLURM trainer side;
  SFT/eval entrypoints; the llm-d router; in-place vLLM weight swaps without an engine pause; direct push (Phase 4);
  providers other than Modal; AWS S3.

Anti-goals
- Orphaned remote GPUs after any failure mode.
- Secrets (Tailscale, Modal, object store, HF, admin token) in configs, rendered scripts, or logs.
- Env-worker→vLLM rollout traffic leaving the remote cluster.
- Any behaviour change for existing (non-remote) runs: rendered SLURM scripts for existing examples must be
  byte-identical.
- A delta path that can silently leave inference on wrong weights.

## 3. How the system works today (facts the design relies on)

- **Weight transports**: sender/receiver pairs in `src/prime_rl/transports/weights/`, selected by
  `weight_broadcast.type` (union at `packages/prime-rl-configs/src/prime_rl/configs/shared.py:102`; factories in
  `transports/weights/__init__.py`). `rl` copies the shared `[weight_broadcast]` into trainer and orchestrator and
  sets `inference.weight_broadcast.type` (`configs/rl.py:444-497`); trainer/orchestrator conflict checks ignore
  `launcher_set = {"host", "inference_world_size"}` (`rl.py:472`).
- **Handshake** (`transports/weights/base.py`): per version the trainer master resets the step dir, touches
  `.sender_ready`; the orchestrator touches `.receiver_ready` (`base.py:184-186`, an empty file; the sender only
  checks existence, `base.py:110`); trainer touches `.started`, runs `_broadcast`, then `.finished`. The trainer
  blocks until the receiver acks (`weight_broadcast.timeout`, default 3600 s). Markers live on the filesystem the
  trainer and orchestrator share — kept.
- **Filesystem transport**: `gather_weights_parallel` leaves each rank a disjoint slice of the HF-format bf16 state
  dict on CPU (`utils/weights.py:197`); `save_state_dict_parallel` writes per-rank safetensors shards, master
  writes the index last. The receiver waits for `.finished`, then `admin_plane.update_weights(dir,
  transport="filesystem")` (`orchestrator/clients.py:249`, `transport: Literal["filesystem","nccl","nixl"]`) pauses
  engines, POSTs `/update_weights`, resumes. vLLM side: `FileSystemWeightUpdateWorker`, picked by
  `WORKER_EXTENSION_CLS[config.weight_broadcast.type]` (`inference/vllm/server.py:53,291`).
- **Startup**: the trainer always broadcasts a startup version (`trainer/rl/train.py:672-678`, v0 on a fresh run);
  the orchestrator blocks in `watcher.sync_startup` (`orchestrator/orchestrator.py:443`).
- **Watcher** (`orchestrator/watcher.py:79-121`): on a published version it sets `ckpt_step`, calls observers'
  `on_version_pending` (the dispatcher sets `policy_update_pending = True` and blocks train scheduling,
  `orchestrator/dispatcher.py:456-466`), calls `receiver.receive(step)`, sets `policy.version`, then
  `on_new_version` (dispatcher resumes scheduling, `dispatcher.py:468-472`). `gauges()` (line 133) feed the
  orchestrator monitors.
- **Connectivity**:
  - orchestrator → vLLM: HTTP (`client.base_url` → router; `client.admin_base_url` → each engine; admin routes
    guarded by the run admin token, engines check `$<var>_SHA256`, `utils/admin_token.py`);
  - orchestrator → env pool: ZMQ DEALER→ROUTER, pool binds (`env/pool.py:104`), orchestrator connects
    (`env/client.py:94`); addresses discovered from `$CONFIG_DIR/envs/<split>/<name>.<rank>.address`
    (`orchestrator/envs.py:95`);
  - env workers → vLLM: HTTP with the client config the orchestrator embeds in each episode request
    (`orchestrator/envs.py:140-146`, `env/worker.py:145-147`);
  - **trainer ranks → env workers**: each env worker runs a `PayloadServer` (HTTP, random port) holding per-token
    training arrays (routing rows for router replay, sampler support/logprobs); trainer ranks `GET`/`DELETE` blobs
    at URLs carried in `trace.info` (`transports/payload.py:1-20, 196-252`). The advertised host comes from
    `payload_host(serve.address)`: for a wildcard bind it is `_routable_address()` — the node's own IP.
- **Multi-node SLURM** (`templates/multi_node_rl.sbatch.j2`): batch-level script runs
  `python -m prime_rl.entrypoints.run_owner rl … srun … bash -s <<'LAUNCH_SH'` **in the foreground** (line 254);
  the heredoc body runs on every node. First `$NUM_INFER_NODES` hosts are inference nodes; the per-rank vLLM loop
  and router args live in the template (lines 118-125, 531-560); `_launch_rank.sh.j2` / `_launch_router.sh.j2`
  only define functions. `$NUM_INFER_NODES` (exported line 32) is used as a count and as an offset (87, 138, 160,
  305, 366, 564). Eleven `{% if num_infer_nodes > 0 %}` gates: 36, 75, 79, 90, 160, 364, 599, 600, 631, 643, 652.
  Node cleanup block: lines 212-239.
- **Config couplings**: `validate_shared_configs` requires `deployment.gpus_per_node ==
  inference.deployment.gpus_per_node` (`rl.py:375-381`); `auto_setup_deployment` derives inference DP and
  `api_server_count` from it (`rl.py:749-755`); `InferenceConfig` validators branch on deployment type
  (`configs/inference.py:640, 827`, `auto_setup_slurm_template`); `write_pd_configs` (`entrypoints/rl.py:191`).
- **Staleness**: `orchestrator.max_staleness` (default 8).
- FP8 inference in the examples is vLLM online quantization; trainer wire format is bf16 regardless.
- `boto3` and `modal` are in `uv.lock` transitively; add both as direct dependencies.
- **Existing Modal tooling** (`../modal-utils`, outside this repo): image `poc-h200-cu129` (published named image
  mirroring the SLURM H200 nodes: Ubuntu 24.04, CUDA 12.9, Python 3.12, uv 0.11.28, ibverbs), Volume
  `poc-work-$USER` (v2) at `/mnt/work` for uv/HF caches, a clustered-Function pattern
  (`@modal.clustered(size=N, rdma=True)`, `gpu="H200:8"`, `efa_disabled`, region `us-central` ≈ 12 ms from the
  SLURM nodes) that registers each node in a `modal.Dict`, deploy-then-`spawn()`, shut down via
  `FunctionCall.cancel(terminate_containers=True)` + `modal app stop`. Gotcha: never put Volume paths in image ENV
  (the builder creates them, the mount then fails silently).

## 4. Approaches considered

| Decision | Options | Picked | Why |
|---|---|---|---|
| Primary knob | `weight_broadcast.type` · `inference.deployment.type` | `inference.deployment.type = "remote"` | Placement is the intent; transport follows. Fits `single_node / multi_node / disaggregated`. |
| Weights across the WAN | Direct TCP push · object store · per-node pulls from trainer | **Nebius Object Storage** (S3 API); direct push later | Outbound-only on both sides; no dependence on Modal's userspace Tailscale throughput; multipart + ranged GETs give parallel chunking; storage sits next to the SLURM nodes and ~12 ms from Modal `us-central`. |
| Applying weights remotely | Patch vLLM tensors · stage an HF dir, reuse the filesystem load path | Stage an HF dir | vLLM params are TP-sharded/fused; filesystem path handles it; LoRA stays possible. |
| Full vs delta decision | Trainer asks remote · orchestrator relays | Orchestrator relays via `.receiver_ready` | One owner of remote state + load decision; trainer reads one file. |
| Modal control | CLI · Python SDK | SDK, deployed named app per run, clustered Function (modal-utils pattern) | Programmatic handles; RDMA needs clustered Functions; a *named deployed* app makes `modal app stop <name>` reliable from a shell trap. |
| Node prep | `sleep` + `exec` · setup in the function body | Function body: bootstrap then watchdog | No exec-session lifetime question; gang semantics fail the whole cluster on any node's failure. `modal shell` stays for debugging. |
| Overlay network | Tailscale · WireGuard · Modal tunnels | Tailscale, **all SLURM nodes of the job join** | NAT traversal, per-node identity, ephemeral nodes. Trainer ranks must reach env workers' payload servers. |
| Tailscale on SLURM | System install · static binaries on shared FS, per-job daemon | Static binaries + per-job `sudo tailscaled` (kernel mode) | No changes to shared nodes; jobs have `sudo` and `/dev/net/tun`. |
| `uv sync` remotely | Shared venv on a Volume · per-node venv + shared uv cache | Per-node venv, uv cache on the Volume | Nodes sync in parallel; downloads are the cost; no torch/vLLM imports off a network FS. |
| WAN transfer vs training | Synchronous upload + fetch inside `receive` · fully asynchronous pipeline | Async: background per-rank upload, orchestrator-side stager, watcher only sees staged versions (§10.2) | The trainer must never wait on the WAN; only the short local vLLM load pauses anything. |

## 5. Configuration surface

```toml
[deployment]
type = "multi_node"            # required with remote inference
num_train_nodes = 1            # SLURM allocation = trainer nodes only
num_infer_nodes = 1            # remote: number of Modal machines

[inference.deployment]
type = "remote"
gpus_per_node = 8              # must equal deployment.gpus_per_node (existing check)
bringup_timeout_s = 3600       # fleet manager: all remote processes healthy within this
heartbeat_timeout_s = 600      # remote watchdog exits if the fleet manager is silent this long
env_base_port = 7000           # remote env pool ports: env_base_port + index
agent_port = 8300
forward_env = ["OPENAI_API_KEY"]   # extra .env/environment keys shipped to remote env pools (names only)

[inference.deployment.provider]
type = "modal"
image = "poc-h200-cu129"       # Modal named image (modal.Image.from_name)
gpu = "H200"                   # → gpu="H200:{gpus_per_node}"
rdma = true
efa_disabled = true
region = "us-central"
volume = "poc-work-{user}"     # uv + HF caches, mounted at /mnt/work
timeout_s = 86400              # hard cap for the clustered function (max 24 h)

[inference.deployment.tailscale]
tag = "tag:prime-rl"
version = "1.104.7"            # static binaries, downloaded only when ~/.local/bin/tailscale is missing
require_direct = false         # fail (instead of warn) when a peer path is relayed (DERP)

[weight_broadcast]             # inferred when unset
type = "remote"
store = "s3"                   # S3 API; Phase 4 adds "direct"
endpoint_url = "https://storage.us-central1.nebius.cloud"
region = "us-central1"
uri = "s3://<bucket>/<prefix>"  # required
credentials_env_prefix = "NEBIUS"   # reads NEBIUS_ACCESS_KEY_ID / NEBIUS_SECRET_ACCESS_KEY
multipart_chunk_mb = 128
max_concurrency = 32            # per trainer rank / per remote node
```

Validators and inference (in `packages/prime-rl-configs/`):
- `RemoteInferenceDeploymentConfig` (`type: Literal["remote"]`) joins `InferenceDeploymentConfig`
  (`configs/inference.py:498`) with a nested provider union (`ModalProviderConfig`). No node count field:
  `deployment.num_infer_nodes` is the single source.
- `RemoteWeightBroadcastConfig` joins `WeightBroadcastConfig` (`shared.py:102`).
  `InferenceWeightBroadcastConfig.type` (`inference.py:210`) gains `"remote"`; `WORKER_EXTENSION_CLS["remote"] =
  FileSystemWeightUpdateWorker`.
- `auto_setup_weight_broadcast` (`rl.py:445`): remote inference ⇒ default `RemoteWeightBroadcastConfig`, error if
  `uri` unset, reject other explicit types; reject `type="remote"` without remote inference. Reject `"remote"` in
  SFT/eval/standalone-trainer configs (`configs/supervised.py:289`, `configs/trainer.py:853,866`).
- `auto_setup_infer_nodes` (`rl.py:252`): remote branch — `num_infer_nodes` required ≥1, not inferred.
- Reject with remote: `deployment.type != "multi_node"`, `num_infer_replicas != 1`, `orchestrator_on_inference`,
  speculative decoding, LoRA, `rollout_transport.retain`, env sources with `serve.address`, Dynamo, router type
  `llm-d`.
- Make every `InferenceConfig` validator that branches on deployment type handle `"remote"` like `"multi_node"`
  with one node per replica (`inference.py:640, 827`, `auto_setup_slurm_template`; `write_pd_configs` skips it).
- New orchestrator field `model.client.agent_urls: list[str] | None` (launcher-set, like `admin_base_url`); it is
  **not** on `weight_broadcast` (the trainer does not need it).
- `.env` is not loaded by `rl`; the submit-time secret check (§7.2) reads the environment and parses `.env`.

## 6. Topology and connectivity

```
home DC (SLURM, Nebius us-central1)                    remote DC (Modal us-central, 1 node for the demo)
┌───────────────────────────────────────┐ tailnet      ┌──────────────────────────────────────────────┐
│ trainer node 0 (+ orchestrator,       │ (TCP)        │ node 0                                       │
│   fleet manager)                      │              │  router :server.port                         │
│ orchestrator ─ HTTP base_url ─────────┼─────────────►│  vLLM   :backend_port+k                      │
│              ─ HTTP admin ────────────┼─────────────►│  env pools :env_base_port+i (ZMQ)            │
│              ─ ZMQ ───────────────────┼─────────────►│   └ env workers' payload servers :random     │
│              ─ HTTP agent ────────────┼─────────────►│  node agent :agent_port, iperf3 :5201        │
│ trainer ranks (all nodes) ─ HTTP ─────┼─────────────►│   (payload GET/DELETE)                       │
│ fleet manager ─ Modal API (HTTPS) ────┼──────────────│  env workers → router over cluster private IP│
│ trainer ranks ── multipart PUT ──► Nebius Object Storage ◄── ranged GET ── node agents          │
└───────────────────────────────────────┘              └──────────────────────────────────────────────┘
```

- Every connection starts on the home side, except object-store GETs from the remote side. No inbound ports in
  the home DC.
- **Every SLURM node in the job joins the tailnet** (trainer ranks fetch payload blobs from remote env workers).
- URLs handed to home-side processes use **tailnet IPs (100.x)**, never MagicDNS (the per-job `tailscaled` must not
  touch the host resolver).
- Env workers call the router at `http://<rank0 cluster-private IPv4>:<server.port>/v1` (§9). The router reaches
  engines on other remote nodes by cluster-private IP.

## 7. Components

### 7.1 Networking (Tailscale)

Credentials (`.env` / environment only): `TS_OAUTH_CLIENT_ID` + `TS_OAUTH_CLIENT_SECRET` — an OAuth client with
scopes `auth_keys` (write) and `devices:core` (write), both restricted to `tag:prime-rl`. Tailnet policy: `tagOwners`
entry for `tag:prime-rl`; grant `tag:prime-rl → tag:prime-rl` on all ports (narrow later).

Binaries: Tailscale's static Linux tarball. On SLURM the binaries are `~/.local/bin/tailscale` and
`~/.local/bin/tailscaled` (home is on the shared FS). The batch-level script — which runs once, before `srun`, so
nodes never race on the download — checks for `~/.local/bin/tailscale`; if it is missing (or `tailscaled` is), it
downloads the pinned `version` into `~/.local/tailscale/<version>/`, verifies the published sha256, extracts, and
symlinks both binaries into `~/.local/bin/`. If present, it uses them as-is and logs `tailscale version`. On Modal, the base image itself carries `tailscale`, `tailscaled` (in `/usr/local/bin`) and `iperf3`:
`../modal-utils/sandbox.py:build_image()` installs them (pinned version, sha256-verified), and the image is
rebuilt with `./msh build` (`TS_VERSION = "1.104.7"`). The fleet manager uses the image as-is. Both sides follow
the same rule: use the binaries if present, download the pinned version only if missing. On Modal the bootstrap
downloads into `/usr/local/bin` (sha256-verified) and logs a warning to rebuild the image; `iperf3` missing only
skips that preflight check.

Join (both sides): `tailscale up --auth-key="$TS_OAUTH_CLIENT_SECRET?ephemeral=true&preauthorized=true"
--advertise-tags=tag:prime-rl --hostname=prl-<run_id[:12]>-<t|i><rank>`, passing the key via
`--auth-key=file:<path>` (mode 0600, deleted after) so it never appears in `ps`.

- **SLURM, every node of the job** (inside the `srun` heredoc, before any component starts): kernel mode via
  `sudo -n ~/.local/bin/tailscaled --tun=prl<short> --statedir=<node-local tmp> --socket=<statedir>/sock`
  (interface name ≤15 chars); `tailscale up … --operator=$USER` so later `status`/`ping` need no sudo. Fail fast if
  `sudo -n true` fails. Do not pass `--accept-dns`. Teardown in the node's exit path: `tailscale logout`, then
  stop the daemon. The node-cleanup block (lines 212-239) also kills a leftover `tailscaled` whose statedir matches
  the `prl-` prefix from a previous job on the node.
- **Modal**: userspace mode (`tailscaled --tun=userspace-networking`, `--statedir=/tmp/ts`). Inbound tailnet
  connections reach local listeners, which covers every remote service, including payload servers on random
  ports. No outbound tailnet traffic is needed remotely.
- **Preflight** (fleet manager, before the orchestrator starts; results logged and written to
  `$LOG_DIR/remote/network.json`):
  1. all expected devices online (device API with the OAuth client) — fatal if not;
  2. `tailscale ping -c 3` home node 0 ↔ each remote node: record `direct` vs `DERP`; DERP → warning, fatal if
     `require_direct`;
  3. `iperf3` home→remote node 0, 1 and 8 streams, 10 s each;
  4. object store: upload 1 GiB from home with the configured multipart settings; download it on remote node 0
     with ranged GETs; report Gbit/s.

### 7.2 Fleet manager (home side)

New `[project.scripts]` entrypoint `remote-fleet @ <config_dir>/remote_fleet.json` (pydantic-config), launched by
the template on `ORCH_PROCID` (trainer node 0) as a peer component covered by the existing `wait -n` teardown.

1. **Secrets check** (also done by `rl` at submit time): `TS_OAUTH_CLIENT_ID`, `TS_OAUTH_CLIENT_SECRET`,
   `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `<prefix>_ACCESS_KEY_ID` / `<prefix>_SECRET_ACCESS_KEY` (default
   `NEBIUS_*`: a Nebius static access key; passed explicitly to boto3 so no `AWS_*` variable or `~/.aws` config is
   ever consulted), `HF_TOKEN` if the model is gated, every name in
   `forward_env`, and the admin token.
2. **Leftover sweep**: stop the app recorded in `$LAUNCHER_DIR/remote_app.json` and any deployed app named
   `prl-<run.name>`.
3. **Deploy + spawn**: build the app (`prl-<run.name>`), `app.deploy()`, then `.spawn()` the clustered function
   (§7.4); write `{app_name, app_id, function_call_id}` to `$LAUNCHER_DIR/remote_app.json` immediately (the batch
   trap and leftover sweep read it). Inputs: the resolved `inference.json`, the env-pool configs, the rendered node
   script, run id/name, ports. Secrets: one `modal.Secret.from_dict` with the Tailscale secret, object-store keys,
   `HF_TOKEN`, `forward_env` values, and **only the admin token's digest** (`$<var>_SHA256`). Code: the project
   checkout via `add_local_dir(copy=False)` excluding `.venv`, `outputs`, `.git`, `_demo` (ships exactly what the
   SLURM side runs, uncommitted changes included).
4. **Discover**: nodes register `{rank, task_id, tailnet_ip, cluster_ipv4, cluster_ipv6}` in a run-scoped
   `modal.Dict` (modal-utils pattern); then poll each agent's `/health` over the tailnet.
5. **Preflight network checks** (§7.1) and **readiness**: every engine `/health`, router `/health`, env pools
   answering, all within `bringup_timeout_s` (the error must name the remote bring-up, not the weight broadcast).
6. **Publish endpoints** (atomic tmp+rename): `$CONFIG_DIR/remote/endpoints.env` with `INFER_URLS`,
   `ADMIN_URLS`, `AGENT_URLS` (tailnet IPs), and env-pool address files
   `$CONFIG_DIR/envs/<split>/<name>.<rank>.address = tcp://<tailnet ip>:<port>` (existing orchestrator discovery).
7. **Run**: `POST /heartbeat` to every agent every 60 s; pull logs (§7.6); if the function call finishes or raises,
   exit non-zero.
8. **Shutdown** on every exit path (§7.5).

### 7.3 Node agent (remote, one per node)

`remote-node-agent` (FastAPI/uvicorn, like the inference server), on `agent_port`, guarded by the admin-token
digest.

| Route | Purpose |
|---|---|
| `GET /health` | `{rank, tailnet_ip, cluster_ip, ready, processes: {name: alive}}` |
| `POST /heartbeat` | refresh the watchdog deadline |
| `GET /logs/index`, `GET /logs?name=&offset=` | log shipping (§7.6) |
| `GET /weights/state` | `{"id": installed id or null, "step": int or null}` |
| `POST /weights/fetch` | `{id, step, prefix, kind, base_id}` → download (Phase 3: apply) into `/staging/<id>/`, verify, `{path}`; long-running |
| `POST /weights/commit` | `{id}` → mark installed, delete other staging dirs |
| `POST /weights/abort` | `{id}` → delete `/staging/<id>/` |

Staging on local NVMe, identical path on every node. Each node pulls from the object store independently.

### 7.4 Remote bootstrap (the clustered function)

```python
@app.function(image=modal.Image.from_name(image), gpu=f"{gpu}:{gpus_per_node}", timeout=timeout_s,
              volumes={"/mnt/work": volume}, secrets=[run_secret], region=region,
              experimental_options={"efa_disabled": True} if efa_disabled else None)
@modal.clustered(size=num_infer_nodes, rdma=rdma)
def remote_node(bundle: RemoteBundle) -> None: ...
```

Per node (rank from `modal.Cluster.from_context()`):
1. Start `tailscaled` (userspace) and join; register in the run `modal.Dict`.
2. Environment at runtime (not image ENV): `UV_CACHE_DIR=/mnt/work/.cache/uv`, `HF_HOME=/mnt/work/.cache/huggingface`,
   `UV_PROJECT_ENVIRONMENT=/opt/prl-venv`, `UV_LINK_MODE=copy`. Rank 0 runs `uv sync --all-extras --all-packages
   --frozen` and pre-downloads `model.name`, then `sync /mnt/work`; other ranks wait on a TCP barrier from rank 0
   over the cluster network, `reload()` the volume, then sync. `export UV_NO_SYNC=1`. No NIXL wheel reinstall.
3. Run `remote_inference_node.sh`, rendered from a new template. **Extract the regular (non-disaggregated)
   per-node engine loop and router-arg construction** (`multi_node_rl.sbatch.j2:118-125, 531-560`) into a new
   partial `_launch_infer_node.sh.j2`, included by both the SLURM template and the remote template, together with
   the existing `_launch_rank.sh.j2` / `_launch_router.sh.j2`. The remote template provides the variables they
   read (`LOG_DIR`, `CONFIG_PATH`, `CONFIG_DIR`, ports, `INFERENCE_DP_LOCAL`, host list = cluster-private IPs, a
   stand-in `SLURM_JOB_ID`). Processes start detached (`setsid`), outputs under `/logs/` mirroring `$LOG_DIR`:
   `inference/node_<rank>.log` (+ per-rank), `router.log` (rank 0), `envs/<split>/<name>.<rank>.log`,
   `remote/{agent,tailscaled,bootstrap}.<rank>.log`.
   Env pools: `env-worker-pool @ envs/<split>/<name>.json --serve.address tcp://0.0.0.0:<env_base_port+i>
   --serve.inference-base-url http://<rank0 cluster ipv4>:<server.port>/v1 --serve.advertise-host <tailnet ip>`
   with `--address-file` pointed at a node-local path (the home path in the written config is meaningless remotely),
   plus the resolved `orchestrator_env_vars`, `PRL_RUN_ID`, `PRL_RUN_NAME`, `forward_env`.
4. Start the node agent and `iperf3 -s`.
5. Watchdog: loop while the heartbeat deadline holds and critical processes (engines, router on rank 0, env pools,
   agent) are alive; otherwise log why and raise (Modal ends the whole cluster call).

### 7.5 Guaranteed shutdown (layered)

1. **Fleet manager**: `try/finally` + SIGTERM/SIGINT → `FunctionCall.cancel(terminate_containers=True)`, stop the
   app, delete the run `modal.Dict`.
2. **Batch script**: the foreground `run_owner … srun` must become `… & wait $!` so traps run promptly; an
   EXIT/TERM trap runs `modal app stop -y prl-<run.name>` (and cancels the call id from
   `$LAUNCHER_DIR/remote_app.json` if present), tolerating "not found". `#SBATCH --signal=B:TERM@120` when remote.
3. **Watchdog** on Modal: exit after `heartbeat_timeout_s` without a heartbeat.
4. **Hard cap**: function `timeout`.
5. **Leftover sweep** at fleet-manager start.

### 7.6 Logs

The fleet manager polls each agent's `/logs/index` and `/logs?name=&offset=` every ~1 s and appends into the same
relative paths under the local `$LOG_DIR`; one final pull before shutdown. Modal's function logs keep bootstrap
stdout for when the agent never comes up.

### 7.7 SLURM template and launcher changes

- Export `NUM_LOCAL_INFER_NODES` (= `num_infer_nodes`, or 0 for remote) and use it at every offset site (87, 138,
  160, 305, 366, 564, and the `#SBATCH --nodes` line 4). Keep `NUM_INFER_NODES` as the count.
- Introduce `remote_inference` and decide each `num_infer_nodes > 0` gate explicitly:

| Line | Today | Remote |
|---|---|---|
| 36 | inference/router setup vars | skip |
| 75, 79 | `ORCHESTRATOR_DONE` marker | keep (orchestrator runs) |
| 90 | build `INFER_HOSTS`/URLs from SLURM hosts | skip; URLs come from `endpoints.env` |
| 160 | orchestrator placement | `ORCH_PROCID = 0` (trainer node 0) |
| 364 / 599 | inference node branch / its `fi` | skip |
| 600 | launch orchestrator | keep; first wait for `$CONFIG_DIR/remote/endpoints.env`, source it, pass base/admin/agent URLs; also launch `remote-fleet` on this node |
| 631, 643, 652 | completion waits on orchestrator / inference | keep orchestrator; replace inference with the fleet manager |

- Inside the heredoc, every node starts its per-job `tailscaled` (§7.1) before any component.
- Trap + backgrounded launch + `--signal` (§7.5).
- `entrypoints/rl.py`: pass `num_local_infer_nodes` and `remote_inference`; `write_subconfigs` also writes
  `remote_fleet.json` and keeps writing env-pool configs; export object-store credentials to every trainer node
  (they are in `.env`, which every node sources).
- Acceptance check: render the existing multi-node examples before/after; non-remote output byte-identical.

## 8. Payload servers over the tailnet

Add optional `serve.advertise_host` to the env worker pool config. `payload_host(bind_address, advertise_host)`
returns `("0.0.0.0", advertise_host)` when set (`transports/payload.py:196`). The remote bootstrap sets it to the
node's tailnet IP. Trainer ranks (on the tailnet) then reach every worker's `PayloadServer` directly. Measure blob
fetch time per step in the Phase 2 run (MoE routing rows can be large; this crosses Modal's userspace Tailscale).

## 9. Env workers use an internal inference address

Optional `serve.inference_base_url` on the env worker pool config: when set, the pool replaces `base_url` in each
request's client config before building the episode client (`env/worker.py:145-147`), keeping everything else. The
remote bootstrap always sets it (rank 0 cluster-private IPv4 + `server.port`); unset everywhere else.

## 10. Remote weight transport

### 10.1 Version identity
Each broadcast attempt gets a fresh `id` (uuid4 hex) from the trainer master, broadcast to ranks
(`utils/distributed.broadcast_bytes`). Steps are not unique across resumes, so deltas key off `id`.

### 10.2 Asynchronous pipeline: training never waits on the WAN

Three stages run concurrently, connected by the broadcast-dir markers and the object store:

1. **Trainer (synchronous part, seconds):** gather → convert to HF → each rank writes its slice as safetensors
   shards (≤5 GB, same split helper as `save_state_dict_parallel`) to **node-local tmp**, hands them to its
   background uploader, and **returns to training**.
2. **Trainer upload (background, per rank):** multipart-upload the shards with boto3 (`endpoint_url`, `region`,
   explicit Nebius keys, `TransferConfig(multipart_chunksize, max_concurrency)`) to
   `<uri>/<id>/rank<r>-*.safetensors`, delete the tmp files, touch `.uploaded.rank<r>` in the step dir (shared FS
   of the trainer nodes). The master's uploader waits for every rank's marker, then writes `manifest.json`
   (tensor → shard, dtype, shape, per-tensor xxh3-128; HF index + config/tokenizer files vLLM's loader needs), the
   `.remote_version` marker (`{id, step, prefix, kind, base_id}`), and finally `.finished` — all atomic
   tmp+rename.
3. **Remote staging (orchestrator-driven, background):** a stager task in `RemoteWeightReceiver` acks every offer
   promptly and has every agent download finished versions into `/staging/<id>/`. A version counts as
   **published** to the watcher only once it is staged on every node.

The watcher keeps its shape: it picks the newest staged version, pauses train scheduling (`on_version_pending`),
loads it from local NVMe (`admin_plane.update_weights(/staging/<id>, transport="filesystem")`), commits on the
agents, resumes (`on_new_version`). The only pauses left are the vLLM engines' local load (seconds) and the
orchestrator holding new train rollouts across it; the trainer never waits for the upload, the download, or the
load.

```
trainer:   [train n]─[gather+write tmp]─[train n+1]─[gather+write tmp]─[train n+2] ...
uploader:                 └─[upload v(n)]─.finished        └─[upload v(n+1)]─.finished
stager:    ack v(n) ─────────────────────[agents fetch v(n)]─staged    [fetch v(n+1)] ...
watcher:                                                   └[pause · load v(n) · resume]
```

Required changes and rules:
- **Base class split** (`transports/weights/base.py`): `WeightSender.broadcast` keeps the handshake but delegates
  the commit to a new `_commit(step_dir)` hook that by default touches `.finished` synchronously (all existing
  transports unchanged); the remote sender defers it to the master's uploader.
- **Backpressure:** at most one version uploading per rank. `broadcast(n+1)` first waits for the upload of `n`
  (normally already done), so tmp disk holds at most two versions. Time spent waiting is logged as
  `weights/remote/upload_wait_time`; it should be ~0, otherwise the link is slower than a training step.
- **`_clean`** never deletes a step dir or object-store prefix whose upload or staging is in progress; it deletes
  object-store prefixes older than the newest staged version minus one. Document a bucket lifecycle rule as a
  backstop.
- **Stager** (started by `RemoteWeightReceiver.initialize`; its failure must crash the orchestrator): acks every
  offer (re-offers after a trainer restart included, as in `filesystem.py:75-84`); when several versions are
  finished and unstaged, it stages only the newest (safe with full sends; Phase 3 changes this, §11).
  `is_published` / `next_version` / `wait_published` are overridden to mean "staged on all agents", so
  `on_version_pending` never fires before the bytes are local.
- **Ack content** (`.receiver_ready`, atomic write): `{"base": <id of the newest version staged or being staged on
  every agent, or null>}`. The sender reads it after the wait and passes it to `_broadcast` (optional argument;
  other transports ignore it). Phase 2 ignores the content.
- **Agent fetch:** read `manifest.json`, download shards concurrently with ranged GETs, verify hashes, write
  index/config files.

### 10.3 Startup and the Phase 1 stub
The startup broadcast (`train.py:678`) and `sync_startup` use the same path (full send). Startup may block the
trainer until the version is loaded; there is nothing to train on before it. **Phase 1** (the build stage before
the object store exists, §11) uses `RemoteStartupOnlyReceiver`: acknowledges the startup version without loading
(vLLM already serves `model.name`) and raises on any later version. Valid only for fresh runs from HF weights
(`init.type != "scratch"`, no resume); validate that.

### 10.4 Failures, staleness, metrics
- A staging failure (download/verify) on any agent: `POST /weights/abort` on all; the version is marked failed and
  never published to the watcher; inference keeps serving the previous version; scheduling never pauses for it.
  The stager moves on to the next finished version (in Phase 3 that one must be a full send; the ack carries
  `base = null`).
- A staging failure for the startup version is fatal.
- Any `update_weights` error is fatal (`/update_weights` does not report which engines loaded).
- Consecutive staging failures ≥ `max_staleness // 2` → warning; ≥ `max_staleness` → fail the run.
- Metrics: trainer — `weights/remote/bytes_sent`, `weights/remote/upload_time`, `weights/remote/upload_wait_time`,
  `weights/remote/full_sends_total`, `weights/remote/changed_fraction` (Phase 3). Orchestrator
  (`WeightWatcher.gauges`) — `watcher/remote/stage_time` (max over nodes), `watcher/remote/staging_lag` (newest
  finished step minus newest staged step), `watcher/remote/delta_failures_total`,
  `watcher/remote/consecutive_failures`, `watcher/remote/skipped_versions_total`, plus existing
  pause/transfer/resume.
- Fault injection for acceptance: an agent env var (`PRL_AGENT_FAIL_FETCH_STEP=<n>`) read only in the agent.

## 11. Phases and acceptance

"Phase" means a build stage of this feature (what gets implemented and verified in what order), not a training
step.

**Milestone 0 — throwaway probe (`_demo/`).** One Modal container (`poc-h200-cu129`, rebuilt with Tailscale) and one
SLURM node: kernel-mode Tailscale on SLURM via static binaries; inbound TCP to a Modal service by tailnet IP;
direct vs DERP; iperf3 1/8 streams; Nebius Object Storage upload from SLURM / ranged download on Modal.

**Phase 1 — remote fleet.** Config surface (§5), template + launcher (§7.7), fleet manager, bootstrap, agent
(health/heartbeat/logs), Tailscale on all nodes, preflight, logs, shutdown layers, §8, §9, startup-only receiver.
Acceptance: with the demo config the orchestrator connects to all engines and env pools, the trainer completes
step 1 on remote rollouts (including payload fetches from remote env workers), and the run then stops at the v1
broadcast, which the startup-only receiver rejects by design; logs from every remote process appear under
`$LOG_DIR`; rendered non-remote scripts are byte-identical; trainer failure, `scancel`, and `kill -9` of the fleet
manager leave no running Modal app within ~1 min, ~2 min, `heartbeat_timeout_s` respectively.

**Phase 2 — full sends via Nebius Object Storage.** §10 with `kind = "full"`. Acceptance: the demo trains ≥30
steps; per-step upload/stage/load times and bytes in metrics; `weights/remote/upload_wait_time` ≈ 0 (the trainer
never waits on the WAN) and train scheduling is paused only for the local load;
after one update, `get_weight_fingerprint` on the remote engines matches a second load of the same staged
directory through the filesystem path on the same engines (same TP layout); injected fetch failure skips one
version, scheduling continues, the next version is a full send.

**Phase 3 — deltas.** Trainer keeps the previous bf16 HF slice per rank (pinned CPU; `partition_weights` is
deterministic for a fixed world size — verify). Per tensor: compare `view(torch.int16)`; sparse (positions + new
values) or dense when changed fraction > threshold (e.g. 25%). Full send when base is null/mismatched, cache empty,
every `full_send_interval` versions, or after any skip. Agent applies onto a copy of the installed version,
verifies hashes, installs. With deltas on, the stager stages **every** version in order (deltas are small; a
delta needs its base), while the watcher still loads only the newest staged one; the trainer bases each delta on
its previous upload and sends a delta only if the ack's `base` equals that upload's id. Acceptance: fingerprints identical to a full send; bytes per step track
`changed_fraction`.

**Phase 4 (TODO) — direct push.** `store = "direct"`: trainer ranks push over the tailnet to remote rank 0, which
fans out over the cluster network/RDMA. Same agent API and protocol.

## 12. Must stay possible
- **LoRA**: the wire unit is "a directory the agent materializes into a local path"; never write into vLLM tensors
  directly.
- **Other providers**: Modal code behind an interface — `deploy(bundle) → handle`, `stop()`, `is_running()`,
  node registry.
- **FP8 on the wire**: manifest carries dtype per tensor.
- **More remote nodes**: nothing assumes one node except the GPU quota.

## 13. Testing (per AGENTS.md: conservative)
Unit-test isolated logic only: remote config accept/reject matrix; template render diff for existing examples;
manifest + (Phase 3) delta encode/apply round trip; object-store key layout helpers; `payload_host` with
`advertise_host`. No tests that mock Modal or Tailscale.

## 14. Open items
- **User**: Nebius bucket name/prefix and a static access key (`NEBIUS_ACCESS_KEY_ID` /
  `NEBIUS_SECRET_ACCESS_KEY` in `.env`) with read/write on it, usable from both SLURM and Modal (Nebius bucket access policy).
- **User**: confirm `sudo -n true` works inside a batch job (not just interactive).
- **User**: Tailscale policy + OAuth client set up (`tag:prime-rl`), secrets in `.env`.
- ~~`poc-h200-cu129` rebuilt with Tailscale 1.104.7 + iperf3~~ — done (`../modal-utils`, `./msh build`,
  final image `im-30EskidUqSauY2093fsBy6`).
- **Builder's calls**: exact defaults for ports and timeouts, poll intervals, hash library, shard sizes, the rank-0
  barrier mechanism, the Tailscale version pin. Note each in the PR.

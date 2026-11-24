# modal-utils: persistent H200 dev box on Modal

A detached Modal Sandbox (`sleep infinity`) you attach to with `modal shell`.
It survives disconnects and lives until `./msh down` or its `--hours` cap (max 24h).
The image mirrors the `prime-slurm` H200 nodes (Ubuntu 24.04, CUDA 12.9 devel,
Python 3.12.3, uv 0.11.28, ibverbs/openmpi, tmux/nvtop/jq/vim...), plus static
`tailscale`/`tailscaled` 1.104.7 in `/usr/local/bin` and `iperf3` for prime-rl remote inference.

Requires `uv` and an authenticated `modal` CLI locally.

```bash
./msh build                 # once (and after editing the image in sandbox.py)
./msh up                    # 8x H200, 24h. Prints sandbox id
./msh shell                 # attach; Ctrl-D detaches, box keeps running
./msh exec 'nvidia-smi'     # one-off command
./msh down                  # STOP PAYING
```

Options: `./msh up --gpus 1 --hours 2`, `--idle-minutes 60` (cost guard, but it also
kills `nohup` jobs once no shell is attached), `--cpu`, `--memory-gb`, `--region`.

## Where state lives

| Path | Persists? | Use for |
|---|---|---|
| `/mnt/work` (volume `poc-work-$USER`, v2) | yes, across sandboxes | repos, checkpoints, data, uv + HF caches |
| everything else (`/root`, `/usr`, apt installs) | only via `./msh snapshot` | system tweaks |

* Run `sync /mnt/work` to commit volume writes right away. Writes are also committed every few seconds in the background and when the sandbox stops.
* `./msh snapshot` saves the root FS as an image (30-day TTL), recorded in `.state.json`.
  `./msh up --snapshot` boots from the latest one.
* `./msh push <local> [remote]` / `./msh pull <remote> [local]` / `./msh ls` move files in and out of the volume.

## Differences vs the slurm nodes

* GPU sandboxes **can be preempted**: Modal only guarantees no preemption for CPU sandboxes. Keep anything you care about in `/mnt/work`.
* Runs as root on gVisor (not `kabeer` on a k8s pod). No Docker, no kernel modules.
* Host driver is Modal's (620.x vs 580.x on the nodes). CUDA 12.9 user space is the same.
* CUDA 13.0 toolkit is optional: `POC_WITH_CUDA13=1 ./msh build` (separate image name).
* No InfiniBand in a single sandbox. One sandbox = one node (max 8 GPUs). Multi-node
  needs Modal's `@modal.clustered(size=N, rdma=True)` Functions, not sandboxes.
* CPU and RAM are Modal defaults that scale with the GPU count (`nproc` showed 17 with 1× H200; `free` inside gVisor isn't a reliable measure of allocated memory).
  The nodes have 116 usable cores and 1.5 TB RAM. Pass `--cpu` and `--memory-gb` if you need to match them.

## RDMA gang: `./msh cluster ...` (`cluster.py`)

```bash
./msh cluster up --nodes 2 --region us-central --hours 4   # N× H200:8 + RDMA, plus a CPU box
./msh cluster ls                     # rank, task id, ipv4/ipv6, cloud, IB devices
./msh cluster shell node0            # or node1 / cpu
./msh cluster exec all 'nvidia-smi -L | wc -l'
./msh cluster test                   # ibv_devinfo + cross-node NCCL all_reduce (torch via uv)
./msh cluster down
```

* **Nodes are `@modal.clustered(size=N, rdma=True)` Function containers.** Each one registers itself in a `modal.Dict` and then sleeps. GPU sandboxes can't use RDMA or the private network.
  Every node takes a whole 8-GPU host, nodes are started together, and if one fails or is preempted, all of them restart.
  The shell profile on each node sets `NODE_RANK NNODES MASTER_ADDR MASTER_PORT NODE_IPS` plus Modal's `NCCL_*` variables.
* **The CPU box is a CPU Function with `i6pn=True` in the same region**, with `NODE_IPS6` and `MASTER_ADDR6` in its profile.
  Tested: two CPU i6pn Functions in `us-central` reach each other. A CPU *sandbox* with i6pn (via the experimental
  `Sandbox._experimental_create`) lands on a different private network and **could not** connect to a Function.
* **Region: `us-central` is closest to the Slurm nodes** (Nebius us-central1, Kansas City), measured as TCP connect time to
  `storage.us-central1.nebius.cloud`: us-central ≈ 12 ms, us-east ≈ 32 ms, us-west ≈ 47 ms. Pinning to a narrow region costs 1.75×.
  This workspace can't pin a cloud (`cloud=` is rejected by the plan).
* **Blocker:** the workspace caps concurrent GPUs at **10**, so a 2×8 cluster is rejected at deploy:
  `Cluster GPU count cannot exceed Workspace's max concurrent GPUs (10)`. Ask Modal to raise it to at least 16,
  plus 8 more if you also want the devbox running. The cross-node RDMA path is untested until then.

## Gotcha (already handled)

Don't put volume paths in the image `ENV`. Modal's builder runs uv with the image env,
which creates e.g. `/mnt/work/.cache` inside the image. A volume can't mount over a
non-empty dir, so the sandbox dies instantly (rc=1, no output). Those vars are set at
runtime via `RUNTIME_ENV` in `sandbox.py`.

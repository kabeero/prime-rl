# /// script
# requires-python = ">=3.10"
# dependencies = ["modal>=1.6"]
# ///
"""Gang of RDMA-connected GPU nodes (+ optional CPU box) on Modal, for manual work.

Modal only gives RDMA through `@modal.clustered` *Functions* (GPU sandboxes can't
join the private network). So each node is a clustered Function container that
just registers itself and sleeps; you attach with `modal shell <task-id>`.

The CPU box is a plain CPU Function with i6pn=True in the same region. (CPU
*sandboxes* get i6pn only via the experimental V2 backend, which lands on a
different private network than Functions and can't reach the cluster.)

  up     [--nodes 2] [--gpu H200:8] [--region us-central] [--hours 4] [--no-cpu-box]
  ls     show nodes: rank, task id, IPs, cloud, IB devices
  id     <node0|node1|...|cpu|0|1>  -> task id (for `modal shell`)
  test   ibv_devinfo + 2-node NCCL all_reduce over RDMA (installs torch in /tmp)
  down   cancel everything

Config is read from env at import time (so the container sees the same values):
POC_NODES POC_GPU POC_REGION POC_HOURS POC_CPU_BOX POC_CPU_CORES POC_CPU_MEM_GB
POC_RDMA POC_EFA_DISABLED POC_IMAGE POC_VOLUME POC_USER.  `up --flag` sets these.
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
import time

import modal

E = os.environ.get
CFG = {
    "POC_USER": E("POC_USER") or getpass.getuser(),
    "POC_NODES": E("POC_NODES", "2"),
    "POC_GPU": E("POC_GPU", "H200:8"),  # clustered fns must take every GPU on the host
    "POC_REGION": E("POC_REGION", "us-central"),  # ~12ms to Nebius us-central1 (KC)
    "POC_HOURS": E("POC_HOURS", "4"),  # hard lifetime; max 24
    "POC_CPU_BOX": E("POC_CPU_BOX", "1"),
    "POC_CPU_CORES": E("POC_CPU_CORES", "16"),
    "POC_CPU_MEM_GB": E("POC_CPU_MEM_GB", "64"),
    "POC_RDMA": E("POC_RDMA", "1"),
    "POC_EFA_DISABLED": E("POC_EFA_DISABLED", "1"),  # force InfiniBand verbs hosts (like Nebius), not AWS EFA
    "POC_IMAGE": E("POC_IMAGE", "poc-h200-cu129"),  # built by `msh build`
}
CFG["POC_VOLUME"] = E("POC_VOLUME", f"poc-work-{CFG['POC_USER']}")

USER = CFG["POC_USER"]
APP_NAME = f"poc-cluster-{USER}"
NODES = int(CFG["POC_NODES"])
WORK = "/mnt/work"
PROFILE = "/etc/profile.d/poc-cluster.sh"
TEST_SCRIPT = "/tmp/poc_allreduce.py"

app = modal.App(APP_NAME)
registry = modal.Dict.from_name(f"poc-cluster-{USER}", create_if_missing=True)
image = modal.Image.from_name(CFG["POC_IMAGE"])
volume = modal.Volume.from_name(CFG["POC_VOLUME"], create_if_missing=True, version=2)
# Pass the config into the containers so the module re-import there sees the same values.
cfg_secret = modal.Secret.from_dict({**CFG, "UV_CACHE_DIR": f"{WORK}/.cache/uv", "HF_HOME": f"{WORK}/.cache/huggingface"})
common = dict(image=image, volumes={WORK: volume}, secrets=[cfg_secret], region=CFG["POC_REGION"],
              timeout=int(float(CFG["POC_HOURS"]) * 3600))

ALLREDUCE = r'''
import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.ones(256 * 1024 * 1024, dtype=torch.bfloat16, device="cuda")  # 512 MiB
for _ in range(5): dist.all_reduce(x)
torch.cuda.synchronize(); t = time.perf_counter(); n = 20
for _ in range(n): dist.all_reduce(x)
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / n
busbw = x.numel() * x.element_size() * 2 * (w - 1) / w / dt / 1e9
if r == 0: print(f"all_reduce 512MiB world={w}: {dt*1e3:.2f} ms  busbw={busbw:.1f} GB/s", flush=True)
dist.destroy_process_group()
'''


# ---------------------------------------------------------------------------
# Container side
# ---------------------------------------------------------------------------
def _me() -> dict:
    import socket
    return {
        "task": os.environ.get("MODAL_TASK_ID"),
        "ipv6": socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0],
        "region": os.environ.get("MODAL_REGION"),
        "cloud": os.environ.get("MODAL_CLOUD_PROVIDER", "").replace("CLOUD_PROVIDER_", ""),
        "started": time.strftime("%H:%M:%S"),
    }


def _write_profile(env: dict) -> None:
    lines = [f"export {k}={v!r}" for k, v in env.items()]
    with open(PROFILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open("/root/.bashrc", "a") as f:  # `modal shell` starts a non-login bash
        f.write(f"\n[ -f {PROFILE} ] && . {PROFILE}\n")


def _sleep_forever():
    while True:
        time.sleep(3600)


@app.function(gpu=CFG["POC_GPU"],
              experimental_options={"efa_disabled": True} if CFG["POC_EFA_DISABLED"] == "1" else None,
              **common)
@modal.clustered(size=NODES, rdma=CFG["POC_RDMA"] == "1")
def node():
    c = modal.Cluster.from_context()
    rank = c.container_rank()
    ip4, ip6 = c.container_ips(family="ipv4"), c.container_ips()
    ib = sorted(os.listdir("/sys/class/infiniband")) if os.path.isdir("/sys/class/infiniband") else []
    net_env = {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "UCX_", "FI_", "OMPI_"))}
    _write_profile({
        "NODE_RANK": rank, "NNODES": len(ip4), "MASTER_ADDR": ip4[0], "MASTER_PORT": 29500,
        "NODE_IPS": " ".join(ip4), "NODE_IPS6": " ".join(ip6),
        "UV_CACHE_DIR": f"{WORK}/.cache/uv", "HF_HOME": f"{WORK}/.cache/huggingface", **net_env,
    })
    with open(TEST_SCRIPT, "w") as f:
        f.write(ALLREDUCE)
    registry[f"node{rank}"] = {**_me(), "rank": rank, "ipv4": ip4[rank], "ib": ib, "nccl_env": net_env}
    _sleep_forever()


@app.function(cpu=float(CFG["POC_CPU_CORES"]), memory=int(CFG["POC_CPU_MEM_GB"]) * 1024, i6pn=True, **common)
def cpu_box():
    registry["cpu"] = {**_me(), "rank": "cpu"}
    # once the GPU nodes have registered, drop their addresses into the shell profile
    for _ in range(360):
        nodes = [registry.get(f"node{i}") for i in range(NODES)]
        if all(nodes):
            _write_profile({"NODE_IPS6": " ".join(n["ipv6"] for n in nodes),
                            "MASTER_ADDR6": nodes[0]["ipv6"],
                            "UV_CACHE_DIR": f"{WORK}/.cache/uv", "HF_HOME": f"{WORK}/.cache/huggingface"})
            break
        time.sleep(10)
    _sleep_forever()


# ---------------------------------------------------------------------------
# Local side
# ---------------------------------------------------------------------------
def entries() -> dict:
    return {k: v for k, v in registry.items() if k.startswith(("node", "cpu"))}


def print_table() -> None:
    rows = entries()
    if not rows:
        print("nothing registered (yet)")
        return
    for k in sorted(rows):
        v = rows[k]
        print(f"{k:6s} {v['task']}  {v.get('cloud', '?'):6s} {v.get('region', '?'):11s} "
              f"ipv4={v.get('ipv4', '-'):15s} ipv6={v['ipv6']}  ib={len(v.get('ib', [])) if 'ib' in v else '-'} "
              f"up@{v['started']}")


def resolve(target: str) -> str:
    key = f"node{target}" if target.isdigit() else target
    v = entries().get(key)
    if not v:
        sys.exit(f"unknown/unregistered target {target!r}; have: {', '.join(sorted(entries())) or 'none'}")
    return v["task"]


def cmd_up(a) -> None:
    if entries():
        sys.exit("cluster already registered; `msh cluster down` first (or `ls`)")
    print(f"deploying {APP_NAME}: {NODES}x {CFG['POC_GPU']} rdma={CFG['POC_RDMA']} region={CFG['POC_REGION']} "
          f"hours={CFG['POC_HOURS']} cpu_box={CFG['POC_CPU_BOX']}", file=sys.stderr)
    with modal.enable_output():
        app.deploy(name=APP_NAME)
    registry.clear()
    calls = {"node": modal.Function.from_name(APP_NAME, "node").spawn().object_id}
    if CFG["POC_CPU_BOX"] == "1":
        calls["cpu"] = modal.Function.from_name(APP_NAME, "cpu_box").spawn().object_id
    registry["_calls"] = calls
    want = [f"node{i}" for i in range(NODES)] + (["cpu"] if CFG["POC_CPU_BOX"] == "1" else [])
    t0 = time.time()
    while True:
        have = entries()
        missing = [w for w in want if w not in have]
        if not missing:
            break
        if time.time() - t0 > a.wait_minutes * 60:
            print(f"still waiting for {missing} after {a.wait_minutes} min (capacity?). "
                  "It keeps trying in the background: re-check with `msh cluster ls`, or `down`.", file=sys.stderr)
            break
        print(f"  [{int(time.time() - t0):4d}s] waiting for {missing}", file=sys.stderr)
        time.sleep(15)
    print_table()


def cmd_ls(_) -> None:
    print_table()


def cmd_id(a) -> None:
    print(resolve(a.target))


def cmd_test(_) -> None:
    nodes = sorted(k for k in entries() if k.startswith("node"))
    if len(nodes) < 2:
        sys.exit("need >= 2 registered nodes")
    print("== RDMA devices / NCCL env per node")
    for k in nodes:
        v = entries()[k]
        print(f"{k}: ib={v['ib']}\n      nccl_env={v['nccl_env']}")
    cmd = (f". {PROFILE}; export UV_CACHE_DIR=/tmp/uvcache NCCL_DEBUG=INFO; "
           "ibv_devinfo -l 2>&1 | head -3; "
           "uv run -q --python 3.12 --with torch torchrun --nnodes=$NNODES --nproc-per-node=gpu "
           f"--node-rank=$NODE_RANK --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT {TEST_SCRIPT} 2>&1 "
           "| grep -E 'all_reduce|NET/IB|NET/Socket|Using network|NCCL version|rror' | sort -u | head -20")
    procs = {k: subprocess.Popen(["modal", "container", "exec", entries()[k]["task"], "bash", "-c", cmd],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for k in nodes}
    for k, p in procs.items():
        out, _ = p.communicate()
        print(f"== {k}\n{out}")


def cmd_down(_) -> None:
    calls = registry.get("_calls") or {}
    for name, cid in calls.items():
        try:
            modal.FunctionCall.from_id(cid).cancel(terminate_containers=True)
            print(f"cancelled {name} ({cid})")
        except Exception as e:  # noqa: BLE001
            print(f"cancel {name}: {e}", file=sys.stderr)
    subprocess.run(["modal", "app", "stop", "-y", APP_NAME], check=False)
    registry.clear()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up")
    up.add_argument("--nodes", type=int)
    up.add_argument("--gpu")
    up.add_argument("--region")
    up.add_argument("--hours", type=float)
    up.add_argument("--no-cpu-box", action="store_true")
    up.add_argument("--cpu-cores", type=int)
    up.add_argument("--no-rdma", action="store_true")
    up.add_argument("--wait-minutes", type=int, default=20)
    i = sub.add_parser("id")
    i.add_argument("target")
    for n in ("ls", "test", "down"):
        sub.add_parser(n)
    a = p.parse_args()

    if a.cmd == "up":
        # flags -> env, then re-exec so the module-level decorators pick them up
        overrides = {"POC_NODES": a.nodes, "POC_GPU": a.gpu, "POC_REGION": a.region, "POC_HOURS": a.hours,
                     "POC_CPU_CORES": a.cpu_cores,
                     "POC_CPU_BOX": "0" if a.no_cpu_box else None, "POC_RDMA": "0" if a.no_rdma else None}
        overrides = {k: str(v) for k, v in overrides.items() if v is not None}
        if a.hours is not None and a.hours > 24:
            p.error("--hours max is 24")
        if overrides:
            os.environ.update(overrides)
            os.execv(sys.executable, [sys.executable, __file__, "up", "--wait-minutes", str(a.wait_minutes)])
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()

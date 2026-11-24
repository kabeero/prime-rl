# /// script
# requires-python = ">=3.10"
# dependencies = ["modal>=1.6"]
# ///
"""Long-lived GPU dev box on Modal, mirroring the prime-slurm H200 nodes.

The sandbox is kept alive by `sleep infinity`, not by your terminal, so you can
disconnect and re-attach (`./msh shell`) until you `down` it or it hits --hours.

Reference node (prime-nebius-puku-h200-gpu-*, probed 2026-11-23):
  Ubuntu 24.04.5, kernel 6.11 (host), 8x H200 141GB, driver 580.173.02
  CUDA 12.9 default (CUDA_HOME=/usr/local/cuda-12.9), CUDA 13.0 also installed
  Python 3.12.3 (system), uv 0.11.28, gcc 13.3, git 2.43, tmux, nvtop, jq, vim
  ibverbs / rdmacm / openmpi, 2x32c (116 usable), ~1.5 TB RAM

Commands (normally invoked through ./msh):
  build                 build + publish the named image (do this once / after edits)
  up                    start the named sandbox (no-op if already running)
  id                    print the running sandbox id (used by `msh shell`)
  status                show sandboxes in the app
  snapshot              snapshot the root filesystem -> reusable image id
  down                  terminate the sandbox
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config. Everything overridable via env so the shell wrapper stays dumb.
# ---------------------------------------------------------------------------
USER = os.environ.get("POC_USER", getpass.getuser())
APP_NAME = os.environ.get("POC_APP", "poc-devbox")
SANDBOX_NAME = os.environ.get("POC_NAME", f"devbox-{USER}")
VOLUME_NAME = os.environ.get("POC_VOLUME", f"poc-work-{USER}")
WITH_CUDA13 = os.environ.get("POC_WITH_CUDA13", "0") == "1"
IMAGE_NAME = os.environ.get("POC_IMAGE", "poc-h200-cu129" + ("-cu130" if WITH_CUDA13 else ""))

WORK = "/mnt/work"  # volume mountpoint (persistent). NB: must not pre-exist in the image
STATE_FILE = Path(__file__).with_name(".state.json")

CUDA_TAG = "12.9.1-devel-ubuntu24.04"  # matches node: Ubuntu 24.04 + CUDA 12.9
UV_VERSION = "0.11.28"  # matches node
TS_VERSION = "1.104.7"  # static tailscale/tailscaled; keep in sync with the slurm-side ~/.local/bin/tailscale

# apt packages that are explicitly installed on the slurm nodes
APT_NODE = [
    "build-essential", "ca-certificates", "curl", "wget", "git", "gnupg",
    "python3", "python3-dev", "python3.12", "python3.12-dev",
    "tmux", "vim", "nvtop", "jq", "rsync", "pigz", "socat",
    "openssh-client", "iproute2", "iputils-ping", "procps", "tzdata",
    "ibverbs-providers", "ibverbs-utils", "libibverbs1", "librdmacm1t64",
    "openmpi-bin", "libopenmpi3t64", "libnuma1", "libhwloc15", "libucx0",
]
# not on the nodes, but handy / needed for Modal (python on PATH, pip)
APT_EXTRA = [
    "python-is-python3", "python3-pip", "python3-venv",
    "git-lfs", "htop", "less", "unzip", "zstd", "lsof", "strace",
    "iperf3",  # network preflight between slurm and modal nodes
]

CUDA_HOME = "/usr/local/cuda-12.9"


def build_image() -> modal.Image:
    apt = APT_NODE + APT_EXTRA + (["cuda-toolkit-13-0"] if WITH_CUDA13 else [])
    return (
        modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}")
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC", "LANG": "C.UTF-8"})
        .apt_install(*apt)
        .run_commands(
            f"curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh"
            " | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh",
            "uv --version && python3 --version && nvcc --version | tail -1",
        )
        .run_commands(
            "set -e; cd /tmp"
            f" && TGZ=tailscale_{TS_VERSION}_amd64.tgz"
            ' && curl -fsSLO "https://pkgs.tailscale.com/stable/$TGZ"'
            ' && echo "$(curl -fsSL https://pkgs.tailscale.com/stable/$TGZ.sha256)  $TGZ" | sha256sum -c -'
            ' && tar xzf "$TGZ" --strip-components=1 -C /usr/local/bin'
            f" tailscale_{TS_VERSION}_amd64/tailscale tailscale_{TS_VERSION}_amd64/tailscaled"
            ' && rm "$TGZ"',
            "tailscale version && iperf3 --version | head -1",
        )
        .dockerfile_commands(
            [
                f"ENV CUDA_HOME={CUDA_HOME}",
                f"ENV PATH={CUDA_HOME}/bin:$PATH",
                f"ENV LD_LIBRARY_PATH={CUDA_HOME}/lib64:$LD_LIBRARY_PATH",
                # copy mode: uv cache (volume) and venvs may be different FSes
                "ENV UV_LINK_MODE=copy",
                "ENV UV_PYTHON_PREFERENCE=system",
                "ENV PYTHONUNBUFFERED=1",
            ]
        )
    )


# Paths on the volume are set at *runtime*, not as image ENV: Modal's image
# builder runs uv with the image env, which would create e.g. /mnt/work/.cache
# inside the image -- and a volume can't be mounted over a non-empty dir
# (the sandbox then dies instantly with rc=1 and no output).
RUNTIME_ENV = {
    "UV_CACHE_DIR": f"{WORK}/.cache/uv",
    "HF_HOME": f"{WORK}/.cache/huggingface",
}


def app() -> modal.App:
    return modal.App.lookup(APP_NAME, create_if_missing=True)


def volume() -> modal.Volume:
    return modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


def running() -> modal.Sandbox | None:
    try:
        return modal.Sandbox.from_name(APP_NAME, SANDBOX_NAME)
    except modal.exception.NotFoundError:
        return None


def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_build(_: argparse.Namespace) -> None:
    with modal.enable_output():
        build_image().build(app()).publish(IMAGE_NAME)
    print(f"published image '{IMAGE_NAME}'")


def cmd_up(a: argparse.Namespace) -> None:
    if sb := running():
        print(f"already running: {sb.object_id}  (name={SANDBOX_NAME})", file=sys.stderr)
        print(sb.object_id)
        return

    if a.snapshot:
        snap = a.snapshot if a.snapshot != "latest" else load_state().get("latest_snapshot")
        if not snap:
            sys.exit("no snapshot recorded in .state.json; run `msh snapshot` first")
        image = modal.Image.from_id(snap)
        src = f"snapshot {snap}"
    else:
        image = modal.Image.from_name(IMAGE_NAME)
        src = f"image {IMAGE_NAME}"

    gpu = f"{a.gpu}:{a.gpus}" if a.gpus else a.gpu
    print(f"starting {SANDBOX_NAME}: gpu={gpu} from {src}, volume {VOLUME_NAME} -> {WORK}, "
          f"timeout={a.hours}h" + (f", idle_timeout={a.idle_minutes}m" if a.idle_minutes else ""),
          file=sys.stderr)
    sb = modal.Sandbox.create(
        "sleep", "infinity",
        app=app(),
        name=SANDBOX_NAME,
        image=image,
        gpu=gpu,
        cpu=a.cpu,
        memory=a.memory_gb * 1024 if a.memory_gb else None,
        volumes={WORK: volume()},
        env=RUNTIME_ENV,
        workdir=WORK,
        timeout=int(a.hours * 3600),
        idle_timeout=a.idle_minutes * 60 if a.idle_minutes else None,
        region=a.region,
    )
    sb.set_tags({"owner": USER, "purpose": "poc-devbox"})
    print(sb.object_id)


def cmd_id(_: argparse.Namespace) -> None:
    sb = running()
    if not sb:
        sys.exit(f"no running sandbox named {SANDBOX_NAME} (run `msh up`)")
    print(sb.object_id)


def cmd_status(_: argparse.Namespace) -> None:
    a = app()
    found = False
    for sb in modal.Sandbox.list(app_id=a.app_id):
        found = True
        tags = sb.get_tags()
        print(f"{sb.object_id}  owner={tags.get('owner', '?')}  {sb.get_dashboard_url()}")
    if not found:
        print(f"no running sandboxes in app {APP_NAME}")


def cmd_snapshot(_: argparse.Namespace) -> None:
    sb = running()
    if not sb:
        sys.exit("nothing running to snapshot")
    print("snapshotting root filesystem (volume at /mnt/work is NOT included)...", file=sys.stderr)
    img = sb.snapshot_filesystem(timeout=600)
    state = load_state()
    state["latest_snapshot"] = img.object_id
    state.setdefault("snapshots", []).append(img.object_id)
    save_state(state)
    print(img.object_id)


def cmd_reload(_: argparse.Namespace) -> None:
    if sb := running():
        sb.reload_volumes()


def cmd_down(_: argparse.Namespace) -> None:
    sb = running()
    if not sb:
        print("nothing running")
        return
    sb.terminate(wait=True)
    print(f"terminated {sb.object_id}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    up = sub.add_parser("up")
    up.add_argument("--gpu", default="H200")
    up.add_argument("--gpus", type=int, default=8, help="GPU count (max 8 per node)")
    up.add_argument("--hours", type=float, default=24, help="hard lifetime, max 24")
    up.add_argument("--idle-minutes", type=int, default=None,
                    help="auto-stop when no shell/exec is attached. Careful: kills nohup'd jobs too")
    up.add_argument("--cpu", type=float, default=None, help="reserved cores (default: Modal's per-GPU default)")
    up.add_argument("--memory-gb", type=int, default=None)
    up.add_argument("--region", default=None)
    up.add_argument("--snapshot", nargs="?", const="latest", default=None,
                    help="boot from a filesystem snapshot (id, or omit value for latest)")
    for name in ("id", "status", "snapshot", "reload", "down"):
        sub.add_parser(name)
    a = p.parse_args()
    if getattr(a, "hours", None) and a.hours > 24:
        p.error("--hours max is 24 (Modal sandbox limit)")
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()

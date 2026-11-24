# /// script
# requires-python = ">=3.10"
# dependencies = ["modal>=1.6"]
# ///
"""Milestone 0 probe, Modal side: one container that joins the tailnet in userspace mode
and serves a few TCP listeners for the SLURM side (slurm_probe.sh) to reach.

  up     deploy + spawn, wait for the container, print its tailnet IP
  ls     show the registered container
  logs   print tailscaled's log tail and `tailscale status` from the container
  down   cancel the call and stop the app

`up` needs TS_OAUTH_CLIENT_SECRET in the environment (source .env first).
A CPU container: tailscale and gVisor networking do not depend on the GPU.
"""

from __future__ import annotations

import argparse
import getpass
import os
import socket
import subprocess
import sys
import threading
import time

import modal

USER = os.environ.get("POC_USER") or getpass.getuser()
APP_NAME = f"prl-probe-{USER}"
TAG = "tag:prime-rl"
HOSTNAME = "prl-probe-i0"
HTTP_PORT = 8300
SINK_PORT = 5202
IPERF_PORT = 5201
SOCK = "/tmp/ts/sock"

app = modal.App(APP_NAME)
registry = modal.Dict.from_name(APP_NAME, create_if_missing=True)
image = modal.Image.from_name("poc-h200-cu129")
secret = modal.Secret.from_dict(
    {"TS_OAUTH_CLIENT_SECRET": os.environ.get("TS_OAUTH_CLIENT_SECRET", ""), "POC_USER": USER}
)


def _ts(*args: str) -> str:
    return subprocess.run(
        ["tailscale", f"--socket={SOCK}", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sink(port: int) -> None:
    """Accept connections and discard everything sent: the home side measures the send rate."""
    server = socket.create_server(("0.0.0.0", port))

    def drain(conn: socket.socket) -> None:
        buf = bytearray(1 << 20)
        with conn:
            while conn.recv_into(buf):
                pass

    while True:
        conn, _ = server.accept()
        threading.Thread(target=drain, args=(conn,), daemon=True).start()


@app.function(image=image, region="us-central", timeout=3600, secrets=[secret], cpu=8, memory=16384)
def node() -> None:
    os.makedirs("/tmp/ts", exist_ok=True)
    daemon_log = open("/tmp/tailscaled.log", "w")
    daemon = subprocess.Popen(
        ["tailscaled", "--tun=userspace-networking", "--statedir=/tmp/ts", f"--socket={SOCK}"],
        stdout=daemon_log,
        stderr=subprocess.STDOUT,
    )
    while not os.path.exists(SOCK):
        if daemon.poll() is not None:
            raise RuntimeError("tailscaled exited: " + open("/tmp/tailscaled.log").read()[-2000:])
        time.sleep(0.2)

    key_path = "/tmp/ts/authkey"
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(os.environ["TS_OAUTH_CLIENT_SECRET"] + "?ephemeral=true&preauthorized=true")
    try:
        _ts("up", f"--auth-key=file:{key_path}", f"--advertise-tags={TAG}", f"--hostname={HOSTNAME}",
            "--accept-dns=false")
    finally:
        os.remove(key_path)
    tailnet_ip = _ts("ip", "-4")

    subprocess.Popen([sys.executable, "-m", "http.server", str(HTTP_PORT), "--directory", "/etc/apt"])
    # A second HTTP server on an OS-assigned port stands in for an env worker's payload server.
    probe = socket.socket()
    probe.bind(("0.0.0.0", 0))
    random_port = probe.getsockname()[1]
    probe.close()
    subprocess.Popen([sys.executable, "-m", "http.server", str(random_port), "--directory", "/etc/apt"])
    subprocess.Popen(["iperf3", "-s", "-p", str(IPERF_PORT)])
    threading.Thread(target=_sink, args=(SINK_PORT,), daemon=True).start()

    # A raw TCP tunnel to the sink: the home side dials Modal's public endpoint, no tailnet involved.
    with modal.forward(SINK_PORT, unencrypted=True) as tunnel:
        registry["node"] = {
            "task": os.environ.get("MODAL_TASK_ID"),
            "tailnet_ip": tailnet_ip,
            "http_port": HTTP_PORT,
            "random_port": random_port,
            "sink_port": SINK_PORT,
            "iperf_port": IPERF_PORT,
            "tunnel": "%s:%d" % tunnel.tcp_socket,
            "region": os.environ.get("MODAL_REGION"),
            "cloud": os.environ.get("MODAL_CLOUD_PROVIDER", ""),
        }
        print(f"registered {registry['node']}", flush=True)
        try:
            while True:
                time.sleep(60)
                print(_ts("status"), flush=True)
        finally:
            subprocess.run(["tailscale", f"--socket={SOCK}", "logout"], check=False)


def cmd_up(_: argparse.Namespace) -> None:
    if not os.environ.get("TS_OAUTH_CLIENT_SECRET"):
        sys.exit("TS_OAUTH_CLIENT_SECRET is not set (source .env)")
    with modal.enable_output():
        app.deploy(name=APP_NAME)
    registry.clear()
    registry["_call"] = modal.Function.from_name(APP_NAME, "node").spawn().object_id
    start = time.time()
    while "node" not in registry:
        if time.time() - start > 600:
            sys.exit("container did not register within 10 min; check `modal app logs`, then `down`")
        time.sleep(5)
    print(registry["node"])


def cmd_ls(_: argparse.Namespace) -> None:
    print(registry.get("node"))


def cmd_logs(_: argparse.Namespace) -> None:
    task = registry["node"]["task"]
    command = f"tail -40 /tmp/tailscaled.log; tailscale --socket={SOCK} status; tailscale --socket={SOCK} netcheck"
    subprocess.run(["modal", "container", "exec", task, "bash", "-c", command], check=False)


def cmd_down(_: argparse.Namespace) -> None:
    call_id = registry.get("_call")
    if call_id:
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
        print(f"cancelled {call_id}")
    subprocess.run(["modal", "app", "stop", "-y", APP_NAME], check=False)
    registry.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["up", "ls", "logs", "down"])
    args = parser.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()

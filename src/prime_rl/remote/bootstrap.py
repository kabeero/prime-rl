"""What every node of a remote inference deployment runs for its whole life: join the tailnet, install the
project, run the node script, and end the node when a process dies or the home side stops sending heartbeats.

The provider's function loads this file from the shipped checkout before anything is installed, so it imports
only the standard library and the provider's SDK.
"""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import IO, Any

PROJECT_DIR = "/root/prime-rl"
CONFIG_DIR = Path("/prl/configs")
LOG_DIR = Path("/prl/logs")
VENV_DIR = "/opt/prl-venv"
WORK_DIR = "/mnt/work"
TS_DIR = Path("/tmp/prl-ts")
TS_SOCKET = TS_DIR / "sock"
HEARTBEAT_FILE = Path("/tmp/prl-heartbeat")
BIN_DIR = Path("/usr/local/bin")

NODE_KEY = "node-{rank}"
STATUS_KEY = "status-{rank}"
SYNCED_KEY = "synced"
HEARTBEAT_KEY = "heartbeat"

WATCHDOG_PERIOD_S = 5
REGISTRY_PERIOD_S = 30


class Bootstrap:
    def __init__(self, bundle: dict[str, Any]) -> None:
        import modal

        cluster = modal.Cluster.from_context()
        self.bundle = bundle
        self.rank: int = cluster.container_rank()
        self.cluster_ips: list[str] = cluster.container_ips(family="ipv4")
        self.cluster_ipv6: list[str] = cluster.container_ips()
        self.registry = modal.Dict.from_name(bundle["registry"])
        self.volume = modal.Volume.from_name(bundle["volume"])
        self.registry_error = modal.exception.Error
        (LOG_DIR / "remote").mkdir(parents=True, exist_ok=True)
        self.log_file: IO[str] = (LOG_DIR / "remote" / f"bootstrap.{self.rank}.log").open("a")
        self.tailscaled: subprocess.Popen | None = None
        self.node: subprocess.Popen | None = None

    def log(self, message: str) -> None:
        line = f"[node {self.rank}] {message}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def status(self, stage: str) -> None:
        self.log(stage)
        self.registry[STATUS_KEY.format(rank=self.rank)] = stage

    def relay(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self.log(line.rstrip("\n"))

    def run(self, command: list[str], env: dict[str, str] | None = None) -> None:
        """Run ``command`` to completion with its output in this node's log."""
        self.log("$ " + " ".join(command))
        process = subprocess.Popen(
            command, cwd=PROJECT_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        self.relay(process)
        if process.wait() != 0:
            raise RuntimeError(f"node {self.rank}: `{' '.join(command)}` exited with code {process.returncode}")

    def ensure_tailscale(self) -> None:
        """The image carries the Tailscale binaries. Where it does not, download the pinned version."""
        if shutil.which("tailscale") and shutil.which("tailscaled"):
            return
        version = self.bundle["tailscale_version"]
        self.log(f"WARNING: the image has no tailscale binaries, downloading {version}. Rebuild the image with them.")
        url = f"https://pkgs.tailscale.com/stable/tailscale_{version}_amd64.tgz"
        archive = Path("/tmp/tailscale.tgz")
        urllib.request.urlretrieve(url, archive)
        expected = urllib.request.urlopen(url + ".sha256").read().decode().strip()
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"node {self.rank}: {url} does not match its published sha256")
        with tarfile.open(archive) as tar:
            for name in ("tailscale", "tailscaled"):
                member = tar.getmember(f"tailscale_{version}_amd64/{name}")
                source = tar.extractfile(member)
                assert source is not None
                (BIN_DIR / name).write_bytes(source.read())
                (BIN_DIR / name).chmod(0o755)

    def join_tailnet(self) -> str:
        """Join the tailnet in userspace mode, which is enough for inbound connections, and return this node's
        tailnet address."""
        self.ensure_tailscale()
        TS_DIR.mkdir(parents=True, exist_ok=True)
        daemon_log = (LOG_DIR / "remote" / f"tailscaled.{self.rank}.log").open("w")
        self.tailscaled = subprocess.Popen(
            ["tailscaled", "--tun=userspace-networking", f"--statedir={TS_DIR}", f"--socket={TS_SOCKET}"],
            stdout=daemon_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 30
        while not TS_SOCKET.exists():
            if time.monotonic() > deadline:
                raise RuntimeError(f"node {self.rank}: tailscaled did not open {TS_SOCKET}")
            time.sleep(0.2)
        # The auth key goes through a file so that it never shows in the process list, and no later process
        # inherits it.
        key_file = TS_DIR / "authkey"
        key_file.touch(mode=0o600)
        key_file.write_text(os.environ.pop("TS_OAUTH_CLIENT_SECRET") + "?ephemeral=true&preauthorized=true")
        try:
            self.run(
                [
                    "tailscale",
                    f"--socket={TS_SOCKET}",
                    "up",
                    f"--auth-key=file:{key_file}",
                    f"--advertise-tags={self.bundle['tailscale_tag']}",
                    f"--hostname=prl-{self.bundle['run_id'][:12]}-i{self.rank}",
                    "--accept-dns=false",
                ]
            )
        finally:
            key_file.unlink()
        address = subprocess.run(
            ["tailscale", f"--socket={TS_SOCKET}", "ip", "-4"], check=True, capture_output=True, text=True
        )
        return address.stdout.strip()

    def install(self) -> dict[str, str]:
        """Install the project into a node-local environment and return the environment its processes run in.
        Node 0 goes first and fills the caches on the shared volume, which the other nodes then install from."""
        env = {
            **os.environ,
            "UV_CACHE_DIR": f"{WORK_DIR}/.cache/uv",
            "HF_HOME": f"{WORK_DIR}/.cache/huggingface",
            "UV_PROJECT_ENVIRONMENT": VENV_DIR,
            "UV_LINK_MODE": "copy",
            # Git dependencies versioned by setuptools-scm run `git describe --dirty` in uv's checkout on the volume,
            # where a cold stat of the whole tree takes longer than the default 40 s.
            "SETUPTOOLS_SCM_SUBPROCESS_TIMEOUT": "900",
        }
        sync = ["uv", "sync", "--all-extras", "--all-packages", "--frozen"]
        if self.rank == 0:
            self.status("installing the project")
            self.run(sync, env)
            model = self.bundle["model_name"]
            if not Path(model).is_absolute():
                self.status(f"downloading {model}")
                download = "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])"
                self.run(["uv", "run", "--no-sync", "python", "-c", download, model], env)
            self.run(["sync", WORK_DIR])
            self.registry[SYNCED_KEY] = True
        else:
            self.status("waiting for node 0 to fill the caches")
            while not self.registry.get(SYNCED_KEY):
                time.sleep(WATCHDOG_PERIOD_S)
            self.volume.reload()
            self.status("installing the project")
            self.run(sync, env)
        env["UV_NO_SYNC"] = "1"
        return env

    def write_configs(self, tailnet_ip: str) -> None:
        bundle = self.bundle
        for name, text in bundle["files"].items():
            path = CONFIG_DIR / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        services = {
            f"engine_{k}": f"http://127.0.0.1:{bundle['backend_port'] + k}/health" for k in range(bundle["dp_per_node"])
        }
        if self.rank == 0:
            services["router"] = f"http://127.0.0.1:{bundle['router_port']}/health"
        agent = {
            "rank": self.rank,
            "port": bundle["agent_port"],
            "tailnet_ip": tailnet_ip,
            "cluster_ip": self.cluster_ips[self.rank],
            "log_dir": str(LOG_DIR),
            "heartbeat_file": str(HEARTBEAT_FILE),
            "admin_token_var": bundle["admin_token_var"],
            "http_services": services,
            "address_files": {
                f"env_{split}_{name}": str(CONFIG_DIR / "envs" / split / f"{name}.{self.rank}.address")
                for split, name in bundle["env_pools"]
            },
        }
        (CONFIG_DIR / "remote" / "agent.json").write_text(json.dumps(agent, indent=2))

    def start_node(self, env: dict[str, str], tailnet_ip: str) -> None:
        if shutil.which("iperf3"):
            subprocess.Popen(["iperf3", "-s"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        node_env = {
            **env,
            "PROJECT_DIR": PROJECT_DIR,
            "CONFIG_DIR": str(CONFIG_DIR),
            "LOG_DIR": str(LOG_DIR),
            "NODE_RANK": str(self.rank),
            "CLUSTER_IPS": " ".join(self.cluster_ips),
            "TAILNET_IP": tailnet_ip,
            "PRL_RUN_ID": self.bundle["run_id"],
            "PRL_RUN_NAME": self.bundle["run_name"],
        }
        self.node = subprocess.Popen(
            ["bash", str(CONFIG_DIR / self.bundle["node_script"])],
            env=node_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        threading.Thread(target=self.relay, args=(self.node,), daemon=True).start()

    def watch(self) -> None:
        """Return never: raise when the node script exits or the home side has been silent for too long. The
        fleet manager's heartbeat arrives at the node agent, and in the provider's registry, which also covers
        the time before the agent is up."""
        assert self.node is not None
        timeout = self.bundle["heartbeat_timeout_s"]
        last_beat = last_read = time.monotonic()
        seen = self.registry.get(HEARTBEAT_KEY)
        while True:
            code = self.node.poll()
            if code is not None:
                raise RuntimeError(f"node {self.rank}: a process of the node exited (node script code {code})")
            now = time.monotonic()
            if HEARTBEAT_FILE.exists():
                last_beat = max(last_beat, now - (time.time() - HEARTBEAT_FILE.stat().st_mtime))
            if now - last_read >= REGISTRY_PERIOD_S:
                last_read = now
                # The registry is a remote service: one failed read must not end the node.
                try:
                    value = self.registry.get(HEARTBEAT_KEY)
                except self.registry_error as e:
                    self.log(f"WARNING: could not read the heartbeat from the registry: {e!r}")
                else:
                    if value != seen:
                        seen, last_beat = value, now
            if now - last_beat > timeout:
                raise RuntimeError(f"node {self.rank}: no heartbeat from the home side for {timeout}s")
            time.sleep(WATCHDOG_PERIOD_S)

    def stop(self) -> None:
        if self.node is not None and self.node.poll() is None:
            os.killpg(self.node.pid, signal.SIGTERM)
            try:
                self.node.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.node.pid, signal.SIGKILL)
        if self.tailscaled is not None:
            subprocess.run(["tailscale", f"--socket={TS_SOCKET}", "logout"], capture_output=True)
            self.tailscaled.terminate()

    def main(self) -> None:
        try:
            self.status("joining the tailnet")
            tailnet_ip = self.join_tailnet()
            self.registry[NODE_KEY.format(rank=self.rank)] = {
                "rank": self.rank,
                "task_id": os.environ.get("MODAL_TASK_ID"),
                "tailnet_ip": tailnet_ip,
                "cluster_ipv4": self.cluster_ips[self.rank],
                "cluster_ipv6": self.cluster_ipv6[self.rank],
            }
            env = self.install()
            self.write_configs(tailnet_ip)
            self.status("starting the node's processes")
            self.start_node(env, tailnet_ip)
            self.watch()
        except BaseException as e:
            self.log(f"stopping: {e!r}")
            raise
        finally:
            self.stop()


def main(bundle: dict[str, Any]) -> None:
    Bootstrap(bundle).main()

"""The fleet manager of a remote inference deployment: the home side of the remote nodes' life."""

import json
import os
import re
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from prime_rl.configs.remote import REMOTE_ENDPOINTS, REMOTE_NODE_SCRIPT, RemoteFleetConfig
from prime_rl.configs.shared import ADMIN_TOKEN_HEADER
from prime_rl.remote.provider import RemoteNode, RemoteProvider
from prime_rl.utils.admin_token import admin_token_digest, digest_var
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import env_address_file

POLL_PERIOD_S = 5
LOG_PERIOD_S = 1
HEARTBEAT_PERIOD_S = 60
THROUGHPUT_SECONDS = 10
THROUGHPUT_STREAMS = (1, 8)
THROUGHPUT_CHUNK = bytes(2**20)
TAILSCALE = Path.home() / ".local" / "bin" / "tailscale"
TAILSCALE_API = "https://api.tailscale.com/api/v2"
PONG = re.compile(r"pong from \S+ \(\S+\) via (\S+) in (\d+(?:\.\d+)?)ms")


def required_secrets(config: RemoteFleetConfig) -> list[str]:
    return [
        "PRL_RUN_ID",
        "TS_OAUTH_CLIENT_ID",
        "TS_OAUTH_CLIENT_SECRET",
        config.admin_token_var,
        *config.deployment.forward_env,
    ]


def node_secrets(config: RemoteFleetConfig) -> dict[str, str]:
    """What the remote nodes get in their environment. Of the admin token they only get the digest: they check
    it, and never send it."""
    secrets = {name: os.environ[name] for name in ["TS_OAUTH_CLIENT_SECRET", *config.deployment.forward_env]}
    if hf_token := os.environ.get("HF_TOKEN"):
        secrets["HF_TOKEN"] = hf_token
    secrets[digest_var(config.admin_token_var)] = admin_token_digest(os.environ[config.admin_token_var])
    return secrets


def node_bundle(config: RemoteFleetConfig) -> dict[str, Any]:
    """The bootstrap's input: the configs the nodes' processes read, and what the bootstrap itself needs."""
    names = [
        "inference.json",
        REMOTE_NODE_SCRIPT,
        *(f"envs/{pool.split}/{pool.name}.json" for pool in config.env_pools),
    ]
    return {
        "run_id": os.environ["PRL_RUN_ID"],
        "run_name": config.run_name,
        "files": {name: (config.config_dir / name).read_text() for name in names},
        "node_script": REMOTE_NODE_SCRIPT,
        "model_name": config.model_name,
        "tailscale_tag": config.deployment.tailscale.tag,
        "tailscale_version": config.deployment.tailscale.version,
        "heartbeat_timeout_s": config.deployment.heartbeat_timeout_s,
        "admin_token_var": config.admin_token_var,
        "agent_port": config.deployment.agent_port,
        "router_port": config.router_port,
        "backend_port": config.backend_port,
        "dp_per_node": config.dp_per_node,
        "env_pools": [[pool.split, pool.name] for pool in config.env_pools],
    }


def tailnet_devices() -> list[dict[str, Any]]:
    """Every device of the tailnet, from the Tailscale API."""
    credentials = {"client_id": os.environ["TS_OAUTH_CLIENT_ID"], "client_secret": os.environ["TS_OAUTH_CLIENT_SECRET"]}
    token = httpx.post(f"{TAILSCALE_API}/oauth/token", data=credentials, timeout=30).raise_for_status().json()
    response = httpx.get(
        f"{TAILSCALE_API}/tailnet/-/devices",
        params={"fields": "all"},
        headers={"Authorization": f"Bearer {token['access_token']}"},
        timeout=30,
    )
    return response.raise_for_status().json()["devices"]


def missing_devices(devices: list[dict[str, Any]], home_hostnames: list[str], remote_ips: list[str]) -> list[str]:
    """The expected devices that are not connected to the tailnet: home nodes by hostname, remote nodes by
    address."""
    online = [device for device in devices if device.get("connectedToControl")]
    hostnames = {device["hostname"] for device in online}
    addresses = {address for device in online for address in device["addresses"]}
    return [name for name in home_hostnames if name not in hostnames] + [ip for ip in remote_ips if ip not in addresses]


def parse_ping(output: str) -> dict[str, Any]:
    """The path ``tailscale ping`` ended on: direct, or through a DERP relay."""
    pongs = PONG.findall(output)
    if not pongs:
        return {"reachable": False, "direct": False, "via": None, "latency_ms": None}
    via, latency = pongs[-1]
    return {"reachable": True, "direct": not via.startswith("DERP"), "via": via, "latency_ms": float(latency)}


def tailscale_ping(tailscale_socket: Path, ip: str) -> dict[str, Any]:
    # It exits non-zero when no direct path came up, which is a result here and not an error.
    result = subprocess.run(
        [str(TAILSCALE), f"--socket={tailscale_socket}", "ping", "-c", "3", ip], capture_output=True, text=True
    )
    return parse_ping(result.stdout)


def tcp_open(host: str, port: int) -> bool:
    try:
        socket.create_connection((host, port), timeout=5).close()
    except OSError:
        return False
    return True


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


class Fleet:
    def __init__(self, config: RemoteFleetConfig, provider: RemoteProvider) -> None:
        self.config = config
        self.provider = provider
        self.logger = get_logger()
        self.client = httpx.Client(headers={ADMIN_TOKEN_HEADER: os.environ[config.admin_token_var]}, timeout=30)
        self.nodes: list[RemoteNode] = []
        self.agents_up = False
        self.log_offsets: dict[tuple[int, str], int] = {}
        self.unreachable: set[tuple[str, int]] = set()
        self.stopping = threading.Event()
        self.background = threading.Thread(target=self.keep_alive, name="remote-fleet", daemon=True)
        self.deadline = 0.0

    def agent_url(self, node: RemoteNode) -> str:
        return f"http://{node.tailnet_ip}:{self.config.deployment.agent_port}"

    def engine_urls(self, node: RemoteNode) -> list[str]:
        return [f"http://{node.tailnet_ip}:{self.config.backend_port + k}" for k in range(self.config.dp_per_node)]

    def router_url(self) -> str:
        return f"http://{self.nodes[0].tailnet_ip}:{self.config.router_port}"

    def env_port(self, pool_index: int) -> int:
        return self.config.deployment.env_base_port + pool_index

    # Heartbeats and logs, from the moment the nodes start to the moment they stop.

    def tolerate(self, what: str, node: RemoteNode, action: Callable[[], None]) -> None:
        """Run ``action`` against a node's agent. A failure only warns, once per outage: the watchdog on the
        node decides when the home side has been away for too long."""
        key = (what, node.rank)
        try:
            action()
        except httpx.HTTPError as e:
            if key not in self.unreachable:
                self.logger.warning(f"Remote node {node.rank}: {what} failed: {e!r}")
            self.unreachable.add(key)
        else:
            self.unreachable.discard(key)

    def pull_logs(self, node: RemoteNode) -> None:
        """Append what the node's logs have grown by to the same paths under the local log dir."""
        url = self.agent_url(node)
        index = self.client.get(f"{url}/logs/index").raise_for_status().json()
        for name, size in index.items():
            offset = self.log_offsets.get((node.rank, name), 0)
            while offset < size:
                params = {"name": name, "offset": offset}
                chunk = self.client.get(f"{url}/logs", params=params).raise_for_status().content
                if not chunk:
                    break
                path = self.config.log_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("ab") as f:
                    f.write(chunk)
                offset += len(chunk)
                self.log_offsets[(node.rank, name)] = offset

    def heartbeat(self) -> None:
        try:
            self.provider.beat()
        except Exception as e:
            self.logger.warning(f"Heartbeat through the provider failed: {e!r}")
        for node in self.nodes if self.agents_up else []:
            url = self.agent_url(node)
            self.tolerate("heartbeat", node, lambda: self.client.post(f"{url}/heartbeat").raise_for_status())

    def keep_alive(self) -> None:
        last_beat = 0.0
        while not self.stopping.wait(LOG_PERIOD_S):
            if time.monotonic() - last_beat >= HEARTBEAT_PERIOD_S:
                last_beat = time.monotonic()
                self.heartbeat()
            for node in self.nodes if self.agents_up else []:
                self.tolerate("log pull", node, lambda: self.pull_logs(node))

    # Bring-up.

    def wait_for(self, what: str, done: Callable[[], Any]) -> Any:
        """Poll ``done`` until it returns something truthy, while the nodes are alive and the bring-up has
        time left."""
        statuses: dict[int, str] = {}
        while True:
            self.provider.check()
            if result := done():
                return result
            current = self.provider.statuses()
            for rank, status in current.items():
                if statuses.get(rank) != status:
                    self.logger.info(f"Remote node {rank}: {status}")
            statuses = current
            if time.monotonic() > self.deadline:
                timeout = self.config.deployment.bringup_timeout_s
                raise TimeoutError(
                    f"Remote inference bring-up timed out after {timeout}s (deployment.bringup_timeout_s) "
                    f"waiting for {what}. Node status: {statuses}"
                )
            time.sleep(POLL_PERIOD_S)

    def agent_health(self) -> list[dict[str, Any]] | None:
        try:
            return [self.client.get(f"{self.agent_url(node)}/health").raise_for_status().json() for node in self.nodes]
        except httpx.HTTPError:
            return None

    def not_ready(self) -> list[str]:
        """The remote processes that are not serving yet, as the nodes see them, and then from here over the
        tailnet."""
        health = self.agent_health()
        if health is None:
            return ["node agents"]
        pending = [
            f"node {node['rank']} {name}" for node in health for name, alive in node["processes"].items() if not alive
        ]
        if pending:
            return pending
        urls = [self.router_url(), *(url for node in self.nodes for url in self.engine_urls(node))]
        for url in urls:
            try:
                self.client.get(f"{url}/health").raise_for_status()
            except httpx.HTTPError:
                pending.append(f"{url} from the home side")
        for node in self.nodes:
            for index, pool in enumerate(self.config.env_pools):
                if not tcp_open(node.tailnet_ip, self.env_port(index)):
                    pending.append(f"node {node.rank} env pool {pool.split}/{pool.name} from the home side")
        return pending

    def wait_ready(self) -> None:
        reported: list[str] = []

        def ready() -> bool:
            nonlocal reported
            pending = self.not_ready()
            if pending != reported:
                self.logger.info(f"Waiting for {len(pending)} remote processes: {', '.join(pending)}")
            reported = pending
            return not pending

        self.wait_for("the engines, the router and the env worker pools to become healthy", ready)

    # Network checks.

    def throughput(self, streams: int) -> float:
        """Gbit/s from here to node 0's agent over ``streams`` connections."""
        url = f"{self.agent_url(self.nodes[0])}/sink"
        deadline = time.monotonic() + THROUGHPUT_SECONDS

        def body() -> Iterator[bytes]:
            while time.monotonic() < deadline:
                yield THROUGHPUT_CHUNK

        def send(_: int) -> int:
            response = self.client.post(url, content=body(), timeout=THROUGHPUT_SECONDS + 120)
            return response.raise_for_status().json()["bytes"]

        start = time.monotonic()
        with ThreadPoolExecutor(streams) as pool:
            received = sum(pool.map(send, range(streams)))
        return received * 8 / (time.monotonic() - start) / 1e9

    def preflight(self) -> None:
        """Check the network between the datacenters before anything depends on it, and record what it found."""
        config = self.config
        run_id = os.environ["PRL_RUN_ID"]
        home_hostnames = [f"prl-{run_id[:12]}-t{i}" for i in range(config.num_home_nodes)]
        report: dict[str, Any] = {"home": home_hostnames, "remote": [node.tailnet_ip for node in self.nodes]}
        report_path = config.log_dir / "remote" / "network.json"

        missing = missing_devices(tailnet_devices(), home_hostnames, report["remote"])
        report["missing_devices"] = missing
        if config.tailscale_socket is not None:
            report["paths"] = {
                node.rank: tailscale_ping(config.tailscale_socket, node.tailnet_ip) for node in self.nodes
            }
        write_atomic(report_path, json.dumps(report, indent=2))
        if missing:
            raise RuntimeError(f"Remote inference preflight: not on the tailnet: {missing}")
        for rank, path in report.get("paths", {}).items():
            if not path["reachable"]:
                raise RuntimeError(f"Remote inference preflight: remote node {rank} does not answer a tailscale ping")
            if path["direct"]:
                self.logger.info(f"Remote node {rank}: direct path via {path['via']}, {path['latency_ms']} ms")
                continue
            message = f"Remote node {rank}: the path is relayed through {path['via']}, {path['latency_ms']} ms"
            if config.deployment.tailscale.require_direct:
                raise RuntimeError(f"Remote inference preflight: {message} (tailscale.require_direct)")
            self.logger.warning(f"{message}. Expect a small fraction of the direct path's throughput.")

        report["throughput_gbit_s"] = {
            str(streams): round(self.throughput(streams), 4) for streams in THROUGHPUT_STREAMS
        }
        self.logger.info(f"Throughput to remote node 0 (Gbit/s by stream count): {report['throughput_gbit_s']}")
        report["object_store"] = "skipped: the remote weight transport only syncs the startup version"
        write_atomic(report_path, json.dumps(report, indent=2))

    def publish(self) -> None:
        """Hand the orchestrator the remote addresses, all of them on the tailnet."""
        config = self.config
        for node in self.nodes:
            for index, pool in enumerate(config.env_pools):
                path = env_address_file(config.config_dir, pool.split, pool.name, node.rank)
                write_atomic(path, f"tcp://{node.tailnet_ip}:{self.env_port(index)}")
        admin_urls = [f"{url}/v1" for node in self.nodes for url in self.engine_urls(node)]
        endpoints = {
            "INFER_URLS": f"{self.router_url()}/v1",
            "ADMIN_URLS": " ".join(admin_urls),
            "AGENT_URLS": " ".join(self.agent_url(node) for node in self.nodes),
        }
        text = "".join(f'{name}="{value}"\n' for name, value in endpoints.items())
        write_atomic(config.config_dir / REMOTE_ENDPOINTS, text)
        self.logger.success(f"Remote inference is up: {endpoints}")

    def run(self) -> None:
        """Bring the nodes up, then stay until they stop or this process is told to. The nodes stop with it."""
        config = self.config
        self.deadline = time.monotonic() + config.deployment.bringup_timeout_s
        try:
            self.provider.sweep()
            self.provider.start(node_bundle(config), node_secrets(config))
            self.background.start()
            nodes = self.wait_for("the remote nodes to start and join the tailnet", self.provider.nodes)
            for node in nodes:
                self.logger.info(f"Remote node {node.rank}: {node.node_id} tailnet={node.tailnet_ip}")
            self.nodes = nodes
            self.wait_for("the node agents to answer", self.agent_health)
            self.agents_up = True
            self.preflight()
            self.wait_ready()
            self.publish()
            while True:
                self.provider.check()
                time.sleep(POLL_PERIOD_S)
        finally:
            self.stopping.set()
            if self.background.is_alive():
                self.background.join()
            try:
                for node in self.nodes if self.agents_up else []:
                    self.tolerate("final log pull", node, lambda: self.pull_logs(node))
            finally:
                self.provider.stop()
                self.logger.info("Stopped the remote nodes")

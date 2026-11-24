"""The remote inference nodes as one clustered Modal function: every node runs ``prime_rl.remote.bootstrap``."""

import getpass
import json
import time
from pathlib import Path
from typing import Any

import modal

from prime_rl.configs.remote import REMOTE_APP_RECORD, RemoteFleetConfig
from prime_rl.remote import bootstrap
from prime_rl.remote.provider import RemoteNode
from prime_rl.utils.logger import get_logger

FUNCTION_NAME = "remote_node"
BOOTSTRAP_PATH = f"{bootstrap.PROJECT_DIR}/src/prime_rl/remote/bootstrap.py"

# What the nodes do not get of the checkout: they build their own environment, and run no git command.
IGNORE = [
    ".venv",
    "**/.venv",
    ".git",
    "**/.git",
    "outputs",
    "wandb",
    "_demo",
    "**/__pycache__",
    "**/node_modules",
    "**/.ruff_cache",
    "**/.pytest_cache",
    "**/.mypy_cache",
]


def stop_modal_app(launcher_dir: Path, app_name: str) -> None:
    """Stop the nodes of the app ``app_name`` and of the call recorded in ``launcher_dir``, whichever of them
    exist. Every step runs even when an earlier one fails."""
    record_path = launcher_dir / REMOTE_APP_RECORD
    errors: list[Exception] = []

    def cancel_call() -> None:
        if record_path.exists():
            call_id = json.loads(record_path.read_text())["function_call_id"]
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

    def stop_app() -> None:
        modal.App.lookup(app_name).stop()

    def delete_registry() -> None:
        modal.Dict.objects.delete(app_name, allow_missing=True)

    for step in (cancel_call, stop_app, delete_registry):
        try:
            step()
        except modal.exception.NotFoundError:
            pass
        except Exception as e:
            errors.append(e)
    if errors:
        raise errors[0]
    record_path.unlink(missing_ok=True)


class ModalProvider:
    def __init__(self, config: RemoteFleetConfig) -> None:
        self.config = config
        self.provider = config.deployment.provider
        self.volume_name = self.provider.volume.format(user=getpass.getuser())
        self.registry: modal.Dict | None = None
        self.call: modal.FunctionCall | None = None

    def sweep(self) -> None:
        stop_modal_app(self.config.launcher_dir, self.config.app_name)

    def bundle_fields(self) -> dict[str, Any]:
        return {"registry": self.config.app_name, "volume": self.volume_name}

    def start(self, bundle: dict[str, Any], secrets: dict[str, str]) -> None:
        config, provider = self.config, self.provider
        app = modal.App(config.app_name)
        image = modal.Image.from_name(provider.image).add_local_dir(
            config.project_dir, bootstrap.PROJECT_DIR, copy=False, ignore=IGNORE
        )
        volume = modal.Volume.from_name(self.volume_name, create_if_missing=True, version=2)

        # Serialized, so that the nodes need nothing of this module: the body only loads the bootstrap from
        # the shipped checkout.
        def remote_node(bundle: dict) -> None:
            import importlib.util

            spec = importlib.util.spec_from_file_location("prl_remote_bootstrap", BOOTSTRAP_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.main(bundle)

        app.function(
            name=FUNCTION_NAME,
            serialized=True,
            image=image,
            gpu=f"{provider.gpu}:{config.gpus_per_node}",
            timeout=provider.timeout_s,
            volumes={bootstrap.WORK_DIR: volume},
            secrets=[modal.Secret.from_dict(secrets)],
            region=provider.region,
            experimental_options={"efa_disabled": True} if provider.efa_disabled else None,
        )(modal.clustered(size=config.num_nodes, rdma=provider.rdma)(remote_node))

        app.deploy(name=config.app_name)
        self.registry = modal.Dict.from_name(config.app_name, create_if_missing=True)
        self.registry.clear()
        self.beat()
        self.call = modal.Function.from_name(config.app_name, FUNCTION_NAME).spawn({**bundle, **self.bundle_fields()})
        # Written right away: the batch script's exit trap and the next attempt's sweep stop the nodes from it.
        record = {"provider": "modal", "app_name": config.app_name, "function_call_id": self.call.object_id}
        record_path = config.launcher_dir / REMOTE_APP_RECORD
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record))
        get_logger().info(f"Started Modal app {config.app_name} (call {self.call.object_id})")

    def nodes(self) -> list[RemoteNode] | None:
        assert self.registry is not None
        entries = [self.registry.get(bootstrap.NODE_KEY.format(rank=rank)) for rank in range(self.config.num_nodes)]
        if not all(entries):
            return None
        return [
            RemoteNode(
                rank=entry["rank"],
                node_id=entry["task_id"],
                tailnet_ip=entry["tailnet_ip"],
                cluster_ip=entry["cluster_ipv4"],
            )
            for entry in entries
        ]

    def statuses(self) -> dict[int, str]:
        assert self.registry is not None
        statuses = {
            rank: self.registry.get(bootstrap.STATUS_KEY.format(rank=rank)) for rank in range(self.config.num_nodes)
        }
        return {rank: status for rank, status in statuses.items() if status}

    def beat(self) -> None:
        assert self.registry is not None
        self.registry[bootstrap.HEARTBEAT_KEY] = time.time()

    def check(self) -> None:
        """Raise when the nodes have stopped: with their error when they failed."""
        assert self.call is not None
        try:
            self.call.get(timeout=0)
        except TimeoutError:
            return
        raise RuntimeError(f"The remote nodes of {self.config.app_name} exited")

    def stop(self) -> None:
        stop_modal_app(self.config.launcher_dir, self.config.app_name)

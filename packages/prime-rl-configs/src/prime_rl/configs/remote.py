import re
from pathlib import Path

from prime_rl.configs.inference import RemoteInferenceDeploymentConfig
from prime_rl.configs.shared import LogConfig
from prime_rl.utils.config import BaseConfig

REMOTE_FLEET_CONFIG = "remote_fleet.json"
REMOTE_NODE_SCRIPT = "remote/node.sh"
REMOTE_ENDPOINTS = "remote/endpoints.env"
REMOTE_APP_RECORD = "remote_app.json"


def remote_app_name(run_name: str) -> str:
    """Name of the provider app holding a run's remote nodes: ``prl-<run.name>``, in the characters every provider
    accepts."""
    return "prl-" + re.sub(r"[^A-Za-z0-9._-]", "-", run_name)[:59]


class RemoteEnvPoolConfig(BaseConfig):
    """One env worker pool every remote node runs, configured by ``envs/<split>/<name>.json``."""

    split: str
    name: str


class RemoteFleetConfig(BaseConfig):
    """``uv run remote-fleet``: starts the remote inference nodes of an ``rl`` run, publishes their addresses once
    every remote process is healthy, keeps them alive, mirrors their logs and stops them when it exits. Written
    by the ``rl`` launcher."""

    run_name: str
    app_name: str
    """Name of the provider app holding the nodes."""

    project_dir: Path
    """The checkout the remote nodes run a copy of."""

    config_dir: Path
    """The attempt's config dir: the source of the inference and env pool configs, and where the endpoints go."""

    log_dir: Path
    """The attempt's log dir. Remote logs land under the paths the same processes would use locally."""

    launcher_dir: Path
    """Holds the record of the running app, which the batch script uses to stop it."""

    deployment: RemoteInferenceDeploymentConfig

    num_nodes: int
    """Number of remote nodes."""

    num_home_nodes: int
    """Number of nodes of the SLURM job, each of which joins the tailnet."""

    gpus_per_node: int
    dp_per_node: int
    """Engines per remote node, on ``backend_port + k``."""

    router_port: int
    backend_port: int
    env_pools: list[RemoteEnvPoolConfig] = []

    model_name: str
    """Model the nodes download before the engines start."""

    admin_token_var: str
    """Variable holding the run's admin token, which guards the node agents. The nodes only get its digest."""

    tailscale_socket: Path | None = None
    """Socket of this node's ``tailscaled``, for the network checks. Set by the launcher."""

    log: LogConfig = LogConfig()

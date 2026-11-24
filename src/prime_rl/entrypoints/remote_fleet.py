import os
import signal
from typing import Any

from prime_rl.configs.remote import RemoteFleetConfig
from prime_rl.remote.fleet import Fleet, required_secrets
from prime_rl.remote.provider import create_provider
from prime_rl.utils.config import check_mode, cli
from prime_rl.utils.logger import setup_logger


def terminate(signum: int, frame: Any) -> None:
    # Unwinds the main thread, so the fleet stops the remote nodes on its way out.
    raise SystemExit(128 + signum)


def main():
    """``uv run remote-fleet @ <config_dir>/remote_fleet.json``: run the remote inference nodes of an ``rl`` run."""
    check_mode("remote-fleet")
    config = cli(RemoteFleetConfig)
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    missing = [name for name in required_secrets(config) if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"remote-fleet needs {missing} in the environment")
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        Fleet(config, create_provider(config)).run()
    except Exception:
        logger.opt(exception=True).error("The remote fleet failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

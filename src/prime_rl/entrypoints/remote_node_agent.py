import uvicorn

from prime_rl.configs.remote import RemoteNodeAgentConfig
from prime_rl.remote.agent import create_app
from prime_rl.utils.config import check_mode, cli


def main():
    check_mode("remote-node-agent")
    config = cli(RemoteNodeAgentConfig)
    config.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()

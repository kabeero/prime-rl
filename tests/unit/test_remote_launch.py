from pathlib import Path

from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.rl import write_remote_fleet, write_slurm_script
from prime_rl.remote.fleet import missing_devices, parse_ping

REMOTE = {"type": "remote", "provider": {"type": "modal", "image": "img", "volume": "vol"}}


def _config(tmp_path: Path, remote: bool) -> RLConfig:
    source = {"name": "math", "env": {"taskset": {"id": "i3_math"}}}
    return RLConfig.model_validate(
        {
            "output_dir": tmp_path,
            "deployment": {"type": "multi_node", "num_train_nodes": 1, "num_infer_nodes": 1},
            "slurm": {},
            "trainer": {},
            "orchestrator": {"train": {"source": [source]}},
            "inference": {"deployment": REMOTE} if remote else {},
        }
    )


def _script(tmp_path: Path, remote: bool) -> str:
    config = _config(tmp_path, remote)
    script_path = tmp_path / "rl.sbatch"
    write_slurm_script(config, tmp_path / "configs", tmp_path / "logs", script_path)
    return script_path.read_text()


def test_remote_inference_takes_no_slurm_node_and_stops_the_remote_nodes_on_exit(tmp_path):
    script = _script(tmp_path, remote=True)
    assert "#SBATCH --nodes=1\n" in script
    assert "#SBATCH --signal=B:TERM@120" in script
    assert "export NUM_LOCAL_INFER_NODES=0" in script
    assert "trap stop_remote EXIT" in script
    assert "uv run remote-fleet @ $CONFIG_DIR/remote_fleet.json" in script
    # The engines run on the remote nodes, so the job launches none.
    assert "launch_inference_rank" not in script


def test_local_inference_script_has_nothing_of_the_remote_deployment(tmp_path):
    script = _script(tmp_path, remote=False)
    assert "#SBATCH --nodes=2\n" in script
    for marker in ("tailscale", "remote", "NUM_LOCAL_INFER_NODES", "--signal"):
        assert marker not in script


def test_remote_node_script_runs_the_node_processes_on_the_private_network(tmp_path):
    config = _config(tmp_path, remote=True)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    write_remote_fleet(config, config_dir, tmp_path / "logs")
    script = (config_dir / "remote" / "node.sh").read_text()
    assert "uv run remote-node-agent @ $CONFIG_DIR/remote/agent.json" in script
    assert "--serve.address tcp://0.0.0.0:7000" in script
    assert "--serve.advertise-host $TAILNET_IP" in script
    assert "--serve.inference-base-url http://${INFER_HOSTS[0]}:$ROUTER_PORT/v1" in script
    assert "launch_inference_rank" in script


def test_parse_ping_reports_the_path_the_ping_ended_on():
    relayed = "pong from n (100.64.0.1) via DERP(ord) in 24ms\n"
    assert parse_ping(relayed) == {"reachable": True, "direct": False, "via": "DERP(ord)", "latency_ms": 24.0}
    direct = parse_ping(relayed + "pong from n (100.64.0.1) via 203.0.113.7:41641 in 5ms\n")
    assert direct["direct"] and direct["via"] == "203.0.113.7:41641"
    assert not parse_ping("no reply\n")["reachable"]


def test_missing_devices_matches_home_nodes_by_hostname_and_remote_nodes_by_address():
    devices = [
        {"hostname": "prl-abc-t0", "addresses": ["100.64.0.1"], "connectedToControl": True},
        {"hostname": "prl-abc-t1", "addresses": ["100.64.0.2"], "connectedToControl": False},
        {"hostname": "prl-abc-i0", "addresses": ["100.64.0.3", "fd7a::3"], "connectedToControl": True},
    ]
    missing = missing_devices(devices, ["prl-abc-t0", "prl-abc-t1"], ["100.64.0.3", "100.64.0.4"])
    assert missing == ["prl-abc-t1", "100.64.0.4"]

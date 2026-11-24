"""Modal-only smoke test of the remote bootstrap on one CPU container (no GPUs, no cluster): ships the checkout,
joins the tailnet, runs the real `uv sync` (which also fills the volume's uv cache), then a stub node script that
only starts the node agent. Checks the agent from inside the container, then stops everything.

  source .env && uv run python _demo/smoke/modal_smoke.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

from prime_rl.remote import bootstrap
from prime_rl.remote.modal_provider import BOOTSTRAP_PATH, IGNORE, stop_modal_app
from prime_rl.utils.admin_token import admin_token_digest

APP = "prl-smoke-mkgz"
TOKEN = "smoke-token"
LAUNCHER = Path("/tmp/prl-smoke-launcher")
NODE_SH = """#!/bin/bash
cd $PROJECT_DIR
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
mkdir -p $LOG_DIR/remote
echo "stub node rank=$NODE_RANK ips=$CLUSTER_IPS tailnet=$TAILNET_IP uv_no_sync=$UV_NO_SYNC"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
which vllm-router
uv run remote-node-agent @ $CONFIG_DIR/remote/agent.json > $LOG_DIR/remote/agent.$NODE_RANK.log 2>&1 &
wait -n
exit 1
"""


def main() -> None:
    LAUNCHER.mkdir(exist_ok=True)
    stop_modal_app(LAUNCHER, APP)
    app = modal.App(APP)
    image = modal.Image.from_name("poc-h200-cu129").add_local_dir(
        Path.cwd(), bootstrap.PROJECT_DIR, copy=False, ignore=IGNORE
    )
    volume = modal.Volume.from_name("poc-work-mkgz", create_if_missing=True, version=2)
    secrets = {
        "TS_OAUTH_CLIENT_SECRET": os.environ["TS_OAUTH_CLIENT_SECRET"],
        "PRIME_RL_ADMIN_TOKEN_SHA256": admin_token_digest(TOKEN),
        "HF_TOKEN": os.environ["HF_TOKEN"],
    }

    def remote_node(bundle: dict) -> None:
        import importlib.util

        import modal

        class FakeCluster:
            def container_rank(self):
                return 0

            def container_ips(self, family="ipv6"):
                return ["127.0.0.1"]

        modal.Cluster.from_context = lambda: FakeCluster()
        spec = importlib.util.spec_from_file_location("prl_remote_bootstrap", BOOTSTRAP_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main(bundle)

    app.function(
        name="remote_node",
        serialized=True,
        image=image,
        cpu=8,
        memory=32768,
        timeout=3 * 3600,
        volumes={bootstrap.WORK_DIR: volume},
        secrets=[modal.Secret.from_dict(secrets)],
    )(remote_node)
    app.deploy(name=APP)
    registry = modal.Dict.from_name(APP, create_if_missing=True)
    registry.clear()
    bundle = {
        "registry": APP, "volume": "poc-work-mkgz", "run_id": "smoke0000000000", "run_name": "smoke",
        "files": {"remote/node.sh": NODE_SH, "inference.json": "{}"}, "node_script": "remote/node.sh",
        "model_name": next((arg for arg in sys.argv[1:] if not arg.startswith("--")), "Qwen/Qwen3-0.6B"),
        "tailscale_tag": "tag:prime-rl", "tailscale_version": "1.104.7", "heartbeat_timeout_s": 120,
        "admin_token_var": "PRIME_RL_ADMIN_TOKEN", "agent_port": 8300, "router_port": 8000, "backend_port": 8100,
        "dp_per_node": 1, "env_pools": [],
    }  # fmt: skip
    call = modal.Function.from_name(APP, "remote_node").spawn(bundle)
    (LAUNCHER / "remote_app.json").write_text(json.dumps({"function_call_id": call.object_id}))
    print("call", call.object_id, flush=True)
    t0 = time.time()
    status = None
    try:
        while True:
            registry["heartbeat"] = time.time()
            try:
                call.get(timeout=0)
                raise SystemExit("call finished")
            except TimeoutError:
                pass
            s = registry.get("status-0")
            if s != status:
                status = s
                print(f"[{int(time.time() - t0):5d}s] {s}", flush=True)
            if s == "starting the node's processes":
                break
            time.sleep(10)
        node = registry.get("node-0")
        print("node", node, flush=True)
        time.sleep(45)
        check = (
            f'curl -s -H "X-Prime-Admin-Token: {TOKEN}" localhost:8300/health; echo; '
            f'curl -s -H "X-Prime-Admin-Token: {TOKEN}" localhost:8300/logs/index; echo; '
            "cat /prl/logs/remote/bootstrap.0.log | tail -15; cat /prl/logs/remote/agent.0.log | tail -5; "
            "tailscale --socket=/tmp/prl-ts/sock status | head -5; du -sh /opt/prl-venv"
        )
        out = subprocess.run(
            ["modal", "container", "exec", node["task_id"], "--", "bash", "-c", check], capture_output=True, text=True
        )
        print(out.stdout, out.stderr, flush=True)
        if "--watchdog" in sys.argv:
            print("stopping heartbeats; the call should fail within ~150s", flush=True)
            t1 = time.time()
            while True:
                try:
                    call.get(timeout=0)
                except TimeoutError:
                    time.sleep(10)
                    continue
                except Exception as e:
                    print(f"call ended after {int(time.time() - t1)}s: {e!r}", flush=True)
                    break
    finally:
        stop_modal_app(LAUNCHER, APP)
        print("stopped; record exists:", (LAUNCHER / "remote_app.json").exists(), flush=True)
        stop_modal_app(LAUNCHER, APP)
        print("second stop ok", flush=True)


if __name__ == "__main__":
    main()

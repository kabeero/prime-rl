"""The node agent of a remote inference deployment: reports the health of the processes on its node, takes the
fleet manager's heartbeats and serves the node's logs."""

import asyncio
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from prime_rl.configs.remote import RemoteNodeAgentConfig
from prime_rl.configs.shared import ADMIN_TOKEN_HEADER
from prime_rl.utils.admin_token import admin_token_matches, digest_var, expected_admin_digest

LOG_CHUNK_BYTES = 4 * 2**20


def read_log(log_dir: Path, name: str, offset: int) -> bytes:
    """Up to ``LOG_CHUNK_BYTES`` of the log ``name`` (a path relative to ``log_dir``) from ``offset``."""
    path = (log_dir / name).resolve()
    if not path.is_relative_to(log_dir.resolve()) or not path.is_file():
        raise FileNotFoundError(name)
    with path.open("rb") as f:
        f.seek(offset)
        return f.read(LOG_CHUNK_BYTES)


def log_index(log_dir: Path) -> dict[str, int]:
    """Size of every log under ``log_dir``, by its path relative to it."""
    return {
        str(path.relative_to(log_dir)): path.stat().st_size for path in sorted(log_dir.rglob("*")) if path.is_file()
    }


async def http_alive(client: httpx.AsyncClient, url: str) -> bool:
    try:
        return (await client.get(url)).status_code == status.HTTP_200_OK
    except httpx.HTTPError:
        return False


def create_app(config: RemoteNodeAgentConfig) -> FastAPI:
    expected = expected_admin_digest(config.admin_token_var)
    if expected is None:
        raise RuntimeError(f"The node agent needs ${digest_var(config.admin_token_var)} to check its callers against.")

    app = FastAPI(title="prime-rl remote node agent")
    client = httpx.AsyncClient(timeout=5.0)

    @app.middleware("http")
    async def check_admin_token(request: Request, call_next):
        if not admin_token_matches(request.headers.get(ADMIN_TOKEN_HEADER), expected):
            return JSONResponse({"detail": "Missing or invalid admin token"}, status_code=status.HTTP_401_UNAUTHORIZED)
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict:
        names = list(config.http_services)
        alive = await asyncio.gather(*(http_alive(client, config.http_services[name]) for name in names))
        processes = dict(zip(names, alive)) | {name: path.exists() for name, path in config.address_files.items()}
        return {
            "rank": config.rank,
            "tailnet_ip": config.tailnet_ip,
            "cluster_ip": config.cluster_ip,
            "ready": all(processes.values()),
            "processes": processes,
        }

    @app.post("/heartbeat")
    async def heartbeat() -> dict:
        config.heartbeat_file.touch()
        return {"time": time.time()}

    @app.get("/logs/index")
    async def logs_index() -> dict[str, int]:
        return log_index(config.log_dir)

    @app.get("/logs")
    async def logs(name: str, offset: int = 0) -> Response:
        try:
            return Response(read_log(config.log_dir, name, offset), media_type="application/octet-stream")
        except FileNotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No log named {name!r}")

    @app.post("/sink")
    async def sink(request: Request) -> dict:
        """Discard the request body: the receiving end of the fleet manager's throughput check."""
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
        return {"bytes": received}

    return app

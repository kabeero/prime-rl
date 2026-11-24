import sys
from pathlib import Path

from prime_rl.remote.modal_provider import stop_modal_app


def main() -> None:
    """``python -m prime_rl.entrypoints.remote_stop <launcher_dir> <app_name>``: stop the remote inference nodes
    of a run, if any are left. The batch script runs it on every exit path."""
    launcher_dir, app_name = sys.argv[1:]
    stop_modal_app(Path(launcher_dir), app_name)


if __name__ == "__main__":
    main()

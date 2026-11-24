from dataclasses import dataclass
from typing import Any, Protocol

from prime_rl.configs.remote import RemoteFleetConfig


@dataclass(frozen=True)
class RemoteNode:
    rank: int
    node_id: str
    """The provider's name for the node, for debugging."""
    tailnet_ip: str
    cluster_ip: str
    """The node's address on the private network between the remote nodes."""


class RemoteProvider(Protocol):
    """Where the remote nodes run. Every node runs ``prime_rl.remote.bootstrap`` on a copy of the checkout."""

    def sweep(self) -> None:
        """Stop what an earlier attempt of the run left behind."""

    def start(self, bundle: dict[str, Any], secrets: dict[str, str]) -> None:
        """Start the nodes with ``bundle`` as the bootstrap's input and ``secrets`` in their environment."""

    def nodes(self) -> list[RemoteNode] | None:
        """The nodes by rank, once all of them have joined the tailnet."""

    def statuses(self) -> dict[int, str]:
        """What each node's bootstrap is doing."""

    def beat(self) -> None:
        """Tell the nodes the home side is alive, by a path that needs no node agent."""

    def check(self) -> None:
        """Raise when the nodes have stopped."""

    def stop(self) -> None:
        """Stop the nodes. Safe to call at any point, and more than once."""


def create_provider(config: RemoteFleetConfig) -> RemoteProvider:
    from prime_rl.remote.modal_provider import ModalProvider

    return ModalProvider(config)

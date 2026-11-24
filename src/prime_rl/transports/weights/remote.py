import asyncio
import contextlib
from collections.abc import Callable
from pathlib import Path

import torch.nn as nn

from prime_rl.configs.shared import RemoteWeightBroadcastConfig
from prime_rl.transports.weights.base import (
    FINISHED_MARKER,
    RECEIVER_READY_MARKER,
    SENDER_READY_MARKER,
    WeightReceiver,
    WeightSender,
    marker_exists,
)


class RemoteStartupOnlySender(WeightSender):
    """Offers every version to the consumer but moves no weights: the remote engines already
    serve the weights the trainer starts from. Paired with ``RemoteStartupOnlyReceiver``, which
    accepts only the startup version."""

    def __init__(self, output_dir: Path, config: RemoteWeightBroadcastConfig):
        super().__init__(output_dir, config.timeout)

    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        pass


class RemoteStartupOnlyReceiver(WeightReceiver):
    """Acknowledges the startup version without loading it, and fails on any later version:
    remote inference keeps serving the weights it started with, so a run cannot continue past
    its first update."""

    async def _acknowledge(self, step: int, cancelled: Callable[[], bool] | None = None) -> None:
        weights_dir = self.step_dir(step)
        # A trainer restarted mid-broadcast offers the version again: acknowledge every offer.
        while not marker_exists(weights_dir / FINISHED_MARKER):
            if cancelled is not None and cancelled():
                raise asyncio.CancelledError
            offered = marker_exists(weights_dir / SENDER_READY_MARKER)
            if offered and not marker_exists(weights_dir / RECEIVER_READY_MARKER):
                with contextlib.suppress(FileNotFoundError):
                    self._ack(step)
            await asyncio.sleep(0.1)

    async def receive(self, step: int) -> None:
        raise RuntimeError(
            f"The trainer offered policy v{step}, but weight updates to remote inference are not implemented: "
            "the remote engines still serve the weights the run started with."
        )

    async def sync_startup(self, step: int, timeout: float) -> None:
        await asyncio.wait_for(self.wait_published(step), timeout=timeout)
        await asyncio.wait_for(self._acknowledge(step), timeout=timeout)
        self.logger.warning(
            f"Remote inference keeps its launch weights as policy v{step}; the run stops at the next version."
        )

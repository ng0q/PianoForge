"""Device sync and accelerator memory tracking."""

from __future__ import annotations

import threading
import time


def preferred_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync_device(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


class VramTracker:
    """Peak accelerator memory for the inference process.

    CUDA uses torch.cuda.max_memory_allocated. MPS has no peak counter, so
    the tracker samples torch.mps.current_allocated_memory. CPU has neither.
    """

    def __init__(self, device: str, poll: bool = True):
        self.device = device
        self.source = None
        self.reason = None
        self._peak = 0
        self._stop = threading.Event()
        self._thread = None
        if device.startswith("cuda"):
            import torch

            torch.cuda.reset_peak_memory_stats()
            self.source = "torch.cuda.max_memory_allocated"
        elif device.startswith("mps") and poll:
            self.source = "peak of torch.mps.current_allocated_memory, sampled every 50 ms"
            self._thread = threading.Thread(target=self._poll_mps, daemon=True)
            self._thread.start()
        elif device.startswith("mps"):
            self.source = (
                "torch.mps.current_allocated_memory read on the inference thread after each recording. "
                "A side thread was not used because sampling from one stalls this model in the Metal allocator."
            )
        else:
            self.reason = "inference device is cpu; no accelerator memory counter is available"

    def note(self, value: int) -> None:
        self._peak = max(self._peak, int(value))

    def _poll_mps(self) -> None:
        import torch

        while not self._stop.is_set():
            try:
                self._peak = max(self._peak, int(torch.mps.current_allocated_memory()))
            except Exception:
                break
            time.sleep(0.05)

    def finish(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.device.startswith("cuda"):
            import torch

            return {
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_vram_source": self.source,
                "peak_vram_reason": None,
            }
        if self.device.startswith("mps"):
            return {
                "peak_vram_bytes": int(self._peak) if self._peak else 0,
                "peak_vram_source": self.source,
                "peak_vram_reason": None,
            }
        return {
            "peak_vram_bytes": None,
            "peak_vram_source": None,
            "peak_vram_reason": self.reason,
        }

"""Run one model in this process and write per-recording timings.

Model loading happens before the timer. The timer starts at the first read of
the audio file, inside the model transcribe function, and stops after the note
list is built. The accelerator is synchronized on both boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
import traceback
from pathlib import Path

from amt_benchmark.devices import VramTracker, preferred_device, sync_device
from amt_benchmark.runners import LOADERS, parameter_count, parameter_dtype


def _maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Linux":
        return value * 1024
    return value


def _load(model_id: str, repo: Path, weights: Path, conf: Path | None, device: str):
    if model_id == "transkun_v2":
        from amt_benchmark.runners import load_transkun

        return load_transkun(repo, weights, conf, device)
    return LOADERS[model_id](repo, weights, device)


def transcribe_recordings(model_id: str, repo: Path, weights: Path, conf: Path | None, audio_paths: list[str], requested_device: str) -> dict:
    device = preferred_device(requested_device)
    fallback_reason = None
    try:
        module, transcribe = _load(model_id, repo, weights, conf, device)
    except Exception as first_error:
        if device != "cpu":
            fallback_reason = f"{device} load failed: {first_error}"
            device = "cpu"
            module, transcribe = _load(model_id, repo, weights, conf, device)
        else:
            raise
    poll_vram = os.environ.get("AMT_VRAM_POLL", "1") != "0"
    tracker = VramTracker(device, poll=poll_vram)
    try:
        transcribe(audio_paths[0])
    except Exception as warmup_error:
        tracker.finish()
        if device != "cpu":
            fallback_reason = f"{device} warmup failed: {warmup_error}"
            device = "cpu"
            module, transcribe = _load(model_id, repo, weights, conf, device)
            tracker = VramTracker(device, poll=poll_vram)
            transcribe(audio_paths[0])
        else:
            raise
    recordings = []
    for audio_path in audio_paths:
        try:
            sync_device(device)
            started = time.perf_counter()
            notes = transcribe(audio_path)
            sync_device(device)
            elapsed = time.perf_counter() - started
            if device.startswith("mps"):
                import torch

                tracker.note(int(torch.mps.current_allocated_memory()))
            recordings.append(
                {
                    "audio_path": audio_path,
                    "latency_s": elapsed,
                    "notes": [[int(pitch), float(onset), float(offset)] for pitch, onset, offset in notes],
                    "error": None,
                }
            )
        except Exception as error:
            recordings.append(
                {
                    "audio_path": audio_path,
                    "latency_s": None,
                    "notes": [],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    vram = tracker.finish()
    return {
        "model_id": model_id,
        "device": device,
        "device_fallback": fallback_reason,
        "dtype": parameter_dtype(module),
        "n_parameters": parameter_count(module),
        "ru_maxrss_bytes": _maxrss_bytes(),
        "torch": __import__("torch").__version__,
        "recordings": recordings,
        **vram,
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(LOADERS))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", default=None)
    parser.add_argument("--audio", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    output = Path(args.output)
    try:
        payload = transcribe_recordings(
            args.model,
            Path(args.repo),
            Path(args.weights),
            Path(args.conf) if args.conf else None,
            args.audio,
            args.device,
        )
    except Exception as error:
        payload = {
            "model_id": args.model,
            "device": None,
            "dtype": None,
            "n_parameters": None,
            "ru_maxrss_bytes": _maxrss_bytes(),
            "recordings": [],
            "peak_vram_bytes": None,
            "peak_vram_source": None,
            "peak_vram_reason": None,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload))


if __name__ == "__main__":
    main()

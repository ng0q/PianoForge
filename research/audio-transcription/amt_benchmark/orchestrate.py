"""Prepare the ten-recording manifest and run models one process at a time."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import soundfile as sf

from amt_benchmark.manifest import (
    AUDIO_MIRROR_PREFIX,
    AUDIO_URL_PREFIX,
    MIDI_ZIP_URL,
    load_metadata,
    select_recordings,
)
from amt_benchmark.metrics import EVALUATION, reference_notes, score_pair, self_check
from amt_benchmark.pins import CHECKPOINTS, INFERENCE, REPOS

RESULT_NAME = "maestro_v3_test10.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "filename": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git_revision(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def prepare_hft_weights(weights_dir: Path) -> Path:
    archive = weights_dir / CHECKPOINTS["hft_transformer"]["archive_name"]
    target = weights_dir / "hft" / CHECKPOINTS["hft_transformer"]["filename"]
    if target.exists():
        return target
    if not archive.exists():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith("model_016_003.pkl"))
        target.parent.mkdir(parents=True, exist_ok=True)
        with bundle.open(member) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    return target


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "-C",
            "-",
            "-o",
            str(temporary),
            url,
        ],
        check=True,
    )
    temporary.replace(destination)


def _extract_midi(zip_path: Path, relative: str, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as bundle:
        matches = [name for name in bundle.namelist() if name.endswith(relative)]
        if not matches:
            raise FileNotFoundError(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with bundle.open(matches[0]) as source, destination.open("wb") as handle:
            shutil.copyfileobj(source, handle)


def _audio_complete(path: Path, expected_s: float) -> bool:
    if not path.exists() or path.stat().st_size < 10000:
        return False
    try:
        duration = float(sf.info(str(path)).duration)
    except Exception:
        return False
    return abs(duration - expected_s) < max(2.0, 0.05 * expected_s)


def ensure_recording_files(maestro_root: Path, recordings: list[dict]) -> None:
    midi_zip = maestro_root / "maestro-v3.0.0-midi.zip"
    if any(not (maestro_root / row["midi_filename"]).exists() for row in recordings):
        download(MIDI_ZIP_URL, midi_zip)
    for recording in recordings:
        midi_path = maestro_root / recording["midi_filename"]
        if not midi_path.exists():
            _extract_midi(midi_zip, recording["midi_filename"], midi_path)
        recording["midi_source_url"] = MIDI_ZIP_URL
        audio_path = maestro_root / recording["audio_filename"]
        mirror_audio = AUDIO_MIRROR_PREFIX + recording["audio_filename"]
        if not _audio_complete(audio_path, float(recording["duration_s_metadata"])):
            if audio_path.exists():
                audio_path.unlink()
            download(mirror_audio, audio_path)
        recording["audio_source_url"] = mirror_audio
        recording["audio_source_note"] = (
            "Official per-file audio URLs under storage.googleapis.com/magentadata "
            "return 404. Audio is published inside maestro-v3.0.0.zip (101 GB). "
            "This copy uses the same relative path from the ddPn08/maestro-v3.0.0 mirror. "
            "MIDI comes from the official maestro-v3.0.0-midi.zip."
        )


def audio_duration_s(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.duration)


def build_manifest(maestro_root: Path, seed: int = 20261123) -> dict:
    csv_path = maestro_root / "maestro-v3.0.0.csv"
    rows = load_metadata(csv_path)
    selected = select_recordings(rows, seed=seed, count=10)
    ensure_recording_files(maestro_root, selected["recordings"])
    for recording in selected["recordings"]:
        audio_path = maestro_root / recording["audio_filename"]
        midi_path = maestro_root / recording["midi_filename"]
        recording["audio_path"] = str(audio_path)
        recording["midi_path"] = str(midi_path)
        recording["audio_sha256"] = sha256_file(audio_path)
        recording["midi_sha256"] = sha256_file(midi_path)
        recording["audio_bytes"] = audio_path.stat().st_size
        recording["midi_bytes"] = midi_path.stat().st_size
        recording["audio_duration_s"] = audio_duration_s(audio_path)
        recording["audio_duration_source"] = "soundfile.info"
    selected["metadata"] = file_record(csv_path)
    return selected


def _checkpoint_paths(model_id: str, third_party: Path, weights_dir: Path) -> dict:
    spec = CHECKPOINTS[model_id]
    repo_spec = REPOS[model_id]
    if model_id == "hft_transformer":
        weights = prepare_hft_weights(weights_dir)
        repo = third_party / repo_spec["dirname"]
        return {"repo": repo, "weights": weights, "conf": None}
    if model_id == "transkun_v2":
        repo = third_party / repo_spec["dirname"]
        weights = weights_dir / spec["filename"]
        conf = repo / "transkun" / "pretrained" / "2.0.conf"
        return {"repo": repo, "weights": weights, "conf": conf}
    if model_id == "sft_crnn":
        return {
            "repo": third_party / repo_spec["dirname"],
            "weights": weights_dir / spec["filename"],
            "conf": None,
        }
    if model_id == "hrt":
        return {
            "repo": third_party / repo_spec["inference_dirname"],
            "weights": weights_dir / spec["filename"],
            "conf": None,
        }
    if model_id == "onsets_frames":
        return {
            "repo": third_party / "magenta",
            "weights": weights_dir / spec["filename"],
            "conf": None,
        }
    raise KeyError(model_id)


def describe_checkpoints(third_party: Path, weights_dir: Path) -> dict:
    described = {}
    for model_id, spec in CHECKPOINTS.items():
        paths = _checkpoint_paths(model_id, third_party, weights_dir)
        repo = paths["repo"]
        weights = paths["weights"]
        entry = {
            "name": REPOS[model_id]["name"],
            "repository": REPOS[model_id]["url"],
            "revision": git_revision(repo) if repo.exists() else REPOS[model_id].get("revision"),
            "expected_revision": REPOS[model_id].get("revision"),
            "weight_url": spec["url"],
            "variant": spec["variant"],
            "inference": INFERENCE[model_id],
        }
        if model_id == "hrt":
            entry["implementation_repository"] = REPOS[model_id]["inference_url"]
            entry["implementation_revision"] = git_revision(repo)
            entry["paper_repository"] = REPOS[model_id]["url"]
            entry["paper_revision"] = git_revision(third_party / REPOS[model_id]["dirname"])
        if weights.exists() and weights.stat().st_size > 0 and not _is_html(weights):
            entry["weights"] = file_record(weights)
            entry["available"] = True
            entry["unavailable_reason"] = None
        else:
            entry["weights"] = None
            entry["available"] = False
            entry["unavailable_reason"] = f"weight file is missing: {weights.name}"
        if spec.get("published_sha256") and entry["weights"]:
            matches = entry["weights"]["sha256"] == spec["published_sha256"]
            entry["sha256_matches_published_author_file"] = matches
            entry["published_sha256"] = spec["published_sha256"]
            if model_id == "hrt":
                entry["mirror_url"] = (
                    "https://huggingface.co/Genius-Society/piano_trans/resolve/main/"
                    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth"
                )
                entry["weight_acquisition"] = (
                    "Zenodo returned HTTP 403 from this network. "
                    "The local file is the Hugging Face copy of the same author checkpoint, "
                    "checked against the published SHA-256."
                )
                if not matches:
                    entry["available"] = False
                    entry["unavailable_reason"] = (
                        "local HRT file SHA-256 does not match the published author checkpoint"
                    )
        described[model_id] = entry
    return described


def _is_html(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(64).lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def library_versions() -> dict:
    import importlib.metadata as metadata

    import numpy
    import pretty_midi
    import mir_eval

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": numpy.__version__,
        "mir_eval": mir_eval.__version__,
        "pretty_midi": pretty_midi.__version__,
    }
    # Do not import torch here. Importing it initializes MPS in this process,
    # and a second process using MPS can stall inside the Metal allocator.
    try:
        versions["torch"] = metadata.version("torch")
    except metadata.PackageNotFoundError as error:
        versions["torch"] = None
        versions["torch_import_error"] = str(error)
    for name in ("torchaudio", "librosa", "soundfile"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def _peak_rss(process) -> int:
    import psutil

    peak = 0
    ps_process = psutil.Process(process.pid)
    while process.poll() is None:
        try:
            peak = max(peak, int(ps_process.memory_info().rss))
        except psutil.Error:
            break
        time.sleep(0.1)
    try:
        peak = max(peak, int(ps_process.memory_info().rss))
    except psutil.Error:
        pass
    return peak


def run_model_process(python: str, model_id: str, repo: Path, weights: Path, conf: Path | None, audio_paths: list[str], output: Path, device: str) -> dict:
    weights_missing = (not weights.exists()) or weights.stat().st_size == 0 or _is_html(weights)
    if weights_missing and model_id != "onsets_frames":
        return {
            "model_id": model_id,
            "error": f"author weights are not available at {weights}",
            "recordings": [],
            "device": None,
            "dtype": None,
            "n_parameters": None,
            "peak_vram_bytes": None,
            "peak_vram_source": None,
            "peak_vram_reason": "model did not run",
            "peak_ram_bytes": None,
            "peak_ram_source": None,
            "peak_ram_reason": "model did not run",
        }
    command = [
        python,
        "-m",
        "amt_benchmark.worker",
        "--model",
        model_id,
        "--repo",
        str(repo),
        "--weights",
        str(weights),
        "--output",
        str(output),
        "--device",
        device,
        "--audio",
        *audio_paths,
    ]
    if conf is not None:
        command.extend(["--conf", str(conf)])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]
    )
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env["PYTHONUNBUFFERED"] = "1"
    log_path = output.with_suffix(".log")
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    sampled_rss = _peak_rss(process)
    return_code = process.wait()
    if not output.exists():
        return {
            "model_id": model_id,
            "error": f"worker exited {return_code} without a result file",
            "recordings": [],
            "peak_ram_bytes": sampled_rss or None,
            "peak_ram_source": "psutil.Process.memory_info().rss sampled every 100 ms",
        }
    payload = json.loads(output.read_text())
    worker_rss = payload.get("ru_maxrss_bytes") or 0
    payload["peak_ram_bytes"] = max(int(sampled_rss), int(worker_rss))
    payload["peak_ram_source"] = (
        "max of psutil.Process.memory_info().rss sampled every 100 ms by the parent "
        "and resource.getrusage(RUSAGE_SELF).ru_maxrss of the worker "
        "(bytes on macOS)"
    )
    payload["peak_ram_reason"] = None
    return payload


def _macro(values: list[dict]) -> dict | None:
    if not values:
        return None
    return {
        key: float(sum(item[key] for item in values) / len(values))
        for key in ("precision", "recall", "f1")
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def summarize_model(model_id: str, worker: dict, recordings: list[dict], references: dict[str, list]) -> dict:
    per_recording = []
    onset_scores = []
    offset_scores = []
    latencies = []
    latency_sum = 0.0
    duration_sum = 0.0
    for recording in recordings:
        audio_name = recording["audio_filename"]
        match = next(
            (item for item in worker.get("recordings", []) if Path(item["audio_path"]).name == Path(audio_name).name or item["audio_path"].endswith(audio_name)),
            None,
        )
        row = {
            "audio_filename": audio_name,
            "midi_filename": recording["midi_filename"],
            "audio_duration_s": recording["audio_duration_s"],
        }
        if match is None or match.get("error"):
            row.update(
                {
                    "latency_s": None,
                    "rtf": None,
                    "onset": None,
                    "onset_offset": None,
                    "n_ref_notes": None,
                    "n_est_notes": None,
                    "error": None if match is None else match.get("error"),
                }
            )
            if match is None and worker.get("error"):
                row["error"] = worker["error"]
        else:
            scored = score_pair(references[audio_name], match["notes"])
            latency = float(match["latency_s"])
            duration = float(recording["audio_duration_s"])
            row.update(
                {
                    "latency_s": latency,
                    "rtf": latency / duration,
                    "onset": scored["onset"],
                    "onset_offset": scored["onset_offset"],
                    "n_ref_notes": scored["n_ref_notes"],
                    "n_est_notes": scored["n_est_notes"],
                    "error": None,
                }
            )
            onset_scores.append(scored["onset"])
            offset_scores.append(scored["onset_offset"])
            latencies.append(latency)
            latency_sum += latency
            duration_sum += duration
        per_recording.append(row)
    n_success = sum(1 for row in per_recording if row["error"] is None and row["onset"] is not None)
    if worker.get("error") and n_success == 0:
        status = "failed"
    elif n_success == len(recordings):
        status = "complete"
    elif n_success == 0:
        status = "failed"
    else:
        status = "partial"
    return {
        "model_id": model_id,
        "name": REPOS[model_id]["name"],
        "status": status,
        "n_success": n_success,
        "n_expected": len(recordings),
        "partial": status == "partial",
        "device": worker.get("device"),
        "device_fallback": worker.get("device_fallback"),
        "dtype": worker.get("dtype"),
        "n_parameters": worker.get("n_parameters"),
        "error": worker.get("error"),
        "macro_onset": _macro(onset_scores),
        "macro_onset_offset": _macro(offset_scores),
        "macro_over_n": n_success,
        "median_latency_s": _median(latencies),
        "overall_rtf": (latency_sum / duration_sum) if duration_sum else None,
        "latency_sum_s": latency_sum if n_success else None,
        "audio_duration_sum_s": duration_sum if n_success else None,
        "peak_ram_bytes": worker.get("peak_ram_bytes"),
        "peak_ram_source": worker.get("peak_ram_source"),
        "peak_ram_reason": worker.get("peak_ram_reason"),
        "peak_vram_bytes": worker.get("peak_vram_bytes"),
        "peak_vram_source": worker.get("peak_vram_source"),
        "peak_vram_reason": worker.get("peak_vram_reason"),
        "recordings": per_recording,
    }


def speed_groups(models: dict) -> dict:
    groups: dict[str, list[str]] = {}
    for model_id, summary in models.items():
        device = summary.get("device") or "not_run"
        groups.setdefault(device, []).append(model_id)
    return groups


def run_benchmark(maestro_root: Path, third_party: Path, weights_dir: Path, python: str, output_json: Path, model_ids: list[str], device: str = "auto", recording_limit: int | None = None, seed: int = 20261123, cache_dir: Path | None = None) -> dict:
    manifest = build_manifest(maestro_root, seed=seed)
    recordings = manifest["recordings"]
    if recording_limit is not None:
        recordings = recordings[:recording_limit]
    checkpoints = describe_checkpoints(third_party, weights_dir)
    references = {
        recording["audio_filename"]: reference_notes(recording["midi_path"])
        for recording in recordings
    }
    audio_paths = [recording["audio_path"] for recording in recordings]
    work = cache_dir or (weights_dir.parent / "worker-output")
    work.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    model_payloads = {}
    summaries = {}

    def _write() -> dict:
        for model_id, summary in summaries.items():
            weights = checkpoints[model_id].get("weights")
            summary["weight_bytes"] = None if not weights else weights["bytes"]
            summary["weight_sha256"] = None if not weights else weights["sha256"]
            summary["weight_filename"] = None if not weights else weights["filename"]
        document = {
            "benchmark": "maestro_v3_test10",
            "manifest": manifest,
            "configuration": {
                "evaluation": EVALUATION,
                "evaluation_self_check": self_check(),
                "timing": {
                    "warmup": "one untimed full transcription of the first selected recording",
                    "latency": "perf_counter from the model's audio read through the note list",
                    "includes": ["preprocessing", "inference", "decoding"],
                    "excludes": ["model load", "metric computation"],
                    "accelerator_sync": "before the timer starts and before it stops",
                    "rtf": "latency_s / audio_duration_s",
                    "overall_rtf": "sum(latency_s) / sum(audio_duration_s) over successful recordings",
                    "median_latency": "median of per-recording latency_s",
                },
                "inference": {model_id: INFERENCE[model_id] for model_id in model_ids},
                "recording_limit": recording_limit,
                "device_request": device,
            },
            "environment": library_versions(),
            "checkpoints": {model_id: checkpoints[model_id] for model_id in model_ids},
            "recordings": {
                model_id: summaries[model_id]["recordings"] for model_id in summaries
            },
            "models": {
                model_id: {key: value for key, value in summaries[model_id].items() if key != "recordings"}
                for model_id in summaries
            },
            "speed_results_by_device": speed_groups(summaries),
            "errors": [
                {"model_id": model_id, "error": summaries[model_id]["error"]}
                for model_id in summaries
                if summaries[model_id]["error"]
            ],
        }
        temporary = output_json.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, indent=2))
        temporary.replace(output_json)
        return document

    for model_id in model_ids:
        paths = _checkpoint_paths(model_id, third_party, weights_dir)
        print(f"running {model_id}", flush=True)
        model_payloads[model_id] = run_model_process(
            python,
            model_id,
            paths["repo"],
            paths["weights"],
            paths["conf"],
            audio_paths,
            work / f"{model_id}.json",
            device,
        )
        summaries[model_id] = summarize_model(
            model_id, model_payloads[model_id], recordings, references
        )
        document = _write()
        print(
            f"finished {model_id} status={summaries[model_id]['status']} "
            f"n_success={summaries[model_id]['n_success']}",
            flush=True,
        )
    return document


def summary_table(document: dict) -> list[dict]:
    rows = []
    for model_id, summary in document["models"].items():
        onset = summary["macro_onset"] or {}
        both = summary["macro_onset_offset"] or {}
        rows.append(
            {
                "model": summary["name"],
                "status": summary["status"],
                "n_success": summary["n_success"],
                "device": summary["device"],
                "dtype": summary["dtype"],
                "onset_f1": onset.get("f1"),
                "onset_precision": onset.get("precision"),
                "onset_recall": onset.get("recall"),
                "onset_offset_f1": both.get("f1"),
                "onset_offset_precision": both.get("precision"),
                "onset_offset_recall": both.get("recall"),
                "median_latency_s": summary["median_latency_s"],
                "overall_rtf": summary["overall_rtf"],
                "peak_ram_bytes": summary["peak_ram_bytes"],
                "peak_vram_bytes": summary["peak_vram_bytes"],
                "n_parameters": summary["n_parameters"],
                "weight_bytes": summary["weight_bytes"],
            }
        )
    return rows

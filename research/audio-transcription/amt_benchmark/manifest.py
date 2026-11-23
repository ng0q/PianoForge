"""Select ten MAESTRO v3.0.0 test recordings of different compositions."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SEED = 20261123
METADATA_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv"
AUDIO_URL_PREFIX = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
# The official release publishes audio only inside the 101 GB zip. This mirror
# keeps the official relative paths. MIDI bytes are taken from the official
# maestro-v3.0.0-midi.zip and checked against the mirror when both exist.
AUDIO_MIRROR_PREFIX = "https://huggingface.co/datasets/ddPn08/maestro-v3.0.0/resolve/main/"
MIDI_ZIP_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"

SELECTION = {
    "dataset": "MAESTRO",
    "version": "3.0.0",
    "metadata_url": METADATA_URL,
    "split": "test",
    "composition_key": ["canonical_composer", "canonical_title"],
    "candidate_sort": ["canonical_composer", "canonical_title", "midi_filename"],
    "one_recording_per_composition": "first row after candidate_sort",
    "seed": SEED,
    "sampler": "numpy.random.Generator.choice, replace=False",
    "selected_order": "sampled rows sorted again by candidate_sort",
}


def _sort_key(row: dict) -> tuple:
    return (
        row["canonical_composer"],
        row["canonical_title"],
        row["midi_filename"],
    )


def load_metadata(csv_path: str | Path) -> list[dict]:
    with open(csv_path, newline="") as handle:
        return list(csv.DictReader(handle))


def select_recordings(rows: list[dict], seed: int = SEED, count: int = 10) -> dict:
    test_rows = [row for row in rows if row["split"] == "test"]
    test_rows.sort(key=_sort_key)
    candidates = []
    seen = set()
    for row in test_rows:
        key = (row["canonical_composer"], row["canonical_title"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(row)
    if len(candidates) < count:
        raise RuntimeError(f"Need {count} test compositions, found {len(candidates)}")
    generator = np.random.default_rng(seed)
    sampled = generator.choice(len(candidates), size=count, replace=False)
    sampled_indices = [int(index) for index in sampled]
    chosen = [candidates[index] for index in sorted(sampled_indices)]
    recordings = []
    for order, row in enumerate(chosen):
        recordings.append(
            {
                "order": order,
                "canonical_composer": row["canonical_composer"],
                "canonical_title": row["canonical_title"],
                "year": int(row["year"]) if str(row["year"]).isdigit() else row["year"],
                "split": row["split"],
                "audio_filename": row["audio_filename"],
                "midi_filename": row["midi_filename"],
                "duration_s_metadata": float(row["duration"]),
            }
        )
    return {
        "selection": {
            **SELECTION,
            "seed": seed,
            "n_test_recordings": len(test_rows),
            "n_candidate_compositions": len(candidates),
            "sampled_candidate_indices": sampled_indices,
        },
        "recordings": recordings,
    }

"""Note events and mir_eval transcription scores.

Reference notes are MIDI note-on/note-off times. Sustain-pedal control
changes are not applied, so note offsets are not extended by the pedal.
Pitches are passed to mir_eval in hertz. mir_eval then compares them in cents.
"""

from __future__ import annotations

import numpy as np
import pretty_midi

ONSET_TOLERANCE_S = 0.05
OFFSET_RATIO = 0.2
OFFSET_MIN_TOLERANCE_S = 0.05
# 50 cents is one half-step short of a semitone, so equal MIDI pitches match
# and neighboring pitches do not when both sides use the same Hz conversion.
PITCH_TOLERANCE_CENTS = 50.0

EVALUATION = {
    "event": "(midi_pitch, onset_s, offset_s)",
    "reference": "pretty_midi note-on/note-off; CC64 sustain is not applied",
    "pedal_extension": False,
    "pitch_hz": "440 * 2 ** ((midi_pitch - 69) / 12)",
    "function": "mir_eval.transcription.precision_recall_f1_overlap",
    "pitch_tolerance_cents": PITCH_TOLERANCE_CENTS,
    "onset": {
        "onset_tolerance_s": ONSET_TOLERANCE_S,
        "offset_ratio": None,
        "pitch_tolerance_cents": PITCH_TOLERANCE_CENTS,
    },
    "onset_offset": {
        "onset_tolerance_s": ONSET_TOLERANCE_S,
        "offset_ratio": OFFSET_RATIO,
        "offset_min_tolerance_s": OFFSET_MIN_TOLERANCE_S,
        "offset_tolerance": "max(50 ms, 20% of the reference note duration)",
        "pitch_tolerance_cents": PITCH_TOLERANCE_CENTS,
    },
    "drop_nonpositive_duration": True,
}


def midi_to_hz(midi_pitch: float) -> float:
    return 440.0 * (2.0 ** ((float(midi_pitch) - 69.0) / 12.0))


def reference_notes(midi_path: str) -> list[tuple[int, float, float]]:
    """Read note-off offsets. Pedal CC events stay control changes."""
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes: list[tuple[int, float, float]] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            end = float(note.end)
            start = float(note.start)
            if end <= start:
                continue
            notes.append((int(note.pitch), start, end))
    notes.sort(key=lambda item: (item[1], item[0], item[2]))
    return notes


def clean_notes(notes) -> list[tuple[int, float, float]]:
    cleaned = []
    for pitch, onset, offset in notes:
        pitch_i = int(pitch)
        onset_f = float(onset)
        offset_f = float(offset)
        if offset_f <= onset_f:
            continue
        if pitch_i < 0 or pitch_i > 127:
            continue
        cleaned.append((pitch_i, onset_f, offset_f))
    cleaned.sort(key=lambda item: (item[1], item[0], item[2]))
    return cleaned


def _arrays(notes: list[tuple[int, float, float]]):
    if not notes:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    intervals = np.array([[onset, offset] for _, onset, offset in notes], dtype=np.float64)
    pitches = np.array([midi_to_hz(pitch) for pitch, _, _ in notes], dtype=np.float64)
    return intervals, pitches


def _empty_scores(n_ref: int, n_est: int) -> dict[str, float]:
    if n_ref == 0 and n_est == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def score_notes(reference, estimated, offset_ratio):
    reference = clean_notes(reference)
    estimated = clean_notes(estimated)
    if not reference or not estimated:
        return _empty_scores(len(reference), len(estimated))
    import mir_eval.transcription

    ref_intervals, ref_pitches = _arrays(reference)
    est_intervals, est_pitches = _arrays(estimated)
    precision, recall, f1, _overlap = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=ONSET_TOLERANCE_S,
        pitch_tolerance=PITCH_TOLERANCE_CENTS,
        offset_ratio=offset_ratio,
        offset_min_tolerance=OFFSET_MIN_TOLERANCE_S,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def score_pair(reference, estimated) -> dict:
    reference = clean_notes(reference)
    estimated = clean_notes(estimated)
    return {
        "n_ref_notes": len(reference),
        "n_est_notes": len(estimated),
        "onset": score_notes(reference, estimated, offset_ratio=None),
        "onset_offset": score_notes(reference, estimated, offset_ratio=OFFSET_RATIO),
    }


def _assert_close(actual: dict, precision: float, recall: float, f1: float) -> None:
    assert abs(actual["precision"] - precision) < 1e-6, actual
    assert abs(actual["recall"] - recall) < 1e-6, actual
    assert abs(actual["f1"] - f1) < 1e-6, actual


def self_check() -> dict:
    """Score a hand-built pair whose matches are known before any model runs.

    Reference: C4, E4, G4.
    Estimate: C4 in tolerance, E4 onset late by 200 ms, G4 onset inside
    50 ms but offset outside max(50 ms, 20%), plus an extra C5.
    """
    reference = [(60, 0.0, 1.0), (64, 1.0, 2.0), (67, 2.0, 3.0)]
    estimated = [(60, 0.01, 1.0), (64, 1.20, 2.0), (67, 2.02, 2.50), (72, 3.0, 3.5)]
    scores = score_pair(reference, estimated)
    _assert_close(scores["onset"], 0.5, 2.0 / 3.0, 4.0 / 7.0)
    _assert_close(scores["onset_offset"], 0.25, 1.0 / 3.0, 2.0 / 7.0)

    mismatch = score_pair([(60, 0.0, 1.0)], [(61, 0.0, 1.0)])
    _assert_close(mismatch["onset"], 0.0, 0.0, 0.0)
    _assert_close(mismatch["onset_offset"], 0.0, 0.0, 0.0)

    # Duration 1.0 s -> offset tolerance max(0.05, 0.20) = 0.20 s.
    on_boundary = score_pair([(60, 0.0, 1.0)], [(60, 0.0, 1.2)])
    _assert_close(on_boundary["onset_offset"], 1.0, 1.0, 1.0)
    past_boundary = score_pair([(60, 0.0, 1.0)], [(60, 0.0, 1.21)])
    _assert_close(past_boundary["onset"], 1.0, 1.0, 1.0)
    _assert_close(past_boundary["onset_offset"], 0.0, 0.0, 0.0)

    # Duration 0.10 s -> offset tolerance max(0.05, 0.02) = 0.05 s.
    short_ok = score_pair([(60, 0.0, 0.1)], [(60, 0.0, 0.15)])
    _assert_close(short_ok["onset_offset"], 1.0, 1.0, 1.0)
    short_miss = score_pair([(60, 0.0, 0.1)], [(60, 0.0, 0.16)])
    _assert_close(short_miss["onset_offset"], 0.0, 0.0, 0.0)

    onset_ok = score_pair([(60, 0.0, 1.0)], [(60, 0.05, 1.0)])
    _assert_close(onset_ok["onset"], 1.0, 1.0, 1.0)
    onset_miss = score_pair([(60, 0.0, 1.0)], [(60, 0.06, 1.0)])
    _assert_close(onset_miss["onset"], 0.0, 0.0, 0.0)

    return {
        "passed": True,
        "case": "synthetic_three_reference_notes",
        "onset": scores["onset"],
        "onset_offset": scores["onset_offset"],
        "evaluation": EVALUATION,
    }

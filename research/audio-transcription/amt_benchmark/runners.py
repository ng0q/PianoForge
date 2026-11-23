"""Load one author implementation and transcribe audio inside the caller.

Each function starts by reading the audio file. The worker places the timer
around the call, after the model is already loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _insert(path: Path) -> None:
    location = str(path)
    if location not in sys.path:
        sys.path.insert(0, location)


def parameter_count(module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def parameter_dtype(module) -> str:
    dtypes = {str(parameter.dtype) for parameter in module.parameters()}
    if len(dtypes) == 1:
        return next(iter(dtypes))
    return ",".join(sorted(dtypes))


def load_hft(repo: Path, weights: Path, device: str):
    _insert(repo)
    import io
    import json

    import numpy as np
    import torch
    from model.amt import AMT

    def _load_storage_on_cpu(blob):
        return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)

    original = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = _load_storage_on_cpu
    try:
        config_path = repo / "corpus" / "config.json"
        with config_path.open() as handle:
            config = json.load(handle)
        config["input"]["max_value"] = 0.0
        if config["feature"]["log_offset"] > 0.0:
            config["input"]["min_value"] = float(np.log(config["feature"]["log_offset"]))
        else:
            config["input"]["min_value"] = float(config["feature"]["log_offset"])
        config["feature"]["n_bins"] = config["feature"]["mel_bins"]
        engine = AMT(config, str(weights), verbose_flag=False)
    finally:
        torch.storage._load_from_bytes = original
    engine.model = engine.model.to(device)
    engine.device = device

    def _move_loose_tensors(module):
        for child in module.children():
            _move_loose_tensors(child)
        for key, value in list(vars(module).items()):
            if torch.is_tensor(value):
                setattr(module, key, value.to(device))

    _move_loose_tensors(engine.model)
    for submodule in engine.model.modules():
        if hasattr(submodule, "device"):
            submodule.device = device
    engine.model.eval()

    def transcribe(audio_path: str):
        feature = engine.wav2feature(audio_path)
        outputs = engine.transcript(feature, mode="combination", ablation_flag=False)
        notes = engine.mpe2note(
            a_onset=outputs[4],
            a_offset=outputs[5],
            a_mpe=outputs[6],
            a_velocity=outputs[7],
            thred_onset=0.5,
            thred_offset=0.5,
            thred_mpe=0.5,
            mode_velocity="ignore_zero",
            mode_offset="shorter",
        )
        return [(int(note["pitch"]), float(note["onset"]), float(note["offset"])) for note in notes]

    return engine.model, transcribe


def _transkun_checkpoint(weights: Path, fallback_conf: Path):
    """Return (state, conf_path). The V2 Drive file is a zip with its own conf."""
    import io
    import zipfile

    import torch

    if zipfile.is_zipfile(weights):
        with zipfile.ZipFile(weights) as bundle:
            conf_name = next(name for name in bundle.namelist() if name.endswith("model.conf"))
            weight_name = next(
                name for name in bundle.namelist() if name.endswith(".pt") and not name.endswith(".conf")
            )
            conf_path = weights.with_name("transkun_v2.model.conf")
            conf_path.write_bytes(bundle.read(conf_name))
            checkpoint = torch.load(io.BytesIO(bundle.read(weight_name)), map_location="cpu", weights_only=False)
        return checkpoint, conf_path
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    return checkpoint, fallback_conf


def load_transkun(repo: Path, weights: Path, conf: Path, device: str):
    _insert(repo)
    import numpy as np
    import torch
    import moduleconf
    from transkun.ModelTransformer import TransKun

    checkpoint, conf_path = _transkun_checkpoint(weights, conf)
    conf_manager = moduleconf.parseFromFile(str(conf_path))
    model_conf = conf_manager["Model"].config
    model = TransKun(conf=model_conf)
    state = checkpoint["best_state_dict"] if "best_state_dict" in checkpoint else checkpoint["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 20:
        raise RuntimeError(
            f"Transkun checkpoint does not match ModelTransformer "
            f"({len(missing)} missing keys, {len(unexpected)} unexpected keys)"
        )
    model = model.to(device)
    model.eval()

    def transcribe(audio_path: str):
        import pydub
        import soxr

        audio = pydub.AudioSegment.from_file(audio_path)
        samples = np.array(audio.get_array_of_samples())
        samples = samples.reshape(-1, audio.channels)
        if audio.sample_width != 2:
            raise RuntimeError(f"expected 16-bit PCM, got sample width {audio.sample_width}")
        waveform = np.float32(samples) / np.float32(2**15)
        if audio.frame_rate != model.fs:
            waveform = soxr.resample(waveform, audio.frame_rate, model.fs)
        tensor = torch.from_numpy(waveform).to(device)
        with torch.no_grad():
            estimated = model.transcribe(
                tensor,
                stepInSecond=None,
                segmentSizeInSecond=None,
                discardSecondHalf=False,
            )
        notes = []
        for note in estimated:
            if note.pitch <= 0:
                continue
            notes.append((int(note.pitch), float(note.start), float(note.end)))
        return notes

    return model, transcribe


def load_sft(repo: Path, weights: Path, device: str):
    _insert(repo)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchaudio
    from hydra import compose, initialize_config_dir
    from nnAudio.features import CQT
    from data.constants import DEFAULT_HOP_WIDTH, DEFAULT_SAMPLE_RATE, MIDI_MIN
    from data.notes_parser import pianoroll_to_notes_list
    from model.SFT_CRNN import SFT_CRNN_Model

    with initialize_config_dir(version_base=None, config_dir=str(repo / "config")):
        config = compose(config_name="main_config")
    config.training.mode = "test"
    config.data.n_frames = 1200
    config.model.dropout_rate = 0.0

    hop_length = int(config.data.hop_length)
    if config.model.inputs_fine_time_res:
        hop_length = hop_length // 2
    extractor = CQT(
        sr=int(config.data.sample_rate),
        hop_length=hop_length,
        fmin=float(config.data.cqt_fmin),
        n_bins=int(config.data.num_cqt_bins),
        bins_per_octave=int(config.data.bins_per_octave),
    )
    model = SFT_CRNN_Model(config.model)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    ignore = set(config.model.checkpoint_ignore_layres or [])
    cleaned = {}
    for key, value in state.items():
        if key in ignore or not hasattr(value, "shape"):
            continue
        if key.startswith("model."):
            cleaned[key[len("model.") :]] = value
        else:
            cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    weight_missing = [key for key in missing if "mask" not in key and "pos" not in key]
    if len(weight_missing) > 20:
        raise RuntimeError(
            f"SFT-CRNN checkpoint does not match the model "
            f"({len(missing)} missing keys, first {weight_missing[:8]}, "
            f"{len(unexpected)} unexpected keys)"
        )
    model = model.to(device).eval()
    extractor = extractor.to(device).eval()
    batch_size = int(config.training.batch_inference)
    n_frames = int(config.data.n_frames)
    use_db = bool(config.data.amplitude_to_db) and config.data.features != "mel"

    def transcribe(audio_path: str):
        import librosa

        waveform, sample_rate = librosa.load(audio_path, sr=None, mono=False)
        waveform = torch.from_numpy(np.asarray(waveform)).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(device)
        if sample_rate != DEFAULT_SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sample_rate, DEFAULT_SAMPLE_RATE).to(device)(waveform)
        with torch.no_grad():
            features = extractor(waveform[:, :-1]).transpose(-1, -2)
            if use_db:
                features = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(features)
            if features.shape[1] % n_frames != 0:
                pad = n_frames - (features.shape[1] % n_frames)
                features = F.pad(features, (0, 0, 0, pad), value=0.0)
            clips = features.view(features.shape[1] // n_frames, n_frames, -1)
            onset_parts = []
            offset_parts = []
            for start in range(0, clips.shape[0], batch_size):
                batch = clips[start : start + batch_size]
                generated = model.generate(batch, global_rank=0)
                onset_parts.append(torch.sigmoid(generated["outputs_pianoroll_onsets"]))
                offset_parts.append(torch.sigmoid(generated["outputs_pianoroll_offsets"]))
            onsets = torch.cat(onset_parts, dim=0)
            offsets = torch.cat(offset_parts, dim=0)
        _batch, _time, pitches = onsets.shape
        intervals, pitch_index = pianoroll_to_notes_list(
            onsets.reshape(-1, pitches),
            offset_pianoroll=offsets.reshape(-1, pitches),
            threshold=0.5,
        )
        resolution = DEFAULT_HOP_WIDTH / DEFAULT_SAMPLE_RATE
        intervals = np.asarray(intervals, dtype=np.float64) * resolution
        pitch_index = np.asarray(pitch_index) + MIDI_MIN
        notes = []
        for interval, pitch in zip(intervals, pitch_index):
            if pitch < 0:
                continue
            onset = float(interval[0])
            duration = float(np.clip(interval[1] - interval[0], 0.05, 10.0))
            notes.append((int(pitch), onset, onset + duration))
        return notes

    return model, transcribe


def _install_librosa_audio_shim() -> None:
    """Keep the author load_audio calls working when librosa has no core.audio."""
    import types

    import librosa
    import librosa.core

    try:
        import librosa.core.audio as audio
    except Exception:
        audio = None
    if audio is not None:
        original = audio.resample

        def resample(y, orig_sr, target_sr, res_type="kaiser_best", **kwargs):
            # Author code calls resample(y, orig_sr, target_sr). Current librosa
            # takes orig_sr and target_sr as keyword-only arguments.
            return original(y, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type, **kwargs)

        audio.resample = resample
        return

    resample_fn = librosa.resample
    to_mono_fn = librosa.to_mono
    buf_to_float = librosa.util.buf_to_float
    audio = types.ModuleType("librosa.core.audio")
    util = types.ModuleType("librosa.core.audio.util")
    util.buf_to_float = buf_to_float
    audio.util = util
    audio.to_mono = to_mono_fn

    def resample(y, orig_sr, target_sr, res_type="kaiser_best"):
        return resample_fn(y, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type)

    audio.resample = resample
    sys.modules["librosa.core.audio"] = audio
    sys.modules["librosa.core.audio.util"] = util
    librosa.core.audio = audio


def load_hrt(repo: Path, weights: Path, device: str):
    _insert(repo)
    _install_librosa_audio_shim()
    import torch
    from piano_transcription_inference import PianoTranscription, load_audio, sample_rate

    # The author class moves the module only when the device string contains "cuda".
    engine = PianoTranscription(model_type="Note_pedal", checkpoint_path=str(weights), device=torch.device("cpu"))
    engine.model = engine.model.to(device)
    engine.model.eval()

    def transcribe(audio_path: str):
        audio, _rate = load_audio(audio_path, sr=sample_rate, mono=True)
        transcribed = engine.transcribe(audio, None)
        notes = []
        for event in transcribed["est_note_events"]:
            notes.append((int(event["midi_note"]), float(event["onset_time"]), float(event["offset_time"])))
        return notes

    return engine.model, transcribe


def load_onsets_frames(repo: Path, weights: Path, device: str):
    raise RuntimeError(
        "Magenta Onsets and Frames publishes maestro_checkpoint.zip for TensorFlow. "
        "magenta 2.1.4 does not install on Python 3.12: it requires Python <3.9 and "
        "TensorFlow 2.3, and this host has no CUDA. The official transcribe entry "
        "point was not started."
    )


LOADERS = {
    "hft_transformer": load_hft,
    "transkun_v2": load_transkun,
    "sft_crnn": load_sft,
    "hrt": load_hrt,
    "onsets_frames": load_onsets_frames,
}

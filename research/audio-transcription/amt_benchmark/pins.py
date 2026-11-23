"""Author repositories, revisions, and checkpoints fixed before the run."""

from __future__ import annotations

REPOS = {
    "hft_transformer": {
        "name": "hFT-Transformer",
        "url": "https://github.com/sony/hFT-Transformer",
        "revision": "226992b071b379b29232cd3df99c9b263cc8c312",
        "dirname": "hFT-Transformer",
    },
    "transkun_v2": {
        "name": "Transkun V2",
        "url": "https://github.com/Yujia-Yan/Transkun",
        "revision": "c5cb8370e17b4a1650b5971ba87ee4b0d208e5b6",
        "dirname": "Transkun",
    },
    "sft_crnn": {
        "name": "SFT-CRNN",
        "url": "https://github.com/WX-Wei/AMT-optimal-transport",
        "revision": "21e69640b7c6c8a2860dab9a8b612e3bab18b87e",
        "dirname": "AMT-optimal-transport",
    },
    "hrt": {
        "name": "HRT",
        "url": "https://github.com/bytedance/piano_transcription",
        "revision": "1ade7dcd4348add669a67c6e6282456c8c6633bd",
        "dirname": "piano_transcription",
        "inference_url": "https://github.com/qiuqiangkong/piano_transcription_inference",
        "inference_revision": "566057930a8423aa31b91296e7002c513269446d",
        "inference_dirname": "piano_transcription_inference",
    },
    "onsets_frames": {
        "name": "OnF",
        "url": "https://github.com/magenta/magenta",
        "tree": "magenta/models/onsets_frames_transcription",
        "revision": None,
    },
}

CHECKPOINTS = {
    "hft_transformer": {
        "variant": "MAESTRO-V3 model_016_003",
        "url": "https://github.com/sony/hFT-Transformer/releases/download/ismir2023/checkpoint.zip",
        "archive_name": "hft-checkpoint.zip",
        "filename": "model_016_003.pkl",
    },
    "transkun_v2": {
        "variant": "Transkun V2",
        "url": "https://drive.google.com/file/d/1pxGpO8eCdFxMRrXi_YUh7_uC0Ae26coB/view",
        "file_id": "1pxGpO8eCdFxMRrXi_YUh7_uC0Ae26coB",
        "filename": "transkun_v2.pt",
        "conf_filename": "model.conf inside the weight archive",
    },
    "sft_crnn": {
        "variant": "SFT-CRNN Piano MAESTRO OT loss, step 340609",
        "url": "https://github.com/WX-Wei/AMT-optimal-transport/releases/download/v0.0.1/SFT-CRNN_Piano-MAESTRO_OTLoss_steps.340609.ckpt",
        "filename": "SFT-CRNN_Piano-MAESTRO_OTLoss_steps.340609.ckpt",
    },
    "hrt": {
        "variant": "Note_pedal CRNN_note_F1=0.9677_pedal_F1=0.9186",
        "url": "https://zenodo.org/records/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1",
        "filename": "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth",
        "published_sha256": "c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141",
    },
    "onsets_frames": {
        "variant": "Magenta Onsets and Frames, MAESTRO checkpoint",
        "url": "https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip",
        "filename": "maestro_checkpoint.zip",
    },
}

# Author inference settings. These stay fixed before the timed run.
INFERENCE = {
    "hft_transformer": {
        "mode": "combination",
        "scored_output": "2nd",
        "n_stride": 0,
        "ablation": False,
        "thred_onset": 0.5,
        "thred_offset": 0.5,
        "thred_mpe": 0.5,
        "mode_velocity": "ignore_zero",
        "mode_offset": "shorter",
        "config": "corpus/config.json plus min_value=log(log_offset) and max_value=0 from corpus/make_dataset.py",
        "preprocessing": "AMT.wav2feature",
        "decoding": "AMT.transcript + AMT.mpe2note",
    },
    "transkun_v2": {
        "segmentHopSizeInSecond": 8,
        "segmentSizeInSecond": 16,
        "discardSecondHalf": False,
        "mergeIncompleteEvent": True,
        "fs": 44100,
        "audio_normalization": "int16 / 2**15",
        "resample": "soxr to model.fs when the file rate differs",
        "wav_reader": "pydub.AudioSegment.from_file; upstream readAudio calls from_mp3",
        "note_filter": "pitch > 0, matching Data.writeMidi",
        "decoding": "ModelTransformer.transcribe",
        "config": (
            "model.conf inside the Google Drive archive. "
            "scoringExpansionFactor=2 and contextDropoutProb=0.1. "
            "The repository file pretrained/2.0.conf belongs to Transkun V2 No Pedal Ext "
            "and is not used."
        ),
    },
    "sft_crnn": {
        "config": "config/main_config.yaml via hydra, experiment_optTransportLoss",
        "n_frames": 1200,
        "dropout_rate": 0.0,
        "training_mode": "test",
        "batch_inference": 64,
        "onset_offset_threshold": 0.5,
        "sample_rate": 48000,
        "hop_length": 1200,
        "features": "cqt",
        "amplitude_to_db": True,
        "duration_clip_s": [0.05, 10.0],
        "precision": "float32; upstream inference.py autocasts float16 on CUDA only",
        "preprocessing": "librosa.load + nnAudio CQT + AmplitudeToDB, as inference.py",
        "decoding": "data.notes_parser.pianoroll_to_notes_list",
    },
    "hrt": {
        "model_type": "Note_pedal",
        "segment_samples": 160000,
        "forward_batch_size": 1,
        "onset_threshold": 0.3,
        "offset_threshold": 0.3,
        "frame_threshold": 0.1,
        "pedal_offset_threshold": 0.2,
        "sample_rate": 16000,
        "preprocessing": "piano_transcription_inference.utilities.load_audio",
        "decoding": "RegressionPostProcessor.output_dict_to_midi_events",
    },
    "onsets_frames": {
        "checkpoint": "maestro_checkpoint.zip",
        "entry": "onsets_frames_transcription_transcribe",
    },
}

23–26 November 2026

# Objective

Compare pretrained models for automatic transcription of solo piano on a fixed set of MAESTRO v3 recordings and on our own recordings with an aligned MIDI reference. Evaluate transcription accuracy, inference speed, and memory usage to select models for the next stage of PianoForge development.

# Candidates

| Model | Role in the study |
| - | - |
| hFT-Transformer (Sony) | Compact candidate with frame, onset, offset, and velocity outputs. |
| Transkun V2 | Candidate that scores full note intervals. |
| SFT-CRNN | CQT-based model using an optimal transport loss checkpoint. It predicts onsets and offsets and has no learned velocity head. |
| HRT (Kong et al.) | High-resolution transcription of notes and sustain pedal events. |
| Onsets and Frames (Magenta) | Historical baseline included in the benchmark configuration. |
| YourMT3 | Reviewed during the architecture survey. Deferred for solo piano because its token decoder adds complexity to extracting confidence scores for the verifier. |

The notebook lists the models selected for the initial run. [Benchmark notebook](../audio-transcription/notebooks/01_amt_benchmark.ipynb)

# Methodology

- **MAESTRO v3.0.0.** Ten complete recordings from the test split, one per composition, totaling approximately 66.3 minutes. The same subset, selected with seed `20261123`, was used for all models.

- **Psycho.** Two 75.746-second recordings of the same waltz performance, with the sustain pedal held for roughly one bar at a time. An iPad on the music stand recorded mono audio. An iPhone on a table approximately two meters away recorded stereo audio. Both tracks were recorded at 48 kHz and aligned to the same MIDI reference containing 878 notes.

- **Hardware and software.** Apple Silicon, macOS, PyTorch with the MPS backend, and float32 inference. The results record Python 3.12.14, PyTorch 2.15.0, mir_eval 0.8.2, and pretty_midi 0.2.11.post0. The run artifacts do not specify the exact chip model or installed RAM.

- **Metrics.** Onset matching requires pitch agreement within 50 cents and onset timing within 50 ms. Onset+offset matching also requires the predicted note end to be within the greater of 50 ms and 20% of the reference note duration. Reference offsets come from MIDI note-off events. The MIDI retains CC64 events, which the scorer excludes from reference note duration calculations. Precision, recall, and F1 are averaged across successfully processed files.

- **Speed and memory.** After warm-up, latency measurements cover audio loading, preprocessing, inference, and decoding. Model loading and metric computation are outside the timed interval. The real-time factor (RTF) is total latency divided by the total duration of successfully processed audio. The MAESTRO run records peak process RSS and, where available, the maximum sampled MPS allocation. The Psycho run records MPS allocations after each file finishes. These samples do not capture peak memory use.

| Model | Checkpoint | Revision | Weights (MB) |
| - | - | - | -: |
| hFT-Transformer | MAESTRO V3 `model_016_003.pkl`, second output in combination mode | `226992b` | 22.14 |
| Transkun V2 | `transkun_v2.pt` and `model.conf` from the release archive | `c5cb837` | 47.83 |
| SFT-CRNN | `SFT-CRNN_Piano-MAESTRO_OTLoss_steps.340609.ckpt` | `21e6964` | 61.98 |
| HRT | `CRNN_note_F1=0.9677_pedal_F1=0.9186.pth`, `Note_pedal` mode | `5660579` | 171.97 |
| Onsets and Frames | Planned Magenta checkpoint `maestro_checkpoint.zip` | — | — |

The results record full revision hashes, file hashes, and decoding settings. [MAESTRO results JSON](https://github.com/ng0q/PianoForge/blob/3aa274cd58d2b82cfb5f3842e63ea77e7015ba3e/research/audio-transcription/results/maestro_v3_test10.json) · [Metric implementation](https://github.com/ng0q/PianoForge/blob/3aa274cd58d2b82cfb5f3842e63ea77e7015ba3e/research/audio-transcription/amt_benchmark/metrics.py)

# MAESTRO Results

| Model | Recordings | Onset P | Onset R | Onset F1 | Onset+offset P | Onset+offset R | Onset+offset F1 |
| - | -: | -: | -: | -: | -: | -: | -: |
| Transkun V2 | 10/10 | 0.994 | 0.953 | 0.973 | 0.298 | 0.286 | 0.292 |
| hFT-Transformer | 10/10 | 0.994 | 0.927 | 0.959 | 0.291 | 0.273 | 0.282 |
| HRT | 10/10 | 0.970 | 0.930 | 0.950 | 0.245 | 0.236 | 0.241 |
| SFT-CRNN | 4/10 | 0.991 | 0.961 | 0.976 | 0.447 | 0.434 | 0.440 |
| Onsets and Frames | 0/10 | — | — | — | — | — | — |

| Model | Mean latency (s) | Median latency (s) | RTF | Peak RSS (GiB) | Peak MPS allocation (GiB) | Parameter count (M) |
| - | -: | -: | -: | -: | -: | -: |
| Transkun V2 | 38.8 | 30.1 | 0.098 | 1.57 | 1.06 | 12.86 |
| hFT-Transformer | 52.1 | 39.8 | 0.131 | 3.33 | 0.58 | 5.52 |
| HRT | 173.6 | 136.0 | 0.436 | 2.67 | 0.29 | 42.95 |
| SFT-CRNN | 198.1 | 169.8 | 0.800 | 1.76 | — | 13.77 |
| Onsets and Frames | — | — | — | — | — | — |

**Comparison on four shared recordings.** F1 was recalculated for the other models on the four files that SFT-CRNN completed. Transkun V2 and SFT-CRNN were within 0.003 of each other on both F1 metrics, with RTFs of 0.112 and 0.800, respectively.

| Model | Onset F1 | Onset+offset F1 | RTF |
| - | -: | -: | -: |
| Transkun V2 | 0.9762 | 0.4430 | 0.112 |
| hFT-Transformer | 0.9662 | 0.4258 | 0.130 |
| HRT | 0.9591 | 0.3740 | 0.433 |
| SFT-CRNN | 0.9755 | 0.4405 | 0.800 |

The shared subset contains recordings of works by Debussy, Schubert/Liszt, Beethoven, and Schumann. [Per-recording results](https://github.com/ng0q/PianoForge/blob/3aa274cd58d2b82cfb5f3842e63ea77e7015ba3e/research/audio-transcription/results/maestro_v3_test10.json)

# Real-world Results

Both tables use the same 878 reference notes. [Psycho results JSON](../audio-transcription/results/psycho_audio_001.json)

## Psycho — iPad on the music stand

| Model | Predicted notes | Onset P | Onset R | Onset F1 | Onset+offset F1 | Latency (s) | RTF |
| - | -: | -: | -: | -: | -: | -: | -: |
| Transkun V2 | 819 | 0.835 | 0.779 | 0.806 | 0.007 | 6.53 | 0.086 |
| hFT-Transformer | 589 | 0.900 | 0.604 | 0.723 | 0.046 | 7.79 | 0.103 |
| HRT | 984 | 0.740 | 0.829 | 0.782 | 0.027 | 33.33 | 0.440 |
| SFT-CRNN | 934 | 0.833 | 0.886 | 0.859 | 0.055 | 32.39 | 0.428 |

## Psycho — iPhone on a table

| Model | Predicted notes | Onset P | Onset R | Onset F1 | Onset+offset F1 | Latency (s) | RTF |
| - | -: | -: | -: | -: | -: | -: | -: |
| Transkun V2 | 1,015 | 0.817 | 0.944 | 0.876 | 0.004 | 6.62 | 0.087 |
| hFT-Transformer | 778 | 0.964 | 0.854 | 0.906 | 0.023 | 7.73 | 0.102 |
| HRT | 963 | 0.825 | 0.904 | 0.863 | 0.008 | 33.37 | 0.441 |
| SFT-CRNN | 887 | 0.922 | 0.932 | 0.927 | 0.016 | 31.74 | 0.419 |

# Observations

- **Accuracy and speed.** Transkun V2 led on both F1 metrics among models that completed all ten MAESTRO recordings. SFT-CRNN achieved the highest onset F1 on both Psycho tracks. Its median latency was 32.07 s, compared with 7.76 s for hFT and 6.58 s for Transkun.

- **Recording conditions.** Every model achieved a higher onset F1 on the iPhone track. hFT recall rose from 0.604 to 0.854. On that track, Transkun matched 94.4% of reference onsets, and hFT achieved 96.4% precision compared with Transkun's 81.7%. Device, microphone placement, and channel count differed between the two recordings.

- **Sustain.** Psycho onset+offset F1 ranged from 0.004 to 0.055. For 876 of the 878 MIDI notes, the key was released while the sustain pedal was down. For one C♯5, the onset was at 4.7805 s, key release at 4.940 s, and pedal release at 5.805 s. A prediction that ends at pedal release has an offset error of 865 ms against a tolerance of 50 ms. The discussion identified the effect of the offset definition and residual audio–MIDI misalignment as priorities for validation. The gap between the two F1 scores on MAESTRO also warrants investigation.

- **Early hFT trials on home recordings.** hFT missed low bass notes and performed worse on loud, dense two-handed chords. It handled fast arpeggios better. These observations identify passages for targeted diagnostic tests.

# Limitations

- The MAESTRO subset covers ten compositions. SFT-CRNN completed four files. The remaining runs stalled or ran out of memory on MPS. The 8.94 GiB value comes from a failed run, and a comparable peak allocation was not measured for SFT-CRNN. Magenta could not run under Python 3.12, and the Onsets and Frames checkpoint was unavailable locally.

- Psycho is a single performance with near-continuous sustain. The two tracks support a paired comparison of recording conditions for that take. Drawing conclusions about recording device, distance, or instrument requires more recordings. Residual audio–MIDI offset and drift were not measured.

- The experiment evaluates pretrained inference in a single environment. Fine-tuning cost, sustain pedal and velocity accuracy, quantization, and the readability of the resulting score have yet to be evaluated.

# Decision

| Model | Preliminary role | Rationale |
| - | - | - |
| Transkun V2 | Primary candidate | Highest F1 scores among models that completed all ten MAESTRO recordings, lowest latency on both datasets, and high recall on the iPhone track. |
| hFT-Transformer | Primary candidate | Most compact model, fast inference on both datasets, and high onset precision on the iPhone track. Its separate outputs are suitable for prototyping the confidence adapter. |
| SFT-CRNN | Comparison baseline | Highest onset F1 on both Psycho tracks and scores close to Transkun on the four shared MAESTRO recordings. Higher latency and incomplete MPS coverage lower its integration priority. |
| HRT | Drop from the active shortlist | Larger checkpoint and slower inference than Transkun, with lower scores on both MAESTRO F1 metrics and lower onset F1 on both Psycho tracks. |
| Onsets and Frames | Defer | The benchmark produced no comparable results. |

In the 25 November discussion, Ilya proposed keeping hFT and Transkun and assessing the cost of fine-tuning them. The choice of a single model for integration remains open.

# Next Steps

1. **Check offset scoring and synchronization.** Re-score the existing predictions against MIDI note-off times and note durations extended using CC64. Account for repeated notes of the same pitch when applying CC64 to approximate sounding duration. Record the same passage without the pedal, verify audio–MIDI alignment at both ends of the take, and inspect onset and offset error distributions.

2. **Expand the home-recording benchmark.** Add examples of low bass, dense chords, fast arpeggios, and repeated notes. Keep a fixed MAESTRO holdout when adapting to home recordings to track changes in performance on clean audio.

3. **Estimate the cost of fine-tuning hFT and Transkun.** Measure step time, throughput, feasible batch size, and peak memory usage on the target hardware.

4. **Evaluate a shared verifier.** Build a model-specific adapter that links decoded notes to the model's original confidence scores. Compare corrections on the same set of errors. Evaluate rhythmic quantization and MusicXML in the downstream processing stage.

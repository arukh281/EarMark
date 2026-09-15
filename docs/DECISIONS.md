# Decisions

The author writes this file. Each entry records one design decision: what was chosen, what it
was chosen over, the evidence, and what would change it. Keep entries short and date them.

```
## YYYY-MM-DD: <decision>
Chose: ...   Over: ...
Because: ... (evidence: test, run ID in results/runs.jsonl, listening note)
Revisit if: ...
```

## Awaiting the author's decision

In week 1, agents wrote the code and made the choices below. Each one is implemented and
documented where it lives, but none is a decision until the author has reviewed it and
written an entry above.

- **Contract additions:** the ERB band widths (DeepFilterNet rule, at least 2 bins per band),
  the periodic sqrt-Hann window, backward FFT scaling, no centring, the barge-in event
  constants (200 ms active after 300 ms silent) and `CONTRACT_HASH`. See `docs/CONTRACT.md`.
- **TSOS** is computed in a level-independent form (a frame counts as over-suppressed below
  about -11 dB) instead of the paper's equation 3. H2 depends on this, so it must be settled
  before the PREREG commit. See `earmark.eval.metrics`.
- **Barge-in scoring:** a detection counts if its decision frame is within 1 s of the
  reference onset, and the dev threshold is matched to 95% *frame-level* VAD recall.
  See `earmark.eval.bargein`.
- **Dev acceptance** requires the speaker-cluster 95% CI lower bound of SI-SDRi to be above
  0 dB, which is stricter than a mean above 0. `clean_control` is left out of the SI-SDRi
  mean and reported as do-no-harm SI-SDR instead.
- **Loss weights** (MR-STFT 0.25, spectral 4, asymmetric 8, SI-SDR 0.03, absent 0.03, VAD 0.1)
  were calibrated on synthetic batches. Recheck them on the mini-full run.
- **Model details beyond the plan:** a residual deep filter (tap 0 = 1 + tanh), the deep
  filter applied after the ERB gains, L2-normalised embeddings and rank-32 FiLM.
- **S-SSM** has 260k parameters, 13% fewer than S-GRU, because it is matched on MACs
  (31.1 vs 30.5 MMAC/s), not on parameters.
- **M or M-256:** undecided until the Kaggle smoke run reports steps/s on a T4.
- **Enrolment front end:** the training embeddings use WeSpeaker's Kaldi fbank with a
  *Hamming* window, not the plan's "povey" window. The browser graph must match training, or
  the embeddings must be recomputed.
- **Engine:** `-ffp-contract=off` everywhere (native matches WASM), a Kaiser resampler
  (120 dB stop band, 7 kHz passband at 16 kHz), an output FIFO primed so that it never
  underruns, and `em_latency_seconds()` for exact fractional latency.
- **Smoke run** checkpoints every 10 minutes instead of 20, so that pruning to the last 3
  checkpoints actually runs within its one hour.

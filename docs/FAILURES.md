# Failures

Two logs. **Model failures** are things Earmark gets wrong, found by listening or by the
suites, each with an example and what was done about it. **Bugs hunted** are defects in the
code or tooling, each with its symptom, cause and fix. The author writes both. Keep each
entry to a few lines.

## Model failures

None logged yet. Nothing has been trained. The week-2 listening pass on M-v1 fills this
section, and M-v2's mixer re-weighting points back to these entries.

```
## <date>: <what goes wrong> (<condition>, e.g. agent-voice interferer at SIR 0 dB)
Example: <mixture id / gallery clip>   Heard: ...   Measured: ...
Cause (suspected or confirmed): ...    Action: ...
```

## Bugs hunted

The entries below were found by agents during the week-1 build and fixed before the week-1
commit. The author's own hunts go above them.

- **`.gitignore` hid two source packages.** The `data/` rule matched `python/earmark/data/` and
  `tests/data/`. Fixed with negation rules.
- **0-d tensors in the weight blob became shape (1,).** `np.ascontiguousarray` promotes 0-d
  arrays. The writer now uses `np.array(order="C")`, and the `weights_small` golden was
  regenerated.
- **The mixer crashed on Apple GPUs.** It moved float64 arrays to MPS, which has no float64.
  It now casts on the host, and a slow test checks that an MPS batch equals the CPU batch.
- **Checkpoints failed to load with `weights_only=True`.** `torch.__version__` is a TorchVersion
  object, not a string. It is now stored as a string.
- **The leak check missed held-out sources.** `HeldOut` did not hold out the VB test DEMAND
  environments or the test Kokoro voices by default. Both are held out now, and the mixer
  keeps only `split == "train"` music and agent rows.
- **`SpeakerCap` dropped the utterance that completes an enrolment pool.** It now fills each
  pool to its budget.
- **The stub speaker encoder was not gain-invariant.** It now uses a floor relative to the
  clip's mean power.
- **The malloc-counter test could pass vacuously.** At -O1 and above, clang deletes unused
  malloc/new pairs. The positive control now stores pointers in a volatile sink.
- **ASan hangs before `main` on macOS 26 with Apple clang 17.** This is inside the runtime
  (`AsanInitFromRtl` spinning), not in Earmark. ASan therefore runs in Linux CI only.
- **Old VoiceBank+DEMAND download URLs return HTML.** The `/bitstream/handle` URLs now serve a
  page instead of the file. The fetch script uses the DataShare REST content URLs with
  SHA-256 pins, so a changed file fails loudly.

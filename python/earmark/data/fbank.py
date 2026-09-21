"""Kaldi log-mel filterbank, the WeSpeaker front end, without torchaudio.

``torchaudio.compliance.kaldi.fbank`` computes the features the frozen speaker encoder
expects, but torchaudio ships no wheel for every torch release (there is none for the
torch the demo runs on), so this module mirrors that function operation for operation
for the settings in :data:`earmark.data.embeddings.KALDI_FBANK`:

* frames of ``frame_length`` ms every ``frame_shift`` ms, whole frames only (Kaldi's
  ``snip_edges=True``);
* per-frame DC removal, pre-emphasis 0.97 against a replicated first sample, then the
  window (Hamming, ``periodic=False``);
* zero-padding to the next power of two, ``|rfft|^2``;
* triangular mel bins between ``low_freq`` and the Nyquist rate, built in float32, with
  a zero column appended for the Nyquist bin;
* ``log(max(energy, float32 eps))``.

Unsupported options (dither, energy, other windows) raise, so a caller cannot silently
get features the encoder was not trained on. ``tests/data/test_fbank.py`` checks the
output against a golden written by torchaudio 2.11 (``scripts/make_fbank_golden.py``).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["kaldi_fbank", "mel_banks"]

#: torchaudio's floor for the log: ``torch.finfo(torch.float).eps``.
_EPSILON = float(torch.finfo(torch.float).eps)


def _mel_scale(freq: Tensor | float) -> Tensor | float:
    if isinstance(freq, Tensor):
        return 1127.0 * (1.0 + freq / 700.0).log()
    return 1127.0 * math.log(1.0 + freq / 700.0)


def mel_banks(num_bins: int, padded_window_size: int, sample_frequency: float, low_freq: float, high_freq: float) -> Tensor:
    """Triangular mel filters ``[num_bins, padded_window_size // 2]`` (float32, no VTLN).

    ``high_freq`` <= 0 means "that far below the Nyquist rate", as in Kaldi.
    """
    if num_bins <= 3:
        raise ValueError(f"need more than 3 mel bins, got {num_bins}")
    if padded_window_size % 2:
        raise ValueError(f"padded window size {padded_window_size} must be even")
    num_fft_bins = padded_window_size / 2
    nyquist = 0.5 * sample_frequency
    if high_freq <= 0.0:
        high_freq += nyquist
    if not (0.0 <= low_freq < nyquist and 0.0 < high_freq <= nyquist and low_freq < high_freq):
        raise ValueError(f"bad mel range: low {low_freq}, high {high_freq}, nyquist {nyquist}")
    fft_bin_width = sample_frequency / padded_window_size
    mel_low = _mel_scale(low_freq)
    mel_high = _mel_scale(high_freq)
    mel_delta = (mel_high - mel_low) / (num_bins + 1)

    bin_index = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low + bin_index * mel_delta
    center_mel = mel_low + (bin_index + 1.0) * mel_delta
    right_mel = mel_low + (bin_index + 2.0) * mel_delta
    mel = _mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    return torch.max(torch.zeros(1), torch.min(up_slope, down_slope))


def kaldi_fbank(
    waveform: Tensor,
    *,
    num_mel_bins: int = 23,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    dither: float = 0.0,
    window_type: str = "hamming",
    use_energy: bool = False,
    sample_frequency: float = 16000.0,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    preemphasis_coefficient: float = 0.97,
) -> Tensor:
    """Log-mel filterbank of one waveform ``[1, T]`` or ``[T]``: ``[frames, num_mel_bins]``.

    Matches ``torchaudio.compliance.kaldi.fbank`` with the same arguments (and its
    defaults for the options this function fixes: ``snip_edges``, ``raw_energy``,
    ``remove_dc_offset`` and ``round_to_power_of_two`` on, no VTLN, no mean subtraction).
    """
    if dither != 0.0:
        raise ValueError("dither must be 0 for a deterministic front end")
    if window_type != "hamming":
        raise ValueError(f"only the hamming window is implemented, got {window_type!r}")
    if use_energy:
        raise ValueError("the energy term is not implemented")
    x = waveform.detach()
    if x.dim() == 2:
        if x.shape[0] != 1:
            raise ValueError(f"expected one channel, got shape {tuple(x.shape)}")
        x = x[0]
    if x.dim() != 1:
        raise ValueError(f"expected [T] or [1, T], got shape {tuple(x.shape)}")
    x = x.float()

    window_shift = int(sample_frequency * frame_shift * 0.001)
    window_size = int(sample_frequency * frame_length * 0.001)
    padded_window_size = 1 if window_size == 0 else 2 ** (window_size - 1).bit_length()
    if not 2 <= window_size <= x.numel():
        raise ValueError(f"window size {window_size} does not fit in {x.numel()} samples")

    frames = x.unfold(0, window_size, window_shift)  # [m, window_size], whole frames only
    frames = frames - frames.mean(dim=1, keepdim=True)  # remove_dc_offset
    if preemphasis_coefficient != 0.0:
        shifted = torch.nn.functional.pad(frames.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
        frames = frames - preemphasis_coefficient * shifted[:, :-1]
    window = torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46, dtype=x.dtype)
    frames = frames * window.unsqueeze(0)
    if padded_window_size != window_size:
        frames = torch.nn.functional.pad(frames, (0, padded_window_size - window_size))

    spectrum = torch.fft.rfft(frames).abs().pow(2.0)
    banks = mel_banks(num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq).to(x.dtype)
    banks = torch.nn.functional.pad(banks, (0, 1))  # the Nyquist bin gets no energy
    energies = torch.mm(spectrum, banks.T)
    return torch.max(energies, torch.tensor(_EPSILON, dtype=x.dtype)).log()

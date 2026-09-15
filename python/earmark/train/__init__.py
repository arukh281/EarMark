"""Training: losses, run presets, checkpoints, checkpoint storage and the trainer.

Entry points:

* ``python -m earmark.train.train --preset smoke ...``: the command line the Kaggle
  notebook launches (:mod:`earmark.train.train`, which also holds the ``Trainer`` class;
  it is not imported here, so running it with ``-m`` stays warning-free).
* :class:`~earmark.train.losses.EarmarkLoss` and :class:`~earmark.train.losses.LossConfig`.
* :data:`~earmark.train.config.PRESETS`: smoke, mini-full, M-v1, S-GRU, S-SSM and M-v2.
* :func:`~earmark.train.checkpoint.model_from_checkpoint`: a trained network, for export
  and scoring.
* :func:`~earmark.train.storage.open_storage`: the private Hub repo or a local folder.
"""

from earmark.train.checkpoint import load_checkpoint, model_from_checkpoint
from earmark.train.config import PRESETS, TrainConfig, preset
from earmark.train.losses import TERMS, EarmarkLoss, LossConfig, LossOutput
from earmark.train.storage import HubStorage, LocalDirStorage, open_storage

__all__ = [
    "PRESETS",
    "TERMS",
    "EarmarkLoss",
    "HubStorage",
    "LocalDirStorage",
    "LossConfig",
    "LossOutput",
    "TrainConfig",
    "load_checkpoint",
    "model_from_checkpoint",
    "open_storage",
    "preset",
]

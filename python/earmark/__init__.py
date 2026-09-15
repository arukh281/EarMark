"""Earmark: personal voice isolation and personal-VAD barge-in gating for voice agents.

Subpackages:

* :mod:`earmark.data` - dataset preparation, the training mixer and benchmark builders.
* :mod:`earmark.model` - WOLA/ERB DSP, recurrent bodies, the Earmark network and streaming.
* :mod:`earmark.train` - losses and the trainer.
* :mod:`earmark.export` - weight blobs and golden tensors for the C++ engine.
* :mod:`earmark.eval` - metrics, baselines and evaluation suites.

Signal-level constants live in :mod:`earmark.constants`, generated from
``contract/signal.yaml`` by ``contract/codegen.py``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

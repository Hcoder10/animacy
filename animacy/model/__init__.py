"""The speech-driven motion model (``docs/MODEL.md``).

``data``      clips -> aligned (features, motion, speaking) tensors, windows, splits
``vq``        motion tokenizer (VQ-VAE, EMA + dead-code revival)
``a2m``       audio -> code logits (Transformer encoder, causal flag) + bigram prior
``infer``     sampling, VQ decode, smoothing -> a ``HumanClip``
``retrieval`` motion-matching baseline + browser index export
``metrics``   the held-out evaluation from ``docs/MODEL.md``
``train``     ``python -m animacy.model.train``
``export``    ONNX + ``web/models/`` bundle
"""

from .data import MODEL_CHANNELS, N_MODEL  # noqa: F401

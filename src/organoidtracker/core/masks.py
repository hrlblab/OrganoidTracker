"""Compact binary masks for tracking results.

SAM2 returns one float32 logit map per object per frame at video resolution. At the
4096x4096 resolution of the lab videos that is 64 MB per mask, and the application used to
keep every mask on the GPU, so memory grew as objects x frames x 64 MB. ``PackedMask``
stores the binarized mask (logit > 0, i.e. probability > 0.5) as packed bits on the CPU,
2 MB per 4096x4096 mask, and exposes the small tensor-like surface that the output,
analysis and viewer code already use: ``mask.cpu().numpy()``, ``shape``, ``ndim``,
``squeeze()`` and comparisons such as ``mask > 0.5``.
"""

from __future__ import annotations

import numpy as np


class PackedMask:
    """A 2-D boolean mask stored as packed bits."""

    __slots__ = ("_bits", "shape", "area")

    def __init__(self, mask) -> None:
        arr = np.asarray(mask)
        if arr.ndim > 2:
            arr = arr.squeeze()
        if arr.ndim != 2:
            raise ValueError(f"mask must be 2-D after squeezing, got shape {arr.shape}")
        mask_bool = arr.astype(bool, copy=False)
        self.shape = (int(mask_bool.shape[0]), int(mask_bool.shape[1]))
        self.area = int(np.count_nonzero(mask_bool))
        self._bits = np.packbits(mask_bool.ravel())

    @classmethod
    def from_logits(cls, logits, threshold: float = 0.0) -> PackedMask:
        """Binarize SAM2 mask logits (tensor or array). Logit > 0 means probability > 0.5."""
        if hasattr(logits, "detach"):
            arr = logits.detach().to("cpu").numpy()
        else:
            arr = np.asarray(logits)
        return cls(arr > threshold)

    @classmethod
    def from_packed_bits(cls, bits, shape) -> PackedMask:
        """Rebuild a mask from its stored form: the row-major packed bits of :attr:`packed_bits` and the shape."""
        height, width = int(shape[0]), int(shape[1])
        if height < 1 or width < 1:
            raise ValueError(f"mask shape must be positive, got {height}x{width}")
        arr = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8).ravel())
        expected = (height * width + 7) // 8
        if arr.size != expected:
            raise ValueError(f"packed bits hold {arr.size} bytes, a {height}x{width} mask needs {expected}")
        packed = cls.__new__(cls)
        packed.shape = (height, width)
        packed.area = int(np.count_nonzero(np.unpackbits(arr, count=height * width)))
        packed._bits = arr
        return packed

    @property
    def packed_bits(self) -> np.ndarray:
        """The stored form: ``numpy.packbits`` of the boolean mask, row-major (uint8, read-only use)."""
        return self._bits

    # --- array access -------------------------------------------------------------------
    def numpy(self) -> np.ndarray:
        """Return the mask as a boolean (H, W) array."""
        height, width = self.shape
        return np.unpackbits(self._bits, count=height * width).reshape(height, width).astype(bool)

    def __array__(self, dtype=None, copy=None):
        arr = self.numpy()
        return arr if dtype is None else arr.astype(dtype)

    # --- tensor-like surface used by legacy consumers ----------------------------------
    def cpu(self) -> PackedMask:
        return self

    def detach(self) -> PackedMask:
        return self

    def squeeze(self, *args, **kwargs) -> PackedMask:
        return self

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self):
        return np.dtype(bool)

    @property
    def nbytes(self) -> int:
        return int(self._bits.nbytes)

    def sum(self) -> int:
        return self.area

    def any(self) -> bool:
        return self.area > 0

    def __gt__(self, other):
        return self.numpy() > other

    def __ge__(self, other):
        return self.numpy() >= other

    def __lt__(self, other):
        return self.numpy() < other

    def __le__(self, other):
        return self.numpy() <= other

    def __repr__(self) -> str:
        return f"PackedMask(shape={self.shape}, area={self.area})"

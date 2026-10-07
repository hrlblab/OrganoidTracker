import numpy as np
import pytest

from organoidtracker.core.masks import PackedMask


def test_round_trip_and_area():
    mask = np.zeros((37, 53), bool)
    mask[5:20, 10:40] = True
    packed = PackedMask(mask)
    assert packed.shape == (37, 53)
    assert packed.area == 15 * 30
    assert np.array_equal(packed.numpy(), mask)
    assert packed.nbytes < mask.size  # packed bits, not one byte per pixel


def test_from_logits_threshold_is_zero():
    logits = np.array([[-3.0, -0.1, 0.0], [0.1, 2.0, 5.0]], np.float32)
    packed = PackedMask.from_logits(logits)
    assert np.array_equal(packed.numpy(), logits > 0)


def test_from_torch_logits_with_leading_dims():
    torch = pytest.importorskip("torch")
    logits = torch.full((1, 1, 4, 6), -2.0)
    logits[0, 0, 1:3, 2:5] = 3.0
    packed = PackedMask.from_logits(logits)
    assert packed.shape == (4, 6)
    assert packed.area == 6


def test_legacy_consumer_surface():
    mask = np.eye(8, dtype=bool)
    packed = PackedMask(mask)
    # The output, analysis and viewer code call .cpu().numpy(), check ndim, squeeze, and compare > 0.5
    arr = packed.cpu().numpy()
    assert arr.dtype == bool and arr.ndim == 2
    assert packed.ndim == 2 and packed.squeeze() is packed
    assert np.array_equal(packed > 0.5, mask)
    assert np.array_equal(np.asarray(packed), mask)
    assert np.asarray(packed, dtype=np.uint8).sum() == 8
    assert packed.sum() == 8 and packed.any()


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        PackedMask(np.zeros((2, 3, 4), bool))

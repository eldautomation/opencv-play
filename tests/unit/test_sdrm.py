import numpy as np
import pytest

from autocollimator.target_utils import sdrm_2

Q_LIMIT = 0.5


def _profile(shift: float, n: int = 201, sigma: float = 6.0) -> np.ndarray:
    """Gaussian line profile on a background, centered `shift` px right of the array center."""
    x = np.arange(n, dtype=np.float64)
    return 20.0 + 200.0 * np.exp(-0.5 * ((x - (n - 1) / 2 - shift) / sigma) ** 2)


def _ssd(values: np.ndarray, search_size: int, k: int) -> float:
    """Reference SDRM sum of squared differences at offset k (independent of sdrm_2)."""
    i = k + search_size
    j = 2 * search_size - i
    base = float(np.mean(values))
    padded = np.concatenate([np.full(j, base), values, np.full(i, base)])
    flipped = np.concatenate([np.full(i, base), values[::-1], np.full(j, base)])
    d = padded - flipped
    return float(np.sum(d * d))


@pytest.mark.unit
@pytest.mark.parametrize("shift", [-0.4, 0.0, 0.3, 0.6])
def test_sdrm_2_default_returns_integer_offset(shift):
    k, _r1 = sdrm_2(_profile(shift), 10, q_limit=Q_LIMIT, debug=False)
    assert type(k) is int
    assert k == round(shift)


@pytest.mark.unit
@pytest.mark.parametrize("neighbors", [1, 2, 3])
@pytest.mark.parametrize("shift", [-0.4, -0.25, 0.1, 0.3, 0.45])
def test_sdrm_2_subpixel_recovers_shift(shift, neighbors):
    k, _r1 = sdrm_2(
        _profile(shift), 10, q_limit=Q_LIMIT, debug=False, subpixel=True, subpixel_neighbors=neighbors
    )
    assert isinstance(k, float)
    assert abs(k - shift) < 0.05, f"shift={shift} neighbors={neighbors} measured={k}"


@pytest.mark.unit
@pytest.mark.parametrize("shift", [-0.4, 0.1, 0.3, 0.45, 0.8])
def test_sdrm_2_one_neighbor_matches_three_point_parabola(shift):
    values = _profile(shift)
    search_size = 10
    k_int, _ = sdrm_2(values, search_size, q_limit=Q_LIMIT, debug=False)
    a, b, c = (_ssd(values, search_size, k) for k in (k_int - 1, k_int, k_int + 1))
    expected = k_int + 0.5 * (a - c) / (a - 2 * b + c)

    k, _ = sdrm_2(values, search_size, q_limit=Q_LIMIT, debug=False, subpixel=True, subpixel_neighbors=1)
    assert k == pytest.approx(expected, abs=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("shift", [-0.4, 0.3])
def test_sdrm_2_quality_ratio_same_in_both_modes(shift):
    values = _profile(shift)
    _, r1_off = sdrm_2(values, 10, q_limit=Q_LIMIT, debug=False)
    _, r1_on = sdrm_2(values, 10, q_limit=Q_LIMIT, debug=False, subpixel=True)
    assert r1_on == r1_off


@pytest.mark.unit
@pytest.mark.parametrize("neighbors", [0, 4, -1, 2.5, True, "2", None])
def test_sdrm_2_invalid_neighbors_raise(neighbors):
    with pytest.raises(ValueError, match="subpixel_neighbors must be an int in 1..3"):
        sdrm_2(_profile(0.0), 10, q_limit=Q_LIMIT, debug=False, subpixel=True, subpixel_neighbors=neighbors)


@pytest.mark.unit
def test_sdrm_2_minimum_at_search_edge_falls_back_to_integer():
    values = _profile(3.3)
    search_size = 3  # true center lies beyond the search range, so the minimum is at k=+3
    k_int, _ = sdrm_2(values, search_size, q_limit=Q_LIMIT, debug=False)
    assert k_int == search_size

    k, _ = sdrm_2(values, search_size, q_limit=Q_LIMIT, debug=False, subpixel=True)
    assert k == k_int

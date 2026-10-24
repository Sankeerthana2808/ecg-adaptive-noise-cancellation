import numpy as np
import pytest

from ecganc import build_scenario, lms, nlms, rls, snr_improvement_db


@pytest.fixture(scope="module")
def sc():
    return build_scenario(duration_s=12, seed=1)


@pytest.mark.parametrize("fn,kw", [(lms, dict(mu=0.02)), (nlms, dict(mu=0.05)), (rls, dict(lam=0.999))])
def test_filter_improves_snr(sc, fn, kw):
    _, e, _ = fn(sc.primary, sc.reference, order=4, **kw)
    assert snr_improvement_db(sc.clean, sc.primary, e, skip=int(4 * sc.fs)) > 10.0


def test_system_identification_exact():
    """With noiseless data the filters must recover a known FIR system."""
    rng = np.random.default_rng(0)
    h = np.array([0.8, -0.4, 0.2, 0.1])
    x = rng.standard_normal(4000)
    d = np.convolve(x, h)[: len(x)]
    for fn, kw in [(nlms, dict(mu=0.9)), (rls, dict(lam=1.0))]:
        _, _, w = fn(d, x, order=4, **kw)
        np.testing.assert_allclose(w, h, atol=1e-2)


def test_rls_converges_faster_than_lms(sc):
    from ecganc import convergence_sample
    _, e_l, _ = lms(sc.primary, sc.reference, order=4, mu=0.005)
    _, e_r, _ = rls(sc.primary, sc.reference, order=4, lam=0.999)
    assert convergence_sample(e_r, sc.clean) < convergence_sample(e_l, sc.clean)


def test_error_is_d_minus_y(sc):
    y, e, _ = nlms(sc.primary, sc.reference, order=4)
    np.testing.assert_allclose(e, sc.primary - y)

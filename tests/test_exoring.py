#!/usr/bin/env python3

import pytest
import numpy as np
from math import pi
from exoring import ExoRing

# multiple of the default lcsum_block size so the light curve size divides
# evenly into reduction blocks
LC_SIZE = 1024


def _fast_ring(**kwargs):
    """A small, coarse-grained ring for tests that don't need precision."""
    kwargs.setdefault("planet_scale", 50)
    kwargs.setdefault("img_array_shape", (128, 256))
    return ExoRing(**kwargs)


def _precise_ring(**kwargs):
    """A higher-resolution ring for tests sensitive to discretisation error."""
    kwargs.setdefault("planet_scale", 200)
    kwargs.setdefault("img_array_shape", (512, 512))
    return ExoRing(**kwargs)


def _build(ring, inner_ring_radius=1.5, outer_ring_radius=2.0,
           ring_optical_depth=1.0, gamma=0.5):
    ring.build_image(
        inner_ring_radius=inner_ring_radius, outer_ring_radius=outer_ring_radius,
        ring_optical_depth=ring_optical_depth, gamma=gamma
    )
    return ring


# ---------------------------------------------------------------------------
# build_image / read_image / get_k — analytical / property tests
# ---------------------------------------------------------------------------

def test_image_opacity_bounds():
    ring = _build(_fast_ring())
    img = ring.read_image()
    assert np.all(img >= 0.0)
    assert np.all(img <= 1.0 + 1e-6)


def test_image_planet_area_no_ring():
    """tau=0 -> ring is fully transparent, only the planet disk contributes."""
    ring = _build(_precise_ring(), ring_optical_depth=0.0, gamma=0.3)
    img = ring.read_image(return_full=False)
    pixel_area = ring.pixel_size ** 2
    # read_image(return_full=False) returns a single quadrant, i.e. 1/4 of
    # the planet disk
    assert abs(4 * img.sum() * pixel_area - pi) < 0.01


def test_k_unity_without_ring():
    """With tau=0 the effective radius scaling factor k should be ~1."""
    ring = _build(_precise_ring(), ring_optical_depth=0.0, gamma=0.3)
    k = ring.get_k()
    assert abs(k - 1.0) < 0.01


def test_k_increases_with_optical_depth():
    k_values = [
        _build(_fast_ring(), ring_optical_depth=tau).get_k()
        for tau in (0.0, 1.0, 5.0)
    ]
    assert k_values[0] < k_values[1] < k_values[2]


def test_image_edge_on_wider():
    """A more inclined (closer to edge-on) ring projects a larger opacity area."""
    areas = []
    for gamma in (0.3, pi / 2 - 0.05):
        ring = _build(_fast_ring(), gamma=gamma)
        img = ring.read_image(return_full=False)
        areas.append(img.sum())
    assert areas[1] > areas[0]


def test_build_image_ring_too_large_raises():
    ring = _fast_ring()
    with pytest.raises(ValueError):
        ring.build_image(inner_ring_radius=1.5, outer_ring_radius=10.0,
                          ring_optical_depth=1.0, gamma=0.5)


def test_init_planet_too_large_raises():
    with pytest.raises(AssertionError):
        ExoRing(planet_scale=1000, img_array_shape=(128, 256))


# ---------------------------------------------------------------------------
# occult_star / read_lightcurve — analytical / property tests
# ---------------------------------------------------------------------------

def test_light_curve_bounds():
    ring = _build(_fast_ring())
    x = np.linspace(-2.0, 2.0, LC_SIZE)
    y = np.zeros_like(x)
    ring.put_xy_array(x, y)
    ring.occult_star(planet_radius=0.1, obliquity=0.0, ld_params=(0.4, 0.3))
    lc = ring.read_lightcurve()
    assert np.all(lc >= 0.0 - 1e-6)
    assert np.all(lc <= 1.0 + 1e-6)


def test_light_curve_no_transit():
    """Planet entirely off-star (large offset) leaves the light curve at 1."""
    ring = _build(_fast_ring())
    x = np.full(LC_SIZE, 5.0)
    y = np.full(LC_SIZE, 5.0)
    ring.put_xy_array(x, y)
    ring.occult_star(planet_radius=0.1, obliquity=0.0, ld_params=(0.4, 0.3))
    lc = ring.read_lightcurve()
    assert np.allclose(lc, 1.0, atol=1e-6)


def test_light_curve_symmetry():
    """A centred transit at zero obliquity is symmetric in x."""
    ring = _build(_fast_ring())
    x = np.linspace(-2.0, 2.0, LC_SIZE)
    y = np.zeros_like(x)
    ring.put_xy_array(x, y)
    ring.occult_star(planet_radius=0.1, obliquity=0.0, ld_params=(0.4, 0.3))
    lc = ring.read_lightcurve()
    assert np.allclose(lc, lc[::-1], atol=1e-5)


def test_light_curve_baseline_unity():
    """Points well outside the stellar disk are exactly 1."""
    ring = _build(_fast_ring())
    x = np.linspace(-5.0, 5.0, LC_SIZE)
    y = np.zeros_like(x)
    ring.put_xy_array(x, y)
    ring.occult_star(planet_radius=0.1, obliquity=0.0, ld_params=(0.4, 0.3))
    lc = ring.read_lightcurve()
    far = np.abs(x) > 3.5
    assert np.allclose(lc[far], 1.0, atol=1e-6)


def test_occult_star_without_xy_raises():
    ring = _build(_fast_ring())
    with pytest.raises(RuntimeError):
        ring.occult_star(planet_radius=0.1, obliquity=0.0, ld_params=(0.4, 0.3))


def test_read_lightcurve_before_occult_raises():
    ring = _build(_fast_ring())
    with pytest.raises(RuntimeError):
        ring.read_lightcurve()


def test_put_xy_array_shape_mismatch_raises():
    ring = _build(_fast_ring())
    with pytest.raises(AssertionError):
        ring.put_xy_array(np.zeros(10), np.zeros(11))


# ---------------------------------------------------------------------------
# get_loglikelihood — cross-checked against an independent numpy calculation
# ---------------------------------------------------------------------------

def _ring_with_lightcurve():
    ring = _build(_fast_ring())
    x = np.linspace(-2.0, 2.0, LC_SIZE)
    y = np.zeros_like(x)
    ring.put_xy_array(x, y)
    ring.occult_star(planet_radius=0.1, obliquity=0.0, ld_params=(0.4, 0.3))
    return ring


def test_loglikelihood_perfect_fit_is_zero():
    ring = _ring_with_lightcurve()
    lc = ring.read_lightcurve()
    ring.put_observed_lc(lc.astype(np.float32),
                         np.full_like(lc, 1e-4, dtype=np.float32))
    ll = ring.get_loglikelihood()
    assert abs(ll) < 1e-3


def test_loglikelihood_matches_numpy_chisq():
    ring = _ring_with_lightcurve()
    lc = ring.read_lightcurve()

    rng = np.random.default_rng(0)
    flux = (lc + rng.normal(0, 1e-3, size=lc.shape)).astype(np.float32)
    flux_error = np.full_like(flux, 1e-3, dtype=np.float32)

    ring.put_observed_lc(flux, flux_error)
    ll = ring.get_loglikelihood()

    expected_chisq = np.sum(((flux - lc) / flux_error) ** 2)
    expected_ll = -0.5 * expected_chisq
    assert abs(ll - expected_ll) < 1e-2 * abs(expected_ll)


def test_get_loglikelihood_without_lightcurve_raises():
    ring = _build(_fast_ring())
    with pytest.raises(RuntimeError):
        ring.get_loglikelihood()


def test_get_loglikelihood_without_observed_flux_raises():
    ring = _ring_with_lightcurve()
    with pytest.raises(RuntimeError):
        ring.get_loglikelihood()


# ---------------------------------------------------------------------------
# put_times_array / compute_xy_array — orbital (x,y) generation via on-device
# Kepler solver, cross-checked against an independent host-side reference
# ---------------------------------------------------------------------------

def _host_kepler_xy(times, t0, period, a, inc, ecc, w):
    """Pure-numpy/host reference implementation of the get_xy CUDA kernel
    (exoring.cu), used to independently verify the on-device Kepler solver
    rather than just checking self-consistency."""
    times = np.asarray(times, dtype=np.float64)
    n = 2.0 * np.pi / period
    f0 = 0.5 * np.pi - w

    if ecc < 1.0e-5:
        m0 = f0
    else:
        e0 = 2.0 * np.arctan(
            np.sqrt((1.0 - ecc) / (1.0 + ecc)) * np.tan(0.5 * f0)
        )
        m0 = e0 - ecc * np.sin(e0)

    tp = t0 - 0.5 * period * m0 / np.pi

    if ecc < 1.0e-5:
        f = np.mod((times - tp) / period, 1.0) * 2.0 * np.pi
    else:
        m = n * (times - tp)
        e = m.copy()
        for _ in range(100):
            e = e - (e - ecc * np.sin(e) - m) / (1.0 - ecc * np.cos(e))
        f = 2.0 * np.arctan(
            np.sqrt((1.0 + ecc) / (1.0 - ecc)) * np.tan(0.5 * e)
        )

    r = a * (1.0 - ecc ** 2) / (1.0 + ecc * np.cos(f))
    x = -r * np.cos(w + f)
    y = -r * np.sin(w + f) * np.cos(inc)
    return x, y


def test_compute_xy_array_circular_matches_host_reference():
    ring = _fast_ring()
    times = np.linspace(0.0, 30.0, LC_SIZE)
    ring.put_times_array(times)
    orbit_params = dict(t0=5.0, period=10.0, a=15.0, inc=1.4, ecc=0.0, w=0.0)
    ring.compute_xy_array(**orbit_params)

    x = ring.x_array.get()
    y = ring.y_array.get()
    x_exp, y_exp = _host_kepler_xy(times, **orbit_params)

    assert np.allclose(x, x_exp, atol=1e-6)
    assert np.allclose(y, y_exp, atol=1e-6)


def test_compute_xy_array_eccentric_matches_host_reference():
    ring = _fast_ring()
    times = np.linspace(0.0, 30.0, LC_SIZE)
    ring.put_times_array(times)
    orbit_params = dict(t0=5.0, period=10.0, a=15.0, inc=1.4, ecc=0.4, w=0.7)
    ring.compute_xy_array(**orbit_params)

    x = ring.x_array.get()
    y = ring.y_array.get()
    x_exp, y_exp = _host_kepler_xy(times, **orbit_params)

    assert np.allclose(x, x_exp, atol=1e-6)
    assert np.allclose(y, y_exp, atol=1e-6)


def test_compute_xy_array_circular_matches_analytic_formula():
    """For ecc=0, w=0 the Kepler solve reduces to plain uniform circular
    motion - cross-check against that closed-form expression directly,
    independent of the general get_xy algorithm."""
    ring = _fast_ring()
    t0, period, a, inc = 5.0, 10.0, 15.0, 1.0
    times = np.linspace(0.0, 30.0, LC_SIZE)
    ring.put_times_array(times)
    ring.compute_xy_array(t0=t0, period=period, a=a, inc=inc, ecc=0.0, w=0.0)

    x = ring.x_array.get()
    y = ring.y_array.get()

    phase = 2.0 * np.pi * (times - t0) / period
    x_exp = a * np.sin(phase)
    y_exp = -a * np.cos(phase) * np.cos(inc)

    assert np.allclose(x, x_exp, atol=1e-6)
    assert np.allclose(y, y_exp, atol=1e-6)


def test_compute_xy_array_mid_transit_impact_parameter():
    """At inferior conjunction (t=t0) for a circular orbit, the planet should
    sit at the orbit's impact parameter b = a*cos(inc), with x = 0."""
    ring = _fast_ring()
    t0, period, a, inc = 3.0, 10.0, 12.0, 1.2
    times = np.array([t0])
    ring.put_times_array(times)
    ring.compute_xy_array(t0=t0, period=period, a=a, inc=inc, ecc=0.0, w=0.0)

    x = ring.x_array.get()
    y = ring.y_array.get()

    assert abs(x[0]) < 1e-6
    assert abs(y[0] - (-a * np.cos(inc))) < 1e-6


def test_compute_xy_array_periodicity():
    ring = _fast_ring()
    t0, period, a, inc = 2.0, 7.0, 10.0, 1.3
    times = np.array([1.0, 1.0 + period, 1.0 + 3 * period])
    ring.put_times_array(times)
    ring.compute_xy_array(t0=t0, period=period, a=a, inc=inc, ecc=0.2, w=0.5)

    x = ring.x_array.get()
    y = ring.y_array.get()

    assert np.allclose(x, x[0], atol=1e-6)
    assert np.allclose(y, y[0], atol=1e-6)


def test_compute_xy_array_without_times_raises():
    ring = _fast_ring()
    with pytest.raises(RuntimeError):
        ring.compute_xy_array(
            t0=0.0, period=10.0, a=10.0, inc=1.5, ecc=0.0, w=0.0
        )


def test_orbital_position_feeds_transit_light_curve():
    """End-to-end: orbital elements -> on-device (x,y) -> opacity image ->
    light curve, producing a transit dip centred near t0."""
    ring = _build(_fast_ring())
    t0, period, a, inc = 5.0, 20.0, 10.0, 1.55  # near edge-on, low impact param
    times = np.linspace(t0 - 1.0, t0 + 1.0, LC_SIZE)
    ring.put_times_array(times)
    ring.compute_xy_array(t0=t0, period=period, a=a, inc=inc, ecc=0.0, w=0.0)
    ring.occult_star(planet_radius=0.1, obliquity=0.0, ld_params=(0.4, 0.3))
    lc = ring.read_lightcurve()

    assert np.all(lc >= 0.0 - 1e-6)
    assert np.all(lc <= 1.0 + 1e-6)
    # mid-transit point should be dimmer than the baseline at the ends
    mid_idx = LC_SIZE // 2
    assert lc[mid_idx] < lc[0]
    assert lc[mid_idx] < lc[-1]


def test_light_curve_size_not_multiple_of_block_size():
    """Regression test for an off-by-one in the light curve reduction kernel
    that overran by one row whenever the light curve size wasn't an exact
    multiple of lcsum_block."""
    ring = _build(_fast_ring())
    size = LC_SIZE + 17
    x = np.linspace(-2.0, 2.0, size)
    y = np.zeros_like(x)
    ring.put_xy_array(x, y)
    ring.occult_star(planet_radius=0.1, obliquity=0.0, ld_params=(0.4, 0.3))
    lc = ring.read_lightcurve()
    assert np.all(lc >= 0.0 - 1e-6)
    assert np.all(lc <= 1.0 + 1e-6)

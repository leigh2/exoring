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
    kwargs.setdefault("super_sample_factor", 5)
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
    with pytest.raises(AssertionError):
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

#!/usr/bin/env python3

import pytest
import numpy as np
from math import pi
from pathlib import Path
from exoring import build_exoring_image, occult_star

TEST_DATA = Path(__file__).parent

test_threshold = 1E-12


# ---------------------------------------------------------------------------
# build_exoring_image — regression tests
# ---------------------------------------------------------------------------

def test_image_generation():
    oig_test_data = np.load(TEST_DATA / "opacity_image_gen_test_data.npz")
    img, xgrid, ygrid, px_area = build_exoring_image(
        int(oig_test_data['ngrid']),
        *oig_test_data['params'],
        super_sample_factor=int(oig_test_data['super_sample_factor']),
        full_output=oig_test_data['full_output']
    )
    assert (
            all(np.abs(
                img - oig_test_data['op_img']).flatten() < test_threshold) and
            all(np.abs(xgrid - oig_test_data[
                'op_xgrid']).flatten() < test_threshold) and
            all(np.abs(ygrid - oig_test_data[
                'op_ygrid']).flatten() < test_threshold) and
            np.abs(px_area - oig_test_data['op_area']) < test_threshold
    )


def test_image_generation_multi_ring():
    d = np.load(TEST_DATA / "multi_ring_image_test_data.npz")
    img, xgrid, ygrid, area = build_exoring_image(
        int(d['ngrid']),
        d['inner_radii'], d['outer_radii'], d['optical_depths'],
        float(d['gamma']),
        super_sample_factor=int(d['ssf']),
        full_output=bool(d['full_output'])
    )
    assert np.all(np.abs(img - d['ref_img']) < test_threshold)
    assert np.all(np.abs(xgrid - d['ref_xgrid']) < test_threshold)
    assert np.all(np.abs(ygrid - d['ref_ygrid']) < test_threshold)
    assert np.abs(area - d['ref_area']) < test_threshold


# ---------------------------------------------------------------------------
# build_exoring_image — analytical / property tests
# ---------------------------------------------------------------------------

def test_image_generation_sparse_consistent():
    """full_output=False returns exactly the nonzero elements of full_output=True."""
    img_full, x_full, y_full, area_full = build_exoring_image(
        50, 1.5, 2.0, 0.5, 0.3, full_output=True
    )
    img_sparse, x_sparse, y_sparse, area_sparse = build_exoring_image(
        50, 1.5, 2.0, 0.5, 0.3, full_output=False
    )
    nonzero = img_full > 0
    assert np.allclose(img_full[nonzero], img_sparse)
    assert np.allclose(x_full[nonzero], x_sparse)
    assert np.allclose(y_full[nonzero], y_sparse)
    assert area_full == area_sparse


def test_image_generation_inner_outer_swap():
    """Auto-swap: passing inner > outer gives the same image as inner < outer."""
    img1, x1, y1, a1 = build_exoring_image(
        80, 1.5, 2.0, 0.5, 0.3, full_output=True
    )
    img2, x2, y2, a2 = build_exoring_image(
        80, 2.0, 1.5, 0.5, 0.3, full_output=True
    )
    assert np.allclose(img1, img2)


def test_image_generation_overlapping_rings():
    """Overlapping rings must raise RuntimeError."""
    with pytest.raises(RuntimeError):
        # ring 1: [1.0, 2.0], ring 2: [1.5, 2.5] — overlap in [1.5, 2.0]
        build_exoring_image(50, [1.0, 1.5], [2.0, 2.5], [0.5, 0.5], 0.3)


def test_image_generation_opacity_bounds():
    """All pixel opacity values must lie in [0, 1] (within floating point noise)."""
    img, _, _, _ = build_exoring_image(
        80, 1.5, 2.0, 5.0, 0.5, full_output=True
    )
    assert np.all(img >= 0.0)
    assert np.all(img <= 1.0 + 1e-12)


def test_image_generation_planet_area():
    """Summed opacity * pixel area ≈ π for a transparent-ring (tau=0) image."""
    # tau=0 → ring opacity=0, only the planet disk contributes
    img, _, _, area = build_exoring_image(
        200, 1.5, 2.0, 0.0, 0.3, full_output=True
    )
    assert abs(np.sum(img) * area - pi) < 0.01


def test_image_generation_edge_on_wider():
    """Edge-on ring (gamma=π/2) occupies more area than a tilted ring."""
    img_edge, _, _, area_e = build_exoring_image(
        80, 1.5, 2.0, 1.0, pi / 2, full_output=True
    )
    img_tilt, _, _, area_t = build_exoring_image(
        80, 1.5, 2.0, 1.0, 0.3, full_output=True
    )
    assert np.sum(img_edge) * area_e > np.sum(img_tilt) * area_t


# ---------------------------------------------------------------------------
# occult_star — regression tests
# ---------------------------------------------------------------------------

def test_light_curve_generation():
    lc_test_data = np.load(TEST_DATA / 'transit_lc_gen_test_data.npz')
    lc = occult_star(
        lc_test_data['lc_img'],
        lc_test_data['lc_xgrid'], lc_test_data['lc_ygrid'],
        lc_test_data['lc_px_area'], lc_test_data['lc_p_rad'],
        lc_test_data['lc_x_steps'], lc_test_data['lc_y_offset'],
        lc_test_data['lc_obliq'], tuple(lc_test_data['lc_ld_params'])
    )
    diffs = np.abs(lc - lc_test_data['lc_result']).flatten()
    assert all(diffs < test_threshold)


def test_light_curve_multi_ring():
    d = np.load(TEST_DATA / 'multi_ring_lc_test_data.npz')
    img, x, y, area = build_exoring_image(
        int(d['ngrid']),
        d['inner_radii'], d['outer_radii'], d['optical_depths'],
        float(d['gamma']),
        super_sample_factor=int(d['ssf']),
        full_output=False
    )
    lc = occult_star(
        img, x, y, area,
        float(d['p_rad']), d['x_steps'], float(d['y_offset']),
        float(d['obliq']), tuple(d['ld_params'])
    )
    assert np.all(np.abs(lc - d['ref_lc']) < test_threshold)


def test_light_curve_non_zero_obliquity():
    d = np.load(TEST_DATA / 'obliquity_lc_test_data.npz')
    lc = occult_star(
        d['img'], d['xgrid'], d['ygrid'], float(d['area']),
        float(d['p_rad']), d['x_steps'], float(d['y_offset']),
        float(d['obliq']), tuple(d['ld_params'])
    )
    assert np.all(np.abs(lc - d['ref_lc']) < test_threshold)


# ---------------------------------------------------------------------------
# occult_star — analytical / property tests
# ---------------------------------------------------------------------------

def test_light_curve_no_transit():
    """Planet entirely off-star (large y offset) returns all ones."""
    img, x, y, area = build_exoring_image(50, 1.5, 2.0, 0.5, 0.3)
    lc = occult_star(
        img, x, y, area, 0.1,
        np.linspace(-3, 3, 100), 2.0, 0.0, (0.4, 0.3)
    )
    assert np.allclose(lc, 1.0)


def test_light_curve_symmetry():
    """Centred transit with zero obliquity produces a symmetric light curve."""
    img, x, y, area = build_exoring_image(50, 1.5, 2.0, 0.5, pi / 4)
    x_offsets = np.linspace(-2.5, 2.5, 1001)
    lc = occult_star(img, x, y, area, 0.05, x_offsets, 0.0, 0.0, (0.4, 0.3))
    assert np.allclose(lc, lc[::-1], atol=1e-10)


def test_light_curve_bounds():
    """All light curve values must lie in [0, 1]."""
    img, x, y, area = build_exoring_image(50, 1.5, 2.0, 0.5, 0.3)
    lc = occult_star(
        img, x, y, area, 0.1,
        np.linspace(-2, 2, 500), 0.0, 0.0, (0.4, 0.3)
    )
    assert np.all(lc >= 0.0)
    assert np.all(lc <= 1.0 + 1e-12)


def test_light_curve_baseline_unity():
    """Points well outside the stellar disk are exactly 1."""
    img, x, y, area = build_exoring_image(50, 1.5, 2.0, 0.5, 0.3)
    x_offsets = np.linspace(-5, 5, 200)
    lc = occult_star(img, x, y, area, 0.05, x_offsets, 0.0, 0.0, (0.4, 0.3))
    far = np.abs(x_offsets) > 3.5
    assert np.allclose(lc[far], 1.0)

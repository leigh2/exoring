#!/usr/bin/env python3

import numpy as np
from exoring import build_exoring_image, occult_star

"""
Notes
-----
to test:
- full output
- inner_rad > outer_rad
"""

test_threshold = 1E-12


def test_image_generation():
    oig_test_data = np.load("tests/opacity_image_gen_test_data.npz")
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


def test_light_curve_generation():
    lc_test_data = np.load('tests/transit_lc_gen_test_data.npz')
    lc = occult_star(
        lc_test_data['lc_img'],
        lc_test_data['lc_xgrid'], lc_test_data['lc_ygrid'],
        lc_test_data['lc_px_area'], lc_test_data['lc_p_rad'],
        lc_test_data['lc_x_steps'], lc_test_data['lc_y_offset'],
        lc_test_data['lc_obliq'], tuple(lc_test_data['lc_ld_params'])
    )
    diffs = np.abs(lc - lc_test_data['lc_result']).flatten()
    assert all(diffs < test_threshold)

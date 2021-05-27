#!/usr/bin/env python3

import numpy as np
from numba import njit
import numba as nb

__version__ = "0.1"


@njit
def quad_limb_dark(radii, a, b):
    """Intensity as a function of radius for the quadratic limb darkening model.

    Parameters
    ----------
    radii : ndarray
        Radii as a fraction of stellar radius.
    a,b, : float, float
        Quadratic limb darkening coefficients.

    Returns
    -------
    intensity : ndarray
        Stellar intensity at the requested radii.

    Todo
    ----
    Check the provided limb darkening coefficients are valid.
    """

    # initialise intensity map
    intensity = np.zeros(radii.shape, dtype=np.float64)

    # numba requires 1d arrays, ravels are views
    flat_r = radii.ravel()
    flat_i = intensity.ravel()

    # evaluate limb darkening model
    le1 = np.abs(flat_r) <= 1.0
    c_mu = 1.0 - np.cos(np.arcsin(flat_r[le1]))
    flat_i[le1] = (1 - a * c_mu - b * c_mu ** 2) / (1 - a / 3 - b / 6) / np.pi

    return intensity


def build_exoring_image(
        pixel_scale,
        inner_ring_radius,
        outer_ring_radius,
        ring_opacity,
        gamma,
        super_sample_factor=10,
        full_output=False,
):
    """

    Parameters
    ----------
    pixel_scale : int
        The scale of the required opacity image, parametrized as the number of
        pixels per planet radius.
    inner_ring_radius : float
        The inner radius of the ring, in units of planet radii.
    outer_ring_radius : float
        The outer radius of the ring, in units of planet radii.
    ring_opacity : float
        The opacity of the ring, where 0 is fully transparent and 1 is fully
        opaque.
    gamma : float
        Inclination angle relative to the line of sight to the observer in
        radians. An angle of 0.0 radians is hence invisible as ring depth is not
        modelled.
    super_sample_factor : int, optional
        Pixels which straddle the planet and/or ring boundary are super-sampled
        to estimate the appropriate opacity. super_sample_factor is the number
        of subdivisions along each dimension. (Default: 10.)
    full_output : bool, optional
        If True return the complete 2-dimensional opacity array (image). If
        False return only the array elements which have nonzero opacity.
        (Default: False.)

    Returns
    -------
    exoring_image : ndarray
        The opacity image. If full_output is True this is an array with ndim=2,
        otherwise it is flattened and includes only elements with nonzero
        opacity.
    x, y : ndarrays
        The x and y positions of each image element in units of planet radii
        and centred at the planet centre. These are either ndim=2 or flattened
        arrays depending on the state of full_output.
        Note that the occulter can be rescaled, rotated, translated, etc. by
        simply applying the appropriate transformations to these arrays.
    area : float
        The area of each transparency profile element in units of planetary
        radii squared.
    """

    # get the half size of the required grid in each dimension
    ngrid_maj = max(
        np.int_(np.ceil(outer_ring_radius * pixel_scale)), pixel_scale
    )
    ngrid_min = max(
        np.int_(np.ceil(ngrid_maj * np.sin(gamma))), pixel_scale
    )

    # initialise the output arrays
    er_image = np.zeros((2 * ngrid_maj, 2 * ngrid_min), dtype=np.float64).T
    xgrid, ygrid = np.meshgrid(
        (np.arange(-ngrid_maj, ngrid_maj) + 0.5)/pixel_scale,
        (np.arange(-ngrid_min, ngrid_min) + 0.5)/pixel_scale
    )

    er_image = fill_opacity_grid(
        xgrid, ygrid, er_image, gamma,
        inner_ring_radius, outer_ring_radius, ring_opacity,
        ssf=super_sample_factor
    )

    # calculate area of each grid element, units of r_planet^2
    elem_area = (1.0 / pixel_scale) ** 2

    return er_image, xgrid, ygrid, elem_area


@njit
def fill_opacity_grid(xgrid, ygrid, image, gamma, i_r, o_r, op, ssf=10):
    """
    Fill an opacity grid image with planet and ring.

    Parameters
    ----------
    xgrid : ndarray
        x grid coordinates
    ygrid : ndarray
        y grid coordinates
    image : ndarray
        image array
    gamma : float
        inclination
    i_r : float
        ring inner radius
    o_r : float
        ring outer radius
    op : float
        ring opacity
    ssf : int, optional
        super-sample factor (default 10)

    Returns
    -------
    filled opacity grid image
    """
    # require some array shape information
    isize, jsize = image.shape
    ihalf = isize // 2
    jhalf = jsize // 2
    # the pixel size in planetary radii (assuming square pixels)
    pixsize = np.float_(xgrid[0,1] - xgrid[0,0])
    half_pixsize = pixsize * 0.5

    # the minor axis sized of the inner and outer ring ellipses
    i_r_min = np.sin(gamma) * i_r
    o_r_min = np.sin(gamma) * o_r

    # for each pixel
    for i in range(ihalf):
        for j in range(jhalf):
            # get the position of the centre of the pixel
            xpos = xgrid[ihalf+i, jhalf+j]
            ypos = ygrid[ihalf+i, jhalf+j]

            # pixel outer radius
            px_outer_rad = np.sqrt(
                (xpos + half_pixsize)**2 + (ypos + half_pixsize)**2
            )
            # pixel inner radius
            px_inner_rad = np.sqrt(
                (xpos - half_pixsize)**2 + (ypos - half_pixsize)**2
            )

            if px_outer_rad <= 1.0:
                # the radius of the furthest vertex of the pixel is inside the
                # planet radius, hence the pixel is fully inside the planet and
                # the opacity is total
                value = 1.0

            elif (((xpos - half_pixsize) / i_r)**2 +
                  ((ypos - half_pixsize) / i_r_min)**2 >= 1
                  and
                  ((xpos + half_pixsize) / o_r)**2 +
                  ((ypos + half_pixsize) / o_r_min)**2 <= 1):
                # the radius of the closest vertex of the pixel is outside the
                # inner radius of the ring, and the radius of its furthest
                # vertex is inside the outer radius of the ring. Hence, the
                # pixel is fully inside the ring. and its opacity is that of
                # the ring.
                value = op

            elif (px_inner_rad >= 1.0 and
                (((xpos + half_pixsize) / i_r) ** 2 +
                 ((ypos + half_pixsize) / i_r_min) ** 2 <= 1
                 or
                 ((xpos - half_pixsize) / o_r) ** 2 +
                 ((ypos - half_pixsize) / o_r_min) ** 2 >= 1)):
                # the pixel is wholly outside the planet and its ring so the
                # opacity is zero
                value = 0.0

            else:
                # the pixel is partially covered by the planet and/or ring and
                # so we super-sample the pixel to estimate the opacity of the pixel
                value = -1
                # TODO finish from here


            # send the value to the four relevant places in the image
            image[ihalf+i, jhalf+j] = \
                image[ihalf-i-1, jhalf+j] = \
                image[ihalf+i, jhalf-j-1] = \
                image[ihalf-i-1, jhalf-j-1] = \
                value


    return image



if __name__ == "__main__":
    print("running tests")

    print("quadratic limb darkening...", end=' ')
    qld_test_data = np.load("tests/quad_limb_dark_test_data.npz")
    test_intensities = quad_limb_dark(
        qld_test_data["radii"], *qld_test_data["ld_params"]
    )
    if all((qld_test_data["intensities"] - test_intensities).flatten() == 0):
        print("passed")
    else:
        print("failed")

    import matplotlib.pyplot as plt
    from time import time

    #_ = build_exoring_image(10, 1.0, 1.0, 1.0, 1.0)
    _ = build_exoring_image(100, 1.0, 1.0, 1.0, 1.0)
    t0 = time()
    img = build_exoring_image(
        100,
        2,
        2.5,
        0.5,
        0.1,
        super_sample_factor=10,
        full_output=True,
    )
    print("ex time:", (time() - t0) * 1000.)
    t0 = time()
    img = build_exoring_image(
        100,
        2.1,
        2.5,
        0.5,
        0.1,
        super_sample_factor=10,
        full_output=True,
    )
    print("ex time:", (time() - t0) * 1000.)

    plt.imshow(1-img[0])
    plt.colorbar()

    '''fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(18, 4))
    for i in range(3):
        cmap = None if i==0 else 'bwr'
        im = axes[i].imshow(img[i], origin="lower", cmap=cmap)
        plt.colorbar(im, ax=axes[i])'''

    '''from ringed_planet_transit import build_ringed_occulter

    t0 = time()
    im2, x2, y2, a2 = build_ringed_occulter(rings=[(2.0, 2.5, 0.5)], gamma=0.1,
                                            full_output=True)
    print((time() - t0) * 1000.)

    fig2, axes2 = plt.subplots(ncols=3, nrows=1, figsize=(18, 4))
    img2 = [im2, x2, y2]
    for i in range(3):
        cmap = None if i==0 else 'bwr'
        im = axes2[i].imshow(img2[i], origin="lower", cmap=cmap)
        plt.colorbar(im, ax=axes2[i])
    '''

    plt.show()

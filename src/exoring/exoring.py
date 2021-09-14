#!/usr/bin/env python3

import numpy as np
from numba import njit
import warnings


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
    Build an exoplanet + ring opacity mask.

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

    # check inputs
    if inner_ring_radius >= outer_ring_radius:
        warnings.warn("inner ring radius >= outer ring radius, values swapped "
                      "so they make logical sense")
        inner_ring_radius, outer_ring_radius = \
            outer_ring_radius, inner_ring_radius

    # get the half size of the required grid in each dimension
    ngrid_maj = max(
        np.int_(np.ceil(outer_ring_radius * pixel_scale)), pixel_scale
    )
    ngrid_min = max(
        np.int_(np.ceil(ngrid_maj * np.sin(gamma))), pixel_scale
    )

    # initialise the output arrays
    er_image = np.zeros((2 * ngrid_maj, 2 * ngrid_min), dtype=np.float64).T
    # meshgrid was marginally faster than mgrid in the scenarios tested
    xgrid, ygrid = np.meshgrid(
        (np.arange(-ngrid_maj, ngrid_maj) + 0.5) / pixel_scale,
        (np.arange(-ngrid_min, ngrid_min) + 0.5) / pixel_scale
    )

    # fill the opacity image
    er_image = _fill_opacity_grid(
        xgrid, ygrid, er_image, gamma,
        inner_ring_radius, outer_ring_radius, ring_opacity,
        ssf=super_sample_factor
    )

    # calculate area of each grid element, units of r_planet^2
    elem_area = np.float64(pixel_scale) ** -2

    # remove zero values if requested
    if not full_output:
        ret = er_image > 0.0
        xgrid, ygrid, er_image = map(
            lambda _a: _a[ret], [xgrid, ygrid, er_image]
        )

    return er_image, xgrid, ygrid, elem_area


@njit
def _fill_opacity_grid(xgrid, ygrid, image, gamma, i_r, o_r, op, ssf=10):
    """
    Fill an opacity grid image with a planet and ring.

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
    # obtain some array shape information
    isize, jsize = image.shape
    ihalf = isize // 2
    jhalf = jsize // 2

    # the pixel size in planetary radii (assuming square pixels)
    pixsize = np.float_(xgrid[0, 1] - xgrid[0, 0])
    half_pixsize = pixsize * 0.5

    # super-sample spacing
    fssf = np.float64(ssf)
    ss_gap = pixsize / fssf
    # super sample pixel contribution
    ss_cont = fssf ** -2

    # the minor axis sized of the inner and outer ring ellipses
    i_r_min = np.sin(gamma) * i_r
    o_r_min = np.sin(gamma) * o_r

    # for each pixel of a single quadrant
    for i in range(ihalf):
        for j in range(jhalf):
            # get the position of the centre of the pixel
            xpos = xgrid[ihalf + i, jhalf + j]
            ypos = ygrid[ihalf + i, jhalf + j]

            # positions of the vertices of the pixel
            x_i = xpos - half_pixsize  # x inner
            x_o = xpos + half_pixsize  # x outer
            y_i = ypos - half_pixsize  # y inner
            y_o = ypos + half_pixsize  # y outer
            # pixel outer radius
            px_outer_rad = np.sqrt(x_o ** 2 + y_o ** 2)
            # pixel inner radius
            px_inner_rad = np.sqrt(x_i ** 2 + y_i ** 2)

            if px_outer_rad <= 1.0:
                # The radius of the furthest vertex of the pixel is inside the
                # planet radius, hence the pixel is fully inside the planet and
                # the opacity is total.
                value = 1.0

            elif (px_inner_rad >= 1.0 and (
                    (x_i / i_r) ** 2 + (y_i / i_r_min) ** 2 >= 1 and (
                    x_o / o_r) ** 2 + (y_o / o_r_min) ** 2 <= 1)):
                # The radius of the closest vertex of the pixel is outside the
                # inner radius of the ring, and the radius of its furthest
                # vertex is inside the outer radius of the ring. The inner
                # vertex of the pixel is also outside the planet, Hence the
                # pixel is fully inside the ring but also fully outside the
                # planet and its opacity is that of the ring.
                value = op

            elif (px_inner_rad >= 1.0 and (
                    (x_o / i_r) ** 2 + (y_o / i_r_min) ** 2 <= 1 or (
                    x_i / o_r) ** 2 + (y_i / o_r_min) ** 2 >= 1)):
                # The inner vertex of the pixel is outside the planet.
                # Additionally, the outer vertex of the  pixels is closer than
                # the inner edge of the ring, or the inner vertex is further
                # than the outer edge of the ring. This means the pixel is
                # wholly outside the planet and the ring, hence its opacity
                # value is zero.
                value = 0.0

            else:
                # The pixel is partially covered by the planet and/or ring and
                # so we must super-sample the pixel to estimate the appropriate
                # opacity value for the pixel.
                value = 0.0
                for m in range(ssf):
                    for n in range(ssf):
                        # central position of super sample pixel
                        _xp = x_i + (0.5 + m) * ss_gap
                        _yp = y_i + (0.5 + n) * ss_gap

                        if np.sqrt(_xp ** 2 + _yp ** 2) < 1.0:
                            # the super sample pixel centre is on the planet
                            value = value + ss_cont

                        elif ((_xp / i_r) ** 2 + (_yp / i_r_min) ** 2 >= 1 and
                              (_xp / o_r) ** 2 + (_yp / o_r_min) ** 2 < 1):
                            # the super sample pixel centre is on the ring
                            value = value + op * ss_cont

            # send the value to the four relevant places in the image
            image[ihalf + i, jhalf + j] = \
                image[ihalf - i - 1, jhalf + j] = \
                image[ihalf + i, jhalf - j - 1] = \
                image[ihalf - i - 1, jhalf - j - 1] = \
                value

    return image


def occult_star(
        img,
        xgrid,
        ygrid,
        px_area,
        planet_radius,
        offset_x,
        offset_y,
        obliquity,
        ld_params
):
    """
    Occult a star by the transparency image, producing a light curve.

    Parameters
    ----------
    img : ndarray
        image array
    xgrid : ndarray
        x grid coordinates of the image array
    ygrid : ndarray
        y grid coordinates of the image array
    px_area : float
        the area of each image grid element
    planet_radius : float
        the radius of the planet relative to the radius of the star
    offset_x : ndarray
        shape (N,) numpy array containing x offset positions of the planet at
        the desired times
    offset_y : float
        the constant offset position in the y direction
    obliquity : float
        the angle of the planet relative to its direction of motion in radians
    ld_params : tuple
        length 2 tuple of quadratic limb darkening parameters

    Returns
    -------
    lc : ndarray
        shape (N,) numpy array of the light curve data points
    """
    # initialise the light curve
    lc = np.ones_like(offset_x, dtype=np.float64)

    # populate the light curve
    lc = _fill_light_curve(
        lc,
        img,
        xgrid,
        ygrid,
        px_area,
        planet_radius,
        offset_x,
        offset_y,
        obliquity,
        ld_params
    )

    return lc


@njit
def _fill_light_curve(
        lc,
        img,
        xgrid,
        ygrid,
        px_area,
        planet_radius,
        offset_x,
        offset_y,
        obliquity,
        ld_params
):
    """
    Populate a light curve

    Parameters
    ----------
    lc : ndarray
        shape (N,) numpy array to fill with the light curve data
    img : ndarray
        image array
    xgrid : ndarray
        x grid coordinates of the image array
    ygrid : ndarray
        y grid coordinates of the image array
    px_area : float
        the area of each image grid element
    planet_radius : float
        the radius of the planet relative to the radius of the star
    offset_x : ndarray
        shape (N,) numpy array containing x offset positions of the planet at
        the desired times
    offset_y : float
        the constant offset position in the y direction
    obliquity : float
        the angle of the planet relative to its direction of motion in radians
    ld_params : tuple
        length 2 tuple of quadratic limb darkening parameters

    Returns
    -------
    lc : ndarray
        shape (N,) numpy array of the light curve data points
    """
    # check inputs
    assert lc.shape == offset_x.shape
    assert img.shape == xgrid.shape
    assert img.shape == ygrid.shape

    # rotate the opacity mask by the obliquity
    c_oblq, s_oblq = np.cos(obliquity), np.sin(obliquity)
    _xgrid = xgrid * c_oblq - ygrid * s_oblq
    _ygrid = xgrid * s_oblq + ygrid * c_oblq

    # scale the opacity mask size
    _xgrid = _xgrid * planet_radius
    _ygrid = _ygrid * planet_radius
    # scale the opacity mask element area
    _px_area = px_area * planet_radius ** 2

    # apply the y direction positional offset
    _ygrid += offset_y

    # now for each x offset position
    for i in range(offset_x.size):
        _xg = _xgrid + offset_x[i]

        # radius of each pixel
        px_rad = np.sqrt(_xg ** 2 + _ygrid ** 2)

        if np.min(px_rad) <= 1.0:
            # compute the obscured flux of the star if an occultation occurs
            px_flux = quad_limb_dark(px_rad, *ld_params) * _px_area

            # subtract this flux from the light curve point
            lc[i] = lc[i] - np.sum(img * px_flux)

    return lc

#!/usr/bin/env python3

import numpy as np
from numba import njit
from math import ceil, sin, cos


def build_exoring_image(
        pixel_scale,
        inner_ring_radii,
        outer_ring_radii,
        ring_optical_depths,
        gamma,
        super_sample_factor=10,
        full_output=False
):
    """
    Build an exoplanet + ring opacity mask.

    Parameters
    ----------
    pixel_scale : int
        The scale of the required opacity image, parametrized as the number of
        pixels per planet radius.
    inner_ring_radius : float or array-like
        The inner radius of the ring(s), in units of planet radii.
    outer_ring_radius : float or array-like
        The outer radius of the ring(s), in units of planet radii.
    ring_optical_depths : float or array-like
        The normal optical depths of the ring(s).
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

    # verify we have nice flat numpy arrays
    outer_ring_radii = np.asarray(outer_ring_radii).flatten()
    inner_ring_radii = np.asarray(inner_ring_radii).flatten()
    ring_optical_depths = np.asarray(ring_optical_depths).flatten()

    # check that the inner ring radii are always larger than the outer radii
    radii = np.column_stack((inner_ring_radii, outer_ring_radii)).copy()
    inner_ring_radii = np.min(radii, axis=1)
    outer_ring_radii = np.max(radii, axis=1)

    # verify that the rings don't overlap if there are multiple
    if np.any((inner_ring_radii[1:] - outer_ring_radii[:-1]) < 0):
        raise RuntimeError("Rings cannot overlap")

    # verify that we have the same number of each ring paramter
    assert outer_ring_radii.size == inner_ring_radii.size
    assert inner_ring_radii.size == ring_optical_depths.size

    # convert normal optical depths to opacities
    ring_opacities = 1 - np.exp(-ring_optical_depths/sin(gamma))

    # get the half size of the required grid in each dimension
    max_outer_rad_px = np.max(outer_ring_radii) * pixel_scale
    ngrid_maj = max(ceil(max_outer_rad_px), pixel_scale)
    ngrid_min = max(ceil(max_outer_rad_px * sin(gamma)), pixel_scale)

    # initialise the output arrays
    er_image = np.zeros((2 * ngrid_maj, 2 * ngrid_min), dtype=np.float64).T
    # meshgrid was marginally faster than mgrid in the scenarios tested
    xgrid, ygrid = np.meshgrid(
        (np.arange(-ngrid_maj, ngrid_maj) + 0.5) / pixel_scale,
        (np.arange(-ngrid_min, ngrid_min) + 0.5) / pixel_scale
    )

    # fill the opacity image
    # there are performance gains made by running single rings through a
    # simplified image generation function (roughly a factor of 2 during my
    # limited testing), hence the different functions
    if inner_ring_radii.size == 1:
        er_image = _fill_opacity_grid_single(
            xgrid, ygrid, er_image, gamma,
            inner_ring_radii[0], outer_ring_radii[0], ring_opacities[0],
            ssf=super_sample_factor
        )
    else:
        er_image = _fill_opacity_grid(
            xgrid, ygrid, er_image, gamma,
            inner_ring_radii, outer_ring_radii, ring_opacities,
            ssf=super_sample_factor
        )

    # calculate area of each grid element, units of r_planet^2
    elem_area = float(pixel_scale) ** -2

    # remove zero values if requested
    if not full_output:
        ret = er_image > 0.0
        xgrid, ygrid, er_image = map(
            lambda _a: _a[ret], [xgrid, ygrid, er_image]
        )

    return er_image, xgrid, ygrid, elem_area


@njit
def _fill_opacity_grid_single(xgrid, ygrid, image, gamma, i_r, o_r, op, ssf=10):
    """
    Fill an opacity grid image with a planet and a single ring.

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
    pixsize = float(xgrid[0, 1] - xgrid[0, 0])
    half_pixsize = pixsize * 0.5

    # super-sample spacing
    fssf = float(ssf)
    ss_gap = pixsize / fssf
    # super sample pixel contribution
    ss_cont = fssf ** -2

    # the minor axis sized of the inner and outer ring ellipses
    i_r_min = sin(gamma) * i_r
    o_r_min = sin(gamma) * o_r

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
            px_outer_rad = (x_o ** 2 + y_o ** 2) ** 0.5
            # pixel inner radius
            px_inner_rad = (x_i ** 2 + y_i ** 2) ** 0.5

            if px_outer_rad <= 1.0:
                # The radius of the furthest vertex of the pixel is inside the
                # planet radius, hence the pixel is fully inside the planet and
                # the opacity is total.
                value = 1.0

            elif (px_inner_rad >= 1.0 and
                  ((x_i / i_r) ** 2 + (y_i / i_r_min) ** 2 >= 1
                   and (x_o / o_r) ** 2 + (y_o / o_r_min) ** 2 <= 1)):
                # The radius of the closest vertex of the pixel is outside the
                # inner radius of the ring, and the radius of its furthest
                # vertex is inside the outer radius of the ring. The inner
                # vertex of the pixel is also outside the planet, Hence the
                # pixel is fully inside the ring but also fully outside the
                # planet and its opacity is that of the ring.
                value = op

            elif (px_inner_rad >= 1.0 and
                  ((x_o / i_r) ** 2 + (y_o / i_r_min) ** 2 <= 1
                   or (x_i / o_r) ** 2 + (y_i / o_r_min) ** 2 >= 1)):
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

                        if (_xp ** 2 + _yp ** 2) ** 0.5 < 1.0:
                            # the super sample pixel centre is on the planet
                            value = value + ss_cont

                        elif ((_xp / i_r) ** 2 + (_yp / i_r_min) ** 2 >= 1
                              and
                              (_xp / o_r) ** 2 + (_yp / o_r_min) ** 2 < 1):
                            # the super sample pixel centre is on the ring
                            value = value + op * ss_cont

            # send the value to the four relevant places in the image
            image[ihalf + i, jhalf + j] = value
            image[ihalf - i - 1, jhalf + j] = value
            image[ihalf + i, jhalf - j - 1] = value
            image[ihalf - i - 1, jhalf - j - 1] = value

    return image


@njit(parallel=True)
def _fill_opacity_grid(xgrid, ygrid, image, gamma, i_rs, o_rs, ops, ssf=10):
    """
    Fill an opacity grid image with a planet and multiple rings.

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
    i_rs : ndarray
        ring inner radii
    o_rs : ndarray
        ring outer radii
    ops : ndarray
        ring opacities
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
    pixsize = float(xgrid[0, 1] - xgrid[0, 0])
    half_pixsize = pixsize * 0.5

    # super-sample spacing
    fssf = float(ssf)
    ss_gap = pixsize / fssf
    # super sample pixel contribution
    ss_cont = fssf ** -2

    # the minor axis sizes of the inner and outer ring ellipses
    i_r_mins = sin(gamma) * i_rs
    o_r_mins = sin(gamma) * o_rs

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
            px_outer_rad = (x_o ** 2 + y_o ** 2) ** 0.5
            # pixel inner radius
            px_inner_rad = (x_i ** 2 + y_i ** 2) ** 0.5

            # set finished flag
            fin = False

            if not fin and px_outer_rad <= 1.0:
                # The radius of the furthest vertex of the pixel is inside the
                # planet radius, hence the pixel is fully inside the planet and
                # the opacity is total.
                value = 1.0
                # no need to continue checking this pixel, set the fin flag True
                fin = True

            if not fin and px_inner_rad >= 1.0:
                # for each ring...
                for n in range(i_rs.size):
                    if (x_i / i_rs[n]) ** 2 + \
                            (y_i / i_r_mins[n]) ** 2 >= 1 \
                            and \
                            (x_o / o_rs[n]) ** 2 + \
                            (y_o / o_r_mins[n]) ** 2 <= 1:
                        # The radius of the closest vertex of the pixel is
                        # outside the inner radius of the ring, and the radius
                        # of its furthest vertex is inside the outer radius of
                        # the ring. The inner vertex of the pixel is also
                        # outside the planet, Hence the pixel is fully inside
                        # the ring but also fully outside the planet and its
                        # opacity is that of the ring.
                        value = ops[n]
                        # no need to continue checking this pixel, set the fin
                        # flag True
                        fin = True
                        break

            if not fin and px_inner_rad >= 1.0:
                # we're not finished yet and the pixel is wholly outside the
                # planet
                # flags for where the pixel is wholly outside each ring
                outside_ring = np.zeros(i_rs.size, dtype=np.bool_)
                # for each ring...
                for n in range(i_rs.size):
                    if (x_o / i_rs[n]) ** 2 + \
                            (y_o / i_r_mins[n]) ** 2 <= 1 \
                            or (x_i / o_rs[n]) ** 2 + \
                            (y_i / o_r_mins[n]) ** 2 >= 1:
                        # The outer vertex of the pixel is closer than the inner
                        # edge of the ring, or the inner vertex is further than
                        # the outer edge of the ring. This means the pixel is
                        # wholly outside the ring.
                        outside_ring[n] = True

                if np.all(outside_ring):
                    # if pixel is wholly outside all rings set the opacity to
                    # zero
                    value = 0.0
                    # no need to continue checking this pixel, set the fin flag
                    # True
                    fin = True

            if not fin:
                # The pixel is partially covered by the planet and/or a ring and
                # so we must super-sample the pixel to estimate the appropriate
                # opacity value for the pixel.
                value = 0.0
                for m in range(ssf):
                    for n in range(ssf):
                        # central position of super sample pixel
                        _xp = x_i + (0.5 + m) * ss_gap
                        _yp = y_i + (0.5 + n) * ss_gap

                        if (_xp ** 2 + _yp ** 2) ** 0.5 < 1.0:
                            # the super sample pixel centre is on the planet
                            value = value + ss_cont

                        else:
                            # for each ring...
                            for n in range(i_rs.size):
                                if (_xp / i_rs[n]) ** 2 + \
                                        (_yp / i_r_mins[n]) ** 2 >= 1 \
                                        and (_xp / o_rs[n]) ** 2 + \
                                        (_yp / o_r_mins[n]) ** 2 < 1:
                                    # the super sample pixel centre is on this
                                    # ring, take its opacity value
                                    value = value + ops[n] * ss_cont
                                    # no need to continue checking the rings
                                    break

            # send the value to the four relevant places in the image
            image[ihalf + i, jhalf + j] = value
            image[ihalf - i - 1, jhalf + j] = value
            image[ihalf + i, jhalf - j - 1] = value
            image[ihalf - i - 1, jhalf - j - 1] = value

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


@njit(parallel=True)
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

    # break out limd darkening parameters
    ld_a, ld_b = ld_params

    # rotate the opacity mask by the obliquity
    c_oblq, s_oblq = cos(obliquity), sin(obliquity)
    _xgrid = xgrid * c_oblq - ygrid * s_oblq
    _ygrid = xgrid * s_oblq + ygrid * c_oblq

    # scale the opacity mask size
    _xgrid = _xgrid * planet_radius
    _ygrid = _ygrid * planet_radius
    # scale the opacity mask element area
    _px_area = px_area * planet_radius ** 2

    # apply the y direction positional offset
    _ygrid += offset_y

    # if the minimum y pixel position is outside the star it will never transit,
    # return the unchanged lightcurve
    _miny = np.min(_ygrid)
    if _miny > 1:
        return lc

    # max extent in x
    _maxx = np.max(_xgrid)

    # now for each x offset position
    for i in range(offset_x.size):

        # skip if the object has no hope of transiting
        if offset_x[i] - _maxx > 1:
            continue

        # apply the x offset
        _xg = _xgrid + offset_x[i]

        # radius of each pixel
        px_rad = (_xg ** 2 + _ygrid ** 2) ** 0.5

        # numba requires 1d arrays for some operations, ravels are views
        flat_rad = px_rad.ravel()
        flat_img = img.ravel()

        # identify pixels that are on the star
        le1 = np.abs(flat_rad) <= 1.0

        # evaluate limb darkening model for these pixels
        c_mu = 1.0 - np.cos(np.arcsin(flat_rad[le1]))
        flat_i = ((1 - ld_a * c_mu - ld_b * c_mu ** 2)
                  / (1 - ld_a / 3 - ld_b / 6) / np.pi) * _px_area

        # subtract this flux from the light curve point
        lc[i] = lc[i] - np.sum(flat_img[le1] * flat_i)

    return lc

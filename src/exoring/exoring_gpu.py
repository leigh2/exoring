#!/usr/bin/env python3

from configobj import ConfigObj
import numpy as np
from math import ceil, sin, cos, asin, pi, log2, exp
from numba import cuda

# make sure a cuda enabled device is available
assert cuda.is_available()

# GPU shared array shape must be set at compile time, this means it and it's
# precursors must be defined here. Read the precursors from a config file, and
# then define the shape of the shared array.

# open the config file
gpu_config = ConfigObj('gpu_config.cfg')

# gpu threads per block for image generation
gpu_tpb_imgen = tuple([int(f) for f in gpu_config['gpu_tpb_imgen']])
# gpu threads per block for light curve generation
gpu_tpb_lcgen = tuple([int(f) for f in gpu_config['gpu_tpb_lcgen']])
# gpu threads per block for light curve reduction operation
gpu_tpb_lcsum = gpu_config.as_int('gpu_tpb_lcsum')

# verify the tuples are the correct length
assert len(gpu_tpb_imgen) == 2
assert len(gpu_tpb_lcgen) == 3
# todo
# verify values are powers of 2

# gpu shared array shape *must* be set at compile time
sarr_shape = (
    gpu_tpb_lcgen[0] * gpu_tpb_lcgen[1],
    gpu_tpb_lcgen[2]
)


class ExoRing:
    """
    Exoring light curve generation class.
    """

    def __init__(
            self,
            planet_scale=200,
            super_sample_factor=10,
            img_array_shape=(512, 1024)
    ):
        """
        Initialise the exoring light curve generation class instance

        Parameters
        ----------
        planet_scale : int, optional
            The number of image array elements per planet radius, this must fit
            within the first element of `img_array_shape`. (Default: 200.)
        super_sample_factor : int, optional
            Pixels which straddle the planet and/or ring boundary are
            super-sampled to estimate the appropriate opacity.
            `super_sample_factor` is the number of subdivisions along each
            dimension. (Default: 10.)
        img_array_shape : tuple, optional
            Tuple of length 2, dictating the shape of the on-device image array
            in which to build the opacity images. Each dimension size should be
            a power of two. (Default: (512, 1024).)
        """
        # todo
        # verify values are powers of 2

        # verify img_array_shape is the correct length
        assert len(img_array_shape) == 2
        # verify that the planet fits in the device image array
        assert planet_scale <= img_array_shape[0]

        # pixel scale, i.e. number of image array elements per planet radius
        self.planet_scale = planet_scale
        # pixel size, i.e. planetary radii per pixel
        self.pixel_size = 1.0 / planet_scale

        # super_sample_factor is the number of subdivisions along each dimension
        # in which to split each image array element when evaluating the mean
        # opacity of an element which spans a ring or planet edge
        self.super_sample_factor = super_sample_factor
        # super-sample spacing
        fssf = float(self.super_sample_factor)
        self.ss_gap = self.pixel_size / fssf
        # super sample pixel contribution
        self.ss_cont = fssf ** -2

        # gpu threads per block for image generation
        self.gpu_tpb_imgen = gpu_tpb_imgen
        # opacity image array
        self.img_array_shape = img_array_shape
        # gpu blocks per grid for image generation
        self.gpu_bpg_imgen = (
            ceil(img_array_shape[0] / gpu_tpb_imgen[0]),
            ceil(img_array_shape[1] / gpu_tpb_imgen[1])
        )

        # open the image array on the device
        # 32 bit floats should be sufficient for a final precision a little
        # better than 0.1 ppm
        self.img_array = cuda.device_array(img_array_shape, dtype=np.float32)

        # gpu threads per block for light curve generation
        self.gpu_tpb_lcgen = gpu_tpb_lcgen
        # gpu threads per block for light curve reduction operation
        self.gpu_tpb_lcsum = gpu_tpb_lcsum

    def build_image(self,
                    inner_ring_radius, outer_ring_radius,
                    ring_optical_depth, gamma):
        """
        Build an exoplanet-plus-ring opacity mask on the gpu.

        Parameters
        ----------
        inner_ring_radius : float or array-like
            The inner radius of the ring in units of planet radii.
        outer_ring_radius : float or array-like
            The outer radius of the ring in units of planet radii.
        ring_optical_depth : float or array-like
            The normal optical depth of the ring.
        gamma : float
            Inclination angle relative to the line of sight to the observer in
            radians. At an angle of 0.0 radians the ring is invisible as ring
            depth is not modelled.
        """
        # sin gamma
        singamma = sin(gamma)

        # verify that the ring fits in the device image array
        assert self.planet_scale * outer_ring_radius <= self.img_array_shape[1]
        assert self.planet_scale * outer_ring_radius * singamma \
               < self.img_array_shape[0]

        # convert normal optical depth to opacity
        ring_opacity = 1 - exp(-ring_optical_depth / singamma)

        # the minor axis size of the inner and outer ring ellipses
        i_r_min = singamma * inner_ring_radius
        o_r_min = singamma * outer_ring_radius

        # fill the image array on the gpu
        _fill_image_gpu[self.gpu_bpg_imgen, self.gpu_tpb_imgen](
            self.img_array, self.pixel_size, i_r_min, inner_ring_radius,
            o_r_min, outer_ring_radius, ring_opacity, self.super_sample_factor,
            self.ss_gap, self.ss_cont
        )

    def read_image(self, return_full=True):
        """
        Copy the exoring opacity image from the gpu to the host and return it,
        mirrored or otherwise.

        Parameters
        ----------
        return_full : bool, optional
            If True, mirror the single quadrant across each axis to produce the
            full image (default behaviour). If False only return the single
            quadrant.

        Returns
        -------
        The opacity array of the exoplanet, either a single quadrant (in which
        case the array is the same shape as `img_array_shape`) or mirrored in
        both dimensions (in which case it is twice `img_array_shape` in each
        dimension).
        """
        # grab the single quadrant array from the gpu
        img1q = self.img_array.copy_to_host()
        if not return_full:
            # return the single quadrant
            return img1q
        else:
            # mirror about both axis
            img4q = np.block([
                [np.flip(img1q), np.flip(img1q, axis=0)],
                [np.flip(img1q, axis=1), img1q]
            ])
            return img4q

    def get_k(self):
        """
        Measure the effective radius scaling factor of the planet plus ring.

        Returns
        -------
        k
        """
        # read the image from the device, only the primary quadrant is needed
        img = self.read_image(return_full=False)

        # sum of opacity elements in this single quadrant
        total_opacity = img.sum()
        # the sum of the opacity elements contributed by the planet in this
        # quadrant
        planet_opacity = 0.25 * pi * self.planet_scale**2

        # calculate the radius scaling factor parameter
        k = (total_opacity / planet_opacity)**0.5

        return k

    def occult_star(
            self,
            x_array, y_array,
            planet_radius, obliquity, ld_params
    ):
        """
        Produce a transit of the pre-generated opacity profile array across a star.

        Parameters
        ----------
        x_array : ndarray
            1D Numpy array of (N,) 'X' positions of the centre of the planet
            relative to the centre of the star in units of stellar radii.
        y_array : ndarray
            1D Numpy array of (N,) 'Y' positions of the centre of the planet
            relative to the centre of the star in units of stellar radii.
        planet_radius : float
            The radius of the planet in units of stellar radii.
        obliquity : float
            The inclination in radians of the planet in the plane of its orbit.
            Runs clockwise, positive from zero parallel to the 'X' axis.
        ld_params : tuple
            Tuple of two quadratic limb darkening parameters for the star

        Returns
        -------
        1D numpy array of (N,) fractional flux values relative to baseline for
        the requested 'X' and 'Y' positions.
        """
        # copy the x and y arrays to the device
        _xa = cuda.to_device(x_array)
        _ya = cuda.to_device(y_array)
        # open an output lightcurve array on the device
        lc = cuda.device_array(x_array.size, dtype=np.float32)

        # blocks per grid for the lc generation
        gpu_bpg_lcgen = (
            ceil(self.img_array_shape[0] / self.gpu_tpb_lcgen[0]),
            ceil(self.img_array_shape[1] / self.gpu_tpb_lcgen[1]),
            ceil(x_array.size / self.gpu_tpb_lcgen[2])
        )

        # open a temporary array on the device to put the per-block results in
        _block_results = cuda.device_array(
            (x_array.size, gpu_bpg_lcgen[0] * gpu_bpg_lcgen[1]),
            dtype=np.float32
        )

        # pixel contribution calculation
        _get_pixel_contrib[gpu_bpg_lcgen, self.gpu_tpb_lcgen](
            _xa, _ya, self.img_array, _block_results, self.pixel_size,
            planet_radius, obliquity, ld_params
        )

        # reduce the block results to produce the lightcurve using:
        # delta_flux = 1 - sum(_block_results, axis=1)
        gpu_bpg_lcsum = ceil(x_array.size / self.gpu_tpb_lcsum)
        _sum_lc_contrib[gpu_bpg_lcsum, self.gpu_tpb_lcsum](_block_results, lc)

        return lc.copy_to_host()


@cuda.jit
def _get_pixel_contrib(xa, ya, img, bres, pxsize, prad, oblqty, ldpars):
    """
    Evaluate the fraction of flux blocked for each opacity grid element for each
    planet position point. Run a first pass sum reduction to reduce the
    resultant large 3D grid to a smaller 2D grid.

    Parameters
    ----------
    xa : ndarray
        1D array of (N,) 'X' positions of the centre of the planet relative to
        the centre of the star in units of stellar radii.
    ya : ndarray
        1D array of (N,) 'Y' positions of the centre of the planet relative to
        the centre of the star in units of stellar radii.
    img : ndarray
        2D opacity grid array.
    bres : ndarray
        Temporary 2D array in which to place the result of the first pass sum
        reduction.
    pxsize : float
        The size of each opacity array element in units of planetary radii
        squared.
    prad : float
        The radius of the planet in units of stellar radii.
    oblqty : float
        The inclination in radians of the planet in the plane of its orbit. Runs
        clockwise, positive from zero parallel to the 'X' axis.
    ldpars : tuple
        Tuple of two quadratic limb darkening parameters for the star
    """
    # break out the limb darkening parameters
    ld_a, ld_b = ldpars

    # open a working array in shared memory
    sarr = cuda.shared.array(shape=sarr_shape, dtype=np.float32)

    # shared array thread index
    ij_idx = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x
    k_idx = cuda.threadIdx.z

    # thread absolute position
    i, j, k = cuda.grid(3)

    # stop this thread if outside image boundary
    if i >= img.shape[0] or j >= img.shape[1] or k >= xa.shape[0]:
        sarr[ij_idx, k_idx] = 0.0
        return

    # set this element in the shared array to the opacity value of this pixel
    sarr[ij_idx, k_idx] = img[i, j]

    # proceed with the following block only if there is nonzero opacity of this
    # pixel
    if sarr[ij_idx, k_idx] > 0.0:

        # calculate the four relevant pixel positions relative to the planet
        # centre, running clockwise from the upper right quadrant (the opacity
        # image is only one quadrant and biaxial symmetry is assumed).
        x0 = (j + 0.5) * pxsize  # x upper right quadrant
        y0 = (i + 0.5) * pxsize  # y upper right quadrant
        x1 = x0  # x lower right quadrant
        y1 = -y0  # y lower right quadrant
        x2 = -x0  # x upper left quadrant
        y2 = -y0  # y upper left quadrant
        x3 = -x0  # x lower left quadrant
        y3 = y0  # y lower left quadrant

        # cosine and sine of the obliquity
        c_oblq, s_oblq = cos(oblqty), sin(oblqty)
        # rotate and scale the pixel positions to units of stellar radii
        _x0 = (x0 * c_oblq - y0 * s_oblq) * prad
        _y0 = (x0 * s_oblq + y0 * c_oblq) * prad
        _x1 = (x1 * c_oblq - y1 * s_oblq) * prad
        _y1 = (x1 * s_oblq + y1 * c_oblq) * prad
        _x2 = (x2 * c_oblq - y2 * s_oblq) * prad
        _y2 = (x2 * s_oblq + y2 * c_oblq) * prad
        _x3 = (x3 * c_oblq - y3 * s_oblq) * prad
        _y3 = (x3 * s_oblq + y3 * c_oblq) * prad

        # calculate the area of each pixel in stellar radii squared, multiply
        # this into the light curve contribution of the shared array
        sarr[ij_idx, k_idx] *= ((pxsize * prad) ** 2)

        # read the x and y offsets for this light curve point
        # the compiler should be good enough to not do this multiple times if
        # placed in the below loop, but just in case it's not...
        x_off = xa[k]
        y_off = ya[k]

        # sum of stellar intensities covered by these four pixels
        _intensity_sum = 0.0

        # for each pixel
        for xx, yy in [(_x0, _y0), (_x1, _y1), (_x2, _y2), (_x3, _y3)]:

            # calculate its radius at this light curve point
            rad = ((xx + x_off) ** 2 + (yy + y_off) ** 2) ** 0.5

            # if it's on the star, calculate the intensity of the star at this
            # position and include it in the sum
            if rad <= 1.0:
                mu = 1.0 - cos(asin(rad))
                fr = ((1 - ld_a * mu - ld_b * mu ** 2)
                      / (1 - ld_a / 3 - ld_b / 6) / pi)
                _intensity_sum += fr

        # multiply in the blocked stellar intensity to the light curve
        # contribution
        sarr[ij_idx, k_idx] *= _intensity_sum

    # sync all threads in this block
    cuda.syncthreads()

    # array sum reduction
    for ll in range(log2(sarr_shape[0]), -1, -1):
        lim = 1 << ll
        if ij_idx < lim:
            sarr[ij_idx, k_idx] += sarr[ij_idx + lim, k_idx]
        cuda.syncthreads()

    # fill the block result element with the sum
    if ij_idx == 0:
        blkidx = cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x
        bres[k, blkidx] = sarr[0, k_idx]


@cuda.jit
def _sum_lc_contrib(inarray, outarray):
    """
    A sum reduction of a two dimensional input array along axis 1 into a one
    dimensional output array.

    Parameters
    ----------
    inarray : ndarray
        Input numpy array of shape (N,M).
    outarray : ndarray
        Output numpy array of shape (N,).

    Todo
    ----
    This method would benefit from some tweaking to improve GPU utilisation.
    Mainly it will improve performance a little, but it will also stop the
    annoying NumbaPerformanceWarning.
    """
    i = cuda.grid(1)
    # quit if out of bounds
    if i > inarray.shape[0]:
        return

    # 1 - sum of all the elements in this row
    _tmp = 1.0
    for j in range(inarray.shape[1]):
        _tmp -= inarray[i, j]

    # send the result to the output array
    outarray[i] = _tmp


@cuda.jit
def _fill_image_gpu(image, pixsize,
                    ir_min, ir_maj, or_min, or_maj, op,
                    ssf, ss_gap, ss_cont):
    """
    Fill the opacity grid array elements with a planet-plus-ring on the gpu.

    Parameters
    ----------
    image : ndarray
        the image array
    pixsize : float
        the pixel size in planetary radii
    ir_min : float
        minor axis radius of ring inner edge
    ir_maj : float
        major axis radius of ring inner edge
    or_min : float
        minor axis radius of ring outer edge
    or_maj : float
        major axis radius of ring outer edge
    op : float
        ring opacity
    ssf : int
        super sample factor
    ss_gap : float
        the size of a super-sample element in planetary radii
    ss_cont : float
        the fractional contribution of a super sample element to its pixel
    """
    # cuda iterator
    i, j = cuda.grid(2)
    if i < image.shape[0] and j < image.shape[1]:
        # fill this element

        # calculate the positions of the vertices of the pixel
        x_i = j * pixsize  # x inner
        x_o = (j + 1) * pixsize  # x outer
        y_i = i * pixsize  # y inner
        y_o = (i + 1) * pixsize  # y outer
        # pixel inner and outer radii
        px_inner_rad = (x_i ** 2 + y_i ** 2) ** 0.5
        px_outer_rad = (x_o ** 2 + y_o ** 2) ** 0.5

        if px_outer_rad <= 1.0:
            # The radius of the furthest vertex of the pixel is inside the
            # planet radius, hence the pixel is fully inside the planet and
            # the opacity is total.
            value = 1.0

        elif (px_inner_rad >= 1.0 and
              ((x_i / ir_maj) ** 2 + (y_i / ir_min) ** 2 >= 1
               and (x_o / or_maj) ** 2 + (y_o / or_min) ** 2 <= 1)):
            # The radius of the closest vertex of the pixel is outside the
            # inner radius of the ring, and the radius of its furthest
            # vertex is inside the outer radius of the ring. The inner
            # vertex of the pixel is also outside the planet, Hence the
            # pixel is fully inside the ring but also fully outside the
            # planet and its opacity is that of the ring.
            value = op

        elif (px_inner_rad >= 1.0 and
              ((x_o / ir_maj) ** 2 + (y_o / ir_min) ** 2 <= 1
               or (x_i / or_maj) ** 2 + (y_i / or_min) ** 2 >= 1)):
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

                    elif ((_xp / ir_maj) ** 2 + (_yp / ir_min) ** 2 >= 1 and
                          (_xp / or_maj) ** 2 + (_yp / or_min) ** 2 < 1):
                        # the super sample pixel centre is on the ring
                        value = value + op * ss_cont

        # send the value to the image pixel
        image[i, j] = value

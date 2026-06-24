#!/usr/bin/env python3

import os
from math import ceil, sin, cos, pi, exp
import numpy as np
from pycuda.compiler import SourceModule
from pycuda import gpuarray

_cu_src = None
_get_xy_kernel = None
_fill_image_kernel = None
_pixel_contrib_kernel = None
_chisq_reduce_kernel = None


def _ensure_kernels():
    """
    Compile exoring.cu and fetch its kernel functions, exactly once per
    process (guarded by the `_cu_src is not None` check below) - module-level
    globals rather than a per-instance attribute, so that multiple ExoRing
    instances in the same process all share one compiled module instead of
    each re-running nvcc.
    """
    global _cu_src
    global _get_xy_kernel
    global _fill_image_kernel, _pixel_contrib_kernel, _chisq_reduce_kernel
    if _cu_src is not None:
        return
    import pycuda.autoinit
    cu_src_file_path = os.path.join(os.path.dirname(__file__), "exoring.cu")
    with open(cu_src_file_path, "r") as cu_src_file:
        _cu_src = SourceModule(
            cu_src_file.read(), options=["-use_fast_math", "-O3"]
        )
    _get_xy_kernel = _cu_src.get_function("get_xy")
    _fill_image_kernel = _cu_src.get_function("fill_image")
    _pixel_contrib_kernel = _cu_src.get_function("pixel_contrib_accumulate")
    _chisq_reduce_kernel = _cu_src.get_function("chisq_reduce")
    _get_xy_kernel.prepare("PPPddddddi")
    _fill_image_kernel.prepare("Piiffffff")
    _pixel_contrib_kernel.prepare("PPiPiiffffffP")
    _chisq_reduce_kernel.prepare("PPPdiP")


def to_gpu(arr, dtype):
    """
    Convenience function for sending an array to the gpu with a specific data
    type.

    Parameters
    ----------
    arr : array-like
        The array to send to the gpu
    dtype : dtype
        The numpy data type to use

    Returns
    -------
    A gpuarray
    """
    return gpuarray.to_gpu(np.ascontiguousarray(arr, dtype=dtype))


class ExoRing:
    """
    Exoring light curve generation class.

    Typical usage:
        1. construct an `ExoRing` instance
        2. `build_image` - builds the opacity mask for a given ring geometry
           on the device
        3. `put_xy_array` / `put_observed_lc` - upload the orbit positions and
           observed light curve once; these stay resident on the device
        4. `occult_star` - projects the opacity mask onto the star to produce
           a model light curve, given the current ring geometry's image
        5. `read_lightcurve` and/or `get_loglikelihood` - read results back

    In an MCMC fit, steps 2 and 4 (and `get_loglikelihood`) are the hot path,
    repeated every step as the ring geometry parameters change; steps 1 and 3
    only need to happen once at the start of a run.
    """

    def __init__(
            self,
            planet_scale=200,
            img_array_shape=(512, 1024),
            imgen_block=(32, 32),
            lcgen_block=(16, 16),
            lcsum_block=1024,
            xygen_block=256
    ):
        """
        Initialise the exoring light curve generation class instance

        Parameters
        ----------
        planet_scale : int, optional
            The number of image array elements per planet radius, this must fit
            within the first element of `img_array_shape`. (Default: 200,
            which gives a model flux error better than 1ppm relative to a
            converged (planet_scale=800) reference - see
            scripts/precision_grid_size.py.)
        img_array_shape : tuple, optional
            Tuple of length 2, dictating the shape of the on-device image array
            in which to build the opacity images. (Default: (512, 1024).)
        imgen_block : tuple, optional
            CUDA threads-per-block (2D) for the opacity image generation
            kernel. (Default: (32, 32).)
        lcgen_block : tuple, optional
            CUDA threads-per-block (2D) for the light curve generation kernel
            - each thread covers one opacity-image pixel and internally loops
            over every light curve point. `lcgen_block[0] * lcgen_block[1]`
            must be a multiple of 32. (Default: (16, 16).)
        lcsum_block : int, optional
            CUDA threads-per-block for the chi-square reduction kernel.
            (Default: 1024.)
        xygen_block : int, optional
            CUDA threads-per-block for the orbital position (Kepler solver)
            kernel - one thread per light curve point. (Default: 256.)
        """
        _ensure_kernels()

        # verify img_array_shape is the correct length
        assert len(img_array_shape) == 2
        # verify that the planet fits in the device image array
        assert planet_scale <= img_array_shape[0]
        # the (i,j)-plane block size must be a whole number of warps so that
        # the warp-shuffle reduction in pixel_contrib_accumulate is valid
        assert (lcgen_block[0] * lcgen_block[1]) % 32 == 0

        # pixel scale, i.e. number of image array elements per planet radius
        self.planet_scale = planet_scale
        # pixel size, i.e. planetary radii per pixel
        self.pixel_size = 1.0 / planet_scale

        # opacity image array, on the device
        self.img_array_shape = img_array_shape
        self.img_array = gpuarray.zeros(img_array_shape, dtype=np.float32)

        # fill_image launch configuration, cached since the image shape is
        # fixed for the life of this instance. blockIdx.x/threadIdx.x track
        # columns and blockIdx.y/threadIdx.y track rows (not the other way
        # round) so that the kernel's image reads/writes are coalesced - see
        # the comment in exoring.cu
        self.imgen_block = (int(imgen_block[0]), int(imgen_block[1]), 1)
        self.imgen_grid = (
            ceil(img_array_shape[1] / imgen_block[0]),
            ceil(img_array_shape[0] / imgen_block[1]),
            1
        )

        # pixel_contrib_accumulate launch configuration. One thread per
        # opacity-image pixel (it internally loops over every light curve
        # point - see exoring.cu), so like imgen_grid this only depends on
        # the image shape, not on the light curve size.
        self.lcgen_block = (int(lcgen_block[0]), int(lcgen_block[1]), 1)
        self.lcgen_grid = (
            ceil(img_array_shape[1] / lcgen_block[0]),
            ceil(img_array_shape[0] / lcgen_block[1]),
            1
        )
        n_warps_lcgen = (lcgen_block[0] * lcgen_block[1] + 31) // 32
        self._lcgen_smem = n_warps_lcgen * 8  # bytes (float64 per warp)

        # chisq_reduce launch configuration
        self.lcsum_block = int(lcsum_block)
        n_warps_lcsum = (lcsum_block + 31) // 32
        self._lcsum_smem = n_warps_lcsum * 8  # bytes (float64 per warp)

        # get_xy launch configuration
        self.xygen_block = int(xygen_block)

        # initialise some instance variables
        self.times_array = None
        self.x_array = None
        self.y_array = None
        self.n_pts = None
        self.offset_flux = None
        self.flux_error = None
        self.lc_accum = None
        self._chisq = None

        # blocks per grid for the lc reduction and orbital position kernels,
        # set once the light curve size is known (see put_xy_array /
        # put_times_array)
        self.lcsum_grid = None
        self.xygen_grid = None

    def put_xy_array(self, xarray, yarray):
        """
        Send a precomputed on-sky planet position trajectory relative to the
        star position to the device. Use this when the (x,y) trajectory is
        computed externally (e.g. in tests); for fitting orbital parameters
        in an MCMC loop, use `put_times_array` once and `compute_xy_array`
        per step instead, so the trajectory is computed on-device from
        orbital elements without a host round-trip.

        Parameters
        ----------
        xarray : ndarray
            1D Numpy array of (N,) 'X' positions of the centre of the planet
            relative to the centre of the star in units of stellar radii.
        yarray : ndarray
            1D Numpy array of (N,) 'Y' positions of the centre of the planet
            relative to the centre of the star in units of stellar radii.
        """
        # input verification
        assert xarray.shape == yarray.shape
        assert len(xarray.shape) == 1
        # send arrays to device. float32 is enough here - pixel_contrib_accumulate
        # immediately truncates these to float anyway, so this loses no
        # precision versus float64 while halving the bytes read in its hot loop
        self.x_array = to_gpu(xarray, np.float32)
        self.y_array = to_gpu(yarray, np.float32)
        self.n_pts = xarray.size

        # blocks per grid for the lc reduction kernel (lcgen_grid is fixed at
        # construction time - see __init__ - since it no longer depends on
        # the light curve size)
        self.lcsum_grid = (ceil(self.n_pts / self.lcsum_block), 1, 1)

    def put_times_array(self, times):
        """
        Send the light curve observation times to the device once, and
        allocate the on-device (x,y) planet position buffers that
        `compute_xy_array` fills in on every subsequent call. Use this
        (instead of `put_xy_array`) when fitting orbital parameters, so the
        time series is uploaded only once per run while the planet's sky
        position is recomputed on-device from orbital elements at every MCMC
        step.

        Parameters
        ----------
        times : ndarray
            1D Numpy array of (N,) light curve observation time points.
        """
        # input verification
        assert len(times.shape) == 1

        self.times_array = to_gpu(times, np.float64)
        self.n_pts = times.size
        self.x_array = gpuarray.zeros(self.n_pts, dtype=np.float32)
        self.y_array = gpuarray.zeros(self.n_pts, dtype=np.float32)

        # blocks per grid for the lc reduction and orbital position kernels
        # (lcgen_grid is fixed at construction time - see __init__ - since it
        # no longer depends on the light curve size)
        self.lcsum_grid = (ceil(self.n_pts / self.lcsum_block), 1, 1)
        self.xygen_grid = (ceil(self.n_pts / self.xygen_block), 1, 1)

    def compute_xy_array(self, t0, period, a, inc, ecc, w):
        """
        Solve Kepler's equation on the device to (re)compute the planet's
        on-sky (x,y) position relative to the star, given a set of orbital
        elements. `put_times_array` must have been called first. This is a
        hot-path call, intended to be repeated every MCMC step as orbital
        parameters vary, same as `build_image` for ring geometry.

        Parameters
        ----------
        t0 : float
            Time of inferior conjunction.
        period : float
            Orbital period.
        a : float
            Semi-major axis, in units of stellar radii.
        inc : float
            Orbital inclination, in radians.
        ecc : float
            Orbital eccentricity.
        w : float
            Longitude of periastron, in radians.
        """
        if self.times_array is None:
            raise RuntimeError(
                "Can't compute orbital positions - call put_times_array first"
            )

        _get_xy_kernel.prepared_call(
            self.xygen_grid, (self.xygen_block, 1, 1),
            self.times_array.gpudata, self.x_array.gpudata,
            self.y_array.gpudata,
            np.float64(t0), np.float64(period), np.float64(a),
            np.float64(inc), np.float64(ecc), np.float64(w),
            np.int32(self.n_pts)
        )

    def put_observed_lc(self, flux, flux_error):
        """
        Send an observed light curve to the device for use with on-device
        model likelihood determination.

        Parameters
        ----------
        flux : ndarray
            1D Numpy array of (N,) flux points (normalised to baseline).
        flux_error : ndarray
            1D Numpy array of (N,) flux error points.
        """
        # input verification
        assert flux.shape == flux_error.shape
        assert len(flux.shape) == 1
        # send arrays to device, pre-converted to the same 0-baseline form as
        # lc_accum (model = 1.0 - lc_accum) so chisq_reduce doesn't have to
        # redo this subtraction on every get_loglikelihood call. float32 is
        # used here (chisq_reduce promotes to double before the lc_accum
        # comparison) since offset_flux/flux_error are depth-scale values,
        # not near-1.0 ones - float32's relative precision there is well
        # below the 1ppm floor set by planet_scale - see
        # scripts/test_offset_flux_float32_precision.py
        self.offset_flux = to_gpu(1.0 - flux, np.float32)
        self.flux_error = to_gpu(flux_error, np.float32)

    def build_image(self,
                    inner_ring_radius, outer_ring_radius,
                    ring_optical_depth, gamma):
        """
        Build an exoplanet-plus-ring opacity mask on the gpu.

        Parameters
        ----------
        inner_ring_radius : float
            The inner radius of the ring in units of planet radii.
        outer_ring_radius : float
            The outer radius of the ring in units of planet radii.
        ring_optical_depth : float
            The normal optical depth of the ring.
        gamma : float
            Inclination angle relative to the line of sight to the observer in
            radians. At an angle of 0.0 radians the ring is invisible as ring
            depth is not modelled.
        """
        # sin gamma
        singamma = sin(gamma % (0.5 * pi))

        # verify that the ring fits in the device image array
        if self.planet_scale * outer_ring_radius > self.img_array_shape[1]:
            raise ValueError(
                f"outer_ring_radius={outer_ring_radius} (planet radii) "
                f"exceeds the image array's column extent of "
                f"{self.img_array_shape[1] / self.planet_scale} planet "
                f"radii (img_array_shape[1]={self.img_array_shape[1]}, "
                f"planet_scale={self.planet_scale})"
            )
        if self.planet_scale * outer_ring_radius * singamma \
                >= self.img_array_shape[0]:
            raise ValueError(
                f"outer_ring_radius={outer_ring_radius} (planet radii) at "
                f"gamma={gamma} rad (sin(gamma)={singamma:.4g}) projects to "
                f"a minor-axis extent of "
                f"{outer_ring_radius * singamma} planet radii, which "
                f"exceeds the image array's row extent of "
                f"{self.img_array_shape[0] / self.planet_scale} planet "
                f"radii (img_array_shape[0]={self.img_array_shape[0]}, "
                f"planet_scale={self.planet_scale})"
            )

        # convert normal optical depth to opacity
        if singamma != 0:
            ring_opacity = 1.0 - exp(-ring_optical_depth / singamma)
        else:
            ring_opacity = 1.0

        # the minor axis size of the inner and outer ring ellipses
        i_r_min = singamma * inner_ring_radius
        o_r_min = singamma * outer_ring_radius

        _fill_image_kernel.prepared_call(
            self.imgen_grid, self.imgen_block,
            self.img_array.gpudata,
            np.int32(self.img_array_shape[0]),
            np.int32(self.img_array_shape[1]),
            np.float32(self.pixel_size),
            np.float32(i_r_min), np.float32(inner_ring_radius),
            np.float32(o_r_min), np.float32(outer_ring_radius),
            np.float32(ring_opacity)
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
        img1q = self.img_array.get()
        if not return_full:
            return img1q
        else:
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
        img = self.read_image(return_full=False)

        total_opacity = img.sum()
        planet_opacity = 0.25 * pi * self.planet_scale ** 2

        k = (total_opacity / planet_opacity) ** 0.5

        return k

    def occult_star(self, planet_radius, obliquity, ld_params):
        """
        Produce a transit of the pre-generated opacity profile array across a
        star. The x and y positions of the planet relative to the star must
        already have been sent to the device.

        Parameters
        ----------
        planet_radius : float
            The radius of the planet in units of stellar radii.
        obliquity : float
            The inclination in radians of the planet in the plane of its orbit.
            Runs clockwise, positive from zero parallel to the 'X' axis.
        ld_params : tuple
            Tuple of two quadratic limb darkening parameters for the star

        Returns
        -------
        None. Use `read_lightcurve` to retrieve the model light curve.
        """
        if self.x_array is None or self.y_array is None:
            raise RuntimeError("Can't produce light curve as x_array and/or "
                               "y_array is empty")

        ld_a, ld_b = ld_params
        # obliquity is the same for every thread in this launch, so its
        # cosine/sine are computed once here rather than recomputed by every
        # thread on the device
        c_oblq, s_oblq = cos(obliquity), sin(obliquity)

        # (re)initialise the light curve accumulator
        self.lc_accum = gpuarray.zeros(self.n_pts, dtype=np.float64)

        _pixel_contrib_kernel.prepared_call(
            self.lcgen_grid, self.lcgen_block,
            self.x_array.gpudata, self.y_array.gpudata, np.int32(self.n_pts),
            self.img_array.gpudata,
            np.int32(self.img_array_shape[0]),
            np.int32(self.img_array_shape[1]),
            np.float32(self.pixel_size), np.float32(planet_radius),
            np.float32(c_oblq), np.float32(s_oblq),
            np.float32(ld_a), np.float32(ld_b),
            self.lc_accum.gpudata,
            shared_size=self._lcgen_smem
        )

    def read_lightcurve(self):
        """
        Read the light curve array from the device.

        Returns
        -------
        The model light curve.
        """
        if self.lc_accum is None:
            raise RuntimeError(
                "model lightcurve array is empty, can't copy from device"
            )
        else:
            return 1.0 - self.lc_accum.get()

    def get_loglikelihood(self, jitter=0.0):
        """
        Compute the full Gaussian log-likelihood (residual term plus its
        per-point normalisation) of the lightcurve data given a set of model
        parameters.

        Parameters
        ----------
        jitter : float, optional
            Optional error-inflation term, in the same units as the
            `flux_error` passed to `put_observed_lc`, added in quadrature:
            `sigma_eff^2 = flux_error^2 + jitter^2`. Pass this as a fitted
            MCMC parameter to model unaccounted-for excess scatter; the
            normalisation term has to be recomputed on the device every call
            in that case, since it no longer cancels between steps once it
            depends on a varying parameter. (Default: 0.0, i.e. no inflation.)

        Returns
        -------
        The log-likelihood of the model given the data.
        """
        if self.lc_accum is None:
            raise RuntimeError(
                "model lightcurve array is empty, can't compute log likelihood"
            )
        elif self.offset_flux is None:
            raise RuntimeError(
                "fluxes not provided, can't compute log likelihood"
            )
        elif self.flux_error is None:
            raise RuntimeError(
                "flux errors not provided, can't compute log likelihood"
            )
        else:
            self._chisq = gpuarray.zeros(1, dtype=np.float64)

            _chisq_reduce_kernel.prepared_call(
                self.lcsum_grid, (self.lcsum_block, 1, 1),
                self.lc_accum.gpudata,
                self.offset_flux.gpudata, self.flux_error.gpudata,
                np.float64(jitter), np.int32(self.n_pts), self._chisq.gpudata,
                shared_size=self._lcsum_smem
            )

            chisq = self._chisq.get()[0]

            return -0.5 * chisq

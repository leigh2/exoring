// Copyright (c) 2024 Leigh C. Smith - lsmith@ast.cam.ac.uk
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Four kernels backing ExoRing (see exoring.py):
//   get_xy                   - solves Kepler's equation to turn orbital
//                              elements into an on-sky (x,y) planet trajectory
//   fill_image               - builds the opacity mask for a planet+ring system
//   pixel_contrib_accumulate - projects the opacity mask onto a star to build
//                              a model light curve
//   chisq_reduce             - compares the model light curve to an observed
//                              one
//
// Units: build_image / fill_image work in units of planetary radii.
// get_xy / pixel_contrib_accumulate / chisq_reduce work in units of stellar
// radii.
//
// Quadrant symmetry: the opacity image only ever holds a single quadrant
// (planet+ring is biaxially symmetric), so pixel_contrib_accumulate mirrors
// each pixel position into all four quadrants itself (see the 4 calls to
// mirrored_point_contribution per light curve point) rather than the image
// covering the full disk - this quarters fill_image's work and memory
// footprint for free.
//
// Single ring only: there is no multi-ring support in this backend.

#include <math.h>

// open the dynamic shared memory array (sized per-launch from the host side)
extern __shared__ char sm[];

// warp-level sum reduction via shuffle, no shared memory or syncthreads needed
__device__ double warpShuffleSumd(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Newton-Raphson solve of Kepler's equation E - e*sin(E) = M for the
// eccentric anomaly E, given the mean anomaly M and eccentricity e (see
// Murray & Correia, in Seager's "Exoplanets", section 3, eqn. 5). Ported
// unchanged from the old gpu_transit_toolkit project's get_ds kernel.
__device__ double getE(double M, double e)
{
    double E = M, eps = 1.0e-7;
    double fe, fs;

    while (fmod(fabs(E - e * sin(E) - M), 2.0 * M_PI) > eps)
    {
        fe = fmod(E - e * sin(E) - M, 2.0 * M_PI);
        fs = fmod(1 - e * cos(E), 2.0 * M_PI);
        E = E - fe / fs;
    }
    return E;
}

// Solve for the planet's on-sky (x,y) position relative to the star centre,
// in units of stellar radii, at every light curve time point, given a set of
// orbital elements. One thread per light curve point - O(n_pts) Newton
// iterations to solve Kepler's equation, negligible next to
// pixel_contrib_accumulate's O(pixels x n_pts) cost. This is a port of
// gpu_transit_toolkit's get_ds kernel, extended to emit the sky-plane (x,y)
// decomposition (batman/rsky.c convention: X along the transit-chord
// direction, Y along the impact-parameter direction) instead of just the
// scalar center-to-center separation - exoring's ring projection needs the
// planet's sky-plane direction relative to the ring's tilt axis, not just
// |separation|.
__global__ void get_xy(
    const double * __restrict__ times,  // (n_pts,) light curve observation times
    double * x_array,                    // (n_pts,) output: planet 'X' position
    double * y_array,                    // (n_pts,) output: planet 'Y' position
    const double t0,                      // time of inferior conjunction
    const double per,                      // orbital period
    const double a,                         // semi-major axis, in stellar radii
    const double inc,                        // orbital inclination, radians
    const double ecc,                         // eccentricity
    const double w,                            // longitude of periastron, radians
    const int n_pts                             // number of light curve points
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pts) return;

    double t = times[idx];

    double n = 2.0 * M_PI / per;
    double f = 0.5 * M_PI - w;

    double E;
    double M;

    // the true (f), eccentric (E) and mean (M) anomaly are all equal for a
    // circular orbit
    if (ecc < 1.0e-5)
    {
        M = f;
    }
    else
    {
        E = 2.0 * atan(sqrt((1.0 - ecc) / (1.0 + ecc)) * tan(0.5 * f));
        M = E - ecc * sin(E);
    }

    // time of periastron
    double tp = t0 - 0.5 * per * M / M_PI;

    if (ecc < 1.0e-5)
    {
        f = fmod((t - tp) / per, 1.0) * 2.0 * M_PI;
    }
    else
    {
        M = n * (t - tp);
        E = getE(M, ecc);
        f = 2.0 * atan(sqrt((1.0 + ecc) / (1.0 - ecc)) * tan(0.5 * E));
    }

    double r = a * (1.0 - ecc * ecc) / (1.0 + ecc * cos(f));

    x_array[idx] = -r * cos(w + f);
    y_array[idx] = -r * sin(w + f) * cos(inc);
}

// Signed distance from (x,y) to an axis-aligned ellipse (semi-axes maj, min_),
// via the Sampson approximation F/|grad F| - exact for a circle, first-order
// accurate for an ellipse when the curve's local radius of curvature is large
// relative to a pixel (true here for every inclination down to gamma ~ 0.1-0.15
// rad; only breaks down near the edge-on limit, which is an unmodelled
// degenerate case anyway). Positive outside the ellipse, negative inside.
__device__ inline float ellipse_signed_dist(float x, float y, float maj, float min_) {
    float F = (x / maj) * (x / maj) + (y / min_) * (y / min_) - 1.0f;
    float gx = 2.0f * x / (maj * maj);
    float gy = 2.0f * y / (min_ * min_);
    return F / sqrtf(gx * gx + gy * gy + 1e-30f);  // epsilon guards (0,0)
}

// Convert a signed distance to boundary-curve coverage via a one-pixel-wide
// linear antialiasing ramp: fraction of the pixel lying on the inside
// (dist<0) of the curve, saturating to 0/1 once the curve is more than half
// a pixel away.
__device__ inline float coverage_inside(float dist, float pixsize) {
    return fminf(fmaxf(0.5f - dist / pixsize, 0.0f), 1.0f);
}

// build a planet-plus-ring opacity mask, single quadrant only (biaxial symmetry
// is assumed and exploited by the caller)
__global__ void fill_image(
    float * image,        // the image array (rows x cols, row-major)
    const int rows,        // number of rows in the image array
    const int cols,         // number of columns in the image array
    const float pixsize,    // the pixel size in planetary radii
    const float ir_min,     // minor axis radius of ring inner edge
    const float ir_maj,     // major axis radius of ring inner edge
    const float or_min,     // minor axis radius of ring outer edge
    const float or_maj,     // major axis radius of ring outer edge
    const float op          // ring opacity
){
    // j (column) is tied to the fastest-varying thread dimension so that
    // consecutive threads in a warp touch consecutive (coalesced) addresses
    // in the row-major image array
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= cols) return;

    // pixel centre
    float xc = (j + 0.5f) * pixsize;
    float yc = (i + 0.5f) * pixsize;

    // signed distance to each boundary curve, planet edge first (exact,
    // since the radial distance to a circle *is* the distance to its
    // boundary), then the ring's inner/outer ellipses (Sampson approximation)
    float dist_p = sqrtf(xc * xc + yc * yc) - 1.0f;
    float dist_ir = ellipse_signed_dist(xc, yc, ir_maj, ir_min);
    float dist_or = ellipse_signed_dist(xc, yc, or_maj, or_min);

    // fraction of the pixel inside the planet, and inside the ring annulus
    // (inside the outer ellipse and outside the inner one) - coverage_inside
    // saturates to exactly 0/1 once a pixel is more than half a pixel-width
    // from every curve, so this single expression also reproduces the old
    // fully-interior/fully-exterior fast paths without any branching
    float f_planet = coverage_inside(dist_p, pixsize);
    float f_annulus = coverage_inside(dist_or, pixsize) * (1.0f - coverage_inside(dist_ir, pixsize));

    image[i * cols + j] = f_planet + (1.0f - f_planet) * f_annulus * op;
}

// Blocked-intensity contribution of a single mirrored pixel position. Takes
// (xq,yq) as plain scalar arguments rather than indexing a small array with a
// loop variable - a loop-indexed array forces the compiler to spill it to
// per-thread "local" memory (registers can't be addressed dynamically),
// which shows up as heavy L1TEX-stall time. Calling this 4 times with literal
// arguments lets the compiler keep everything in registers instead.
__device__ inline float mirrored_point_contribution(
    float xq, float yq, float c_oblq, float s_oblq, float prad,
    float x_off, float y_off, float ld_a, float ld_b
){
    float xr = (xq * c_oblq - yq * s_oblq) * prad;
    float yr = (xq * s_oblq + yq * c_oblq) * prad;

    // rad = sqrt(dx^2 + dy^2) is never needed by itself - only rad<=1
    // (equivalent to rad2<=1) and rad^2 (= rad2) are used below, so the sqrt
    // that would compute rad is skipped entirely
    float dx = xr + x_off;
    float dy = yr + y_off;
    float rad2 = dx * dx + dy * dy;
    if (rad2 <= 1.0f) {
        // mu = 1 - cos(asin(rad)); cos(asin(rad)) === sqrt(1 - rad^2)
        // for rad in [0,1], avoiding two transcendental calls for one
        float mu = 1.0f - sqrtf(1.0f - rad2);
        return (1.0f - ld_a * mu - ld_b * mu * mu)
               / (1.0f - ld_a / 3.0f - ld_b / 6.0f) / ((float) M_PI);
    }
    return 0.0f;
}

// Evaluate the fraction of stellar flux blocked by the (precomputed) opacity
// image for every requested planet position, and atomically accumulate each
// thread block's partial sum directly into the per-position light curve
// accumulator. This fuses what used to be two separate kernels (per-pixel
// contribution + light curve sum reduction) and avoids ever materialising a
// (n_pts, num_blocks) intermediate array.
__global__ void pixel_contrib_accumulate(
    const double * xa,      // 'X' position of the planet centre relative to the star, per lc point
    const double * ya,      // 'Y' position of the planet centre relative to the star, per lc point
    const int n_pts,         // number of light curve points
    const float * image,    // opacity image (single quadrant, rows x cols, row-major)
    const int rows,           // number of rows in the image array
    const int cols,            // number of columns in the image array
    const float pxsize,      // the size of each opacity array element in planetary radii
    const float prad,        // the radius of the planet in units of stellar radii
    const float c_oblq,      // cosine of the obliquity (precomputed on the host: it's the
    const float s_oblq,      //   same value for every thread in this launch, no need to
                              //   recompute sincosf() in every one of them)
    const float ld_a,        // quadratic limb darkening parameter a
    const float ld_b,        // quadratic limb darkening parameter b
    double * lc_accum         // (n_pts,) accumulator: sum of blocked intensity (to be atomically filled)
){
    // one thread per opacity-image pixel (not per pixel-per-lightcurve-point):
    // each thread loads its own image[i,j] value exactly once below, then
    // reuses it for every one of the n_pts light curve points in the loop
    // further down. The previous version launched a separate thread (and
    // re-read image[i,j] from scratch) for every (pixel, point) combination,
    // which meant every warp's very first instruction was always a stall on
    // that load - with nothing independent to overlap it with, occupancy
    // couldn't hide the latency no matter how many warps were resident
    // (profiling showed >80% of cycles were exactly this kind of stall).
    // Loading once and looping amortises that one load over up to n_pts
    // iterations of cheap arithmetic instead.
    //
    // j (column) is tied to the fastest-varying thread dimension so that
    // consecutive threads in a warp touch consecutive (coalesced) addresses
    // when reading image[i * cols + j]
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // unlike fill_image, an out-of-bounds thread here can't just `return` -
    // every thread in the block must keep participating in the __syncthreads()
    // barriers inside the loop below, or the in-bounds threads would deadlock
    // waiting at a barrier that the returned threads never reach. Out-of-bounds
    // threads just carry an opacity of 0 instead, so they harmlessly contribute
    // nothing to every iteration's reduction.
    bool valid_pixel = (i < rows && j < cols);
    float opacity = valid_pixel ? image[i * cols + j] : 0.0f;

    // the four mirrored pixel positions relative to the planet centre,
    // running clockwise from the upper right quadrant - fixed for this
    // thread's whole lifetime, computed once rather than once per point
    float x0 = (j + 0.5f) * pxsize;
    float y0 = (i + 0.5f) * pxsize;
    double area2 = (double) (pxsize * prad) * (double) (pxsize * prad);

    int linear_xy = threadIdx.x + threadIdx.y * blockDim.x;
    int lane = linear_xy % 32;
    int warp_id = linear_xy / 32;
    int n_warps = (blockDim.x * blockDim.y + 31) / 32;

    double * warp_sums = (double *) sm;  // size: n_warps doubles, reused every iteration

    for (int k = 0; k < n_pts; k++) {
        double val = 0.0;

        if (opacity > 0.0f) {
            // single precision is plenty for the per-point geometry/limb-darkening
            // math below (this is a photometric integral, not a precision-critical
            // sum) - keeping it out of double precision halves the register cost
            // of every temporary here
            float x_off = (float) xa[k];
            float y_off = (float) ya[k];

            double intensity_sum = 0.0;
            intensity_sum += mirrored_point_contribution(
                 x0,  y0, c_oblq, s_oblq, prad, x_off, y_off, ld_a, ld_b);
            intensity_sum += mirrored_point_contribution(
                 x0, -y0, c_oblq, s_oblq, prad, x_off, y_off, ld_a, ld_b);
            intensity_sum += mirrored_point_contribution(
                -x0, -y0, c_oblq, s_oblq, prad, x_off, y_off, ld_a, ld_b);
            intensity_sum += mirrored_point_contribution(
                -x0,  y0, c_oblq, s_oblq, prad, x_off, y_off, ld_a, ld_b);

            val = (double) opacity * area2 * intensity_sum;
        }

        // warp-shuffle reduction within this block's (i,j) tile for this k
        val = warpShuffleSumd(val);
        if (lane == 0) {
            warp_sums[warp_id] = val;
        }
        // every thread must see all warps' partial sums before warp 0 reads
        // them below
        __syncthreads();

        if (warp_id == 0) {
            double v = (lane < n_warps) ? warp_sums[lane] : 0.0;
            v = warpShuffleSumd(v);
            if (lane == 0) {
                atomicAdd(&lc_accum[k], v);
            }
        }
        // and every thread must wait for that read to finish before the next
        // iteration is allowed to overwrite warp_sums
        __syncthreads();
    }
}

// Reduce (flux - model)^2 / error^2 over all light curve points straight into a
// single chisq scalar via one atomicAdd per block. Fuses what used to be two
// separate kernels (per-block chisq reduction + final atomic-sum reduction).
__global__ void chisq_reduce(
    const double * lc_accum,    // (n_pts,) light curve accumulator (model = 1 - lc_accum)
    const double * flux,         // (n_pts,) observed flux
    const double * flux_error,    // (n_pts,) observed flux error
    const int n_pts,                // number of light curve points
    double * chisq                  // single-element output (to be atomically filled)
){
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (k < n_pts) {
        double model = 1.0 - lc_accum[k];
        double resid = (flux[k] - model) / flux_error[k];
        val = resid * resid;
    }

    val = warpShuffleSumd(val);

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int n_warps = (blockDim.x + 31) / 32;

    double * warp_sums = (double *) sm;  // size: n_warps doubles
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        double v = (lane < n_warps) ? warp_sums[lane] : 0.0;
        v = warpShuffleSumd(v);
        if (lane == 0) {
            atomicAdd(chisq, v);
        }
    }
}

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

// Three kernels backing ExoRing (see exoring.py):
//   fill_image               - builds the opacity mask for a planet+ring system
//   pixel_contrib_accumulate - projects the opacity mask onto a star to build
//                              a model light curve
//   chisq_reduce             - compares the model light curve to an observed
//                              one
//
// Units: build_image / fill_image work in units of planetary radii.
// pixel_contrib_accumulate / chisq_reduce work in units of stellar radii.
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
    const float op,         // ring opacity
    const int ssf,          // super sample factor
    const float ss_gap,     // the size of a super-sample element in planetary radii
    const float ss_cont     // the fractional contribution of a super sample element to its pixel
){
    // j (column) is tied to the fastest-varying thread dimension so that
    // consecutive threads in a warp touch consecutive (coalesced) addresses
    // in the row-major image array
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= cols) return;

    // calculate the positions of the vertices of the pixel
    float x_i = j * pixsize;        // x inner
    float x_o = (j + 1) * pixsize;  // x outer
    float y_i = i * pixsize;        // y inner
    float y_o = (i + 1) * pixsize;  // y outer
    // pixel inner and outer radii
    float px_inner_rad = sqrtf(x_i * x_i + y_i * y_i);
    float px_outer_rad = sqrtf(x_o * x_o + y_o * y_o);

    float value;

    if (px_outer_rad <= 1.0f) {
        // the furthest vertex of the pixel is inside the planet radius, hence
        // the pixel is fully inside the planet and the opacity is total
        value = 1.0f;

    } else if (px_inner_rad >= 1.0f &&
               ((x_i / ir_maj) * (x_i / ir_maj) + (y_i / ir_min) * (y_i / ir_min) >= 1.0f &&
                (x_o / or_maj) * (x_o / or_maj) + (y_o / or_min) * (y_o / or_min) <= 1.0f)) {
        // the pixel is fully inside the ring but also fully outside the planet
        value = op;

    } else if (px_inner_rad >= 1.0f &&
               ((x_o / ir_maj) * (x_o / ir_maj) + (y_o / ir_min) * (y_o / ir_min) <= 1.0f ||
                (x_i / or_maj) * (x_i / or_maj) + (y_i / or_min) * (y_i / or_min) >= 1.0f)) {
        // the pixel is wholly outside the planet and the ring
        value = 0.0f;

    } else {
        // the pixel is partially covered by the planet and/or ring, super-sample
        // it to estimate the appropriate opacity value
        value = 0.0f;
        for (int m = 0; m < ssf; m++) {
            for (int n = 0; n < ssf; n++) {
                float xp = x_i + (0.5f + m) * ss_gap;
                float yp = y_i + (0.5f + n) * ss_gap;

                if (sqrtf(xp * xp + yp * yp) < 1.0f) {
                    // the super sample pixel centre is on the planet
                    value += ss_cont;
                } else if ((xp / ir_maj) * (xp / ir_maj) + (yp / ir_min) * (yp / ir_min) >= 1.0f &&
                           (xp / or_maj) * (xp / or_maj) + (yp / or_min) * (yp / or_min) < 1.0f) {
                    // the super sample pixel centre is on the ring
                    value += op * ss_cont;
                }
            }
        }
    }

    image[i * cols + j] = value;
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

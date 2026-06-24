#!/usr/bin/env python3
"""
Confirm that chisq_reduce's jitter-aware normalisation term (sigma_eff^2 =
flux_error^2 + jitter^2, plus a log(2*pi*sigma_eff^2) per point - see
exoring.cu) stays negligible even at PLATO-scale light curve lengths, not
just at the ~2000-point scale used by the other benchmark scripts.

occult_star (which runs pixel_contrib_accumulate, O(pixels x n_pts)) is run
once to populate lc_accum; only get_loglikelihood (which runs the O(n_pts)
chisq_reduce kernel) is timed in the loop below, isolating the kernel the
jitter term was added to.
"""
import numpy as np
from time import time
import pycuda.driver as drv
from exoring import ExoRing

# ~2 years at 25s cadence
N_PTS = int(2 * 365.25 * 86400 / 25)
N_EVALS = 200

print(f"N_PTS = {N_PTS}")

ring = ExoRing(planet_scale=200, img_array_shape=(512, 1024))
ring.build_image(inner_ring_radius=1.5, outer_ring_radius=2.5,
                  ring_optical_depth=1.0, gamma=0.6)

t = np.linspace(-50.0, 50.0, N_PTS)
x = t
y = np.full_like(x, 0.1)
ring.put_xy_array(x, y)

t0 = time()
ring.occult_star(planet_radius=0.1, obliquity=0.3, ld_params=(0.3, 0.2))
drv.Context.synchronize()
t1 = time()
print(f"one-time occult_star (pixel_contrib_accumulate) cost: "
      f"{1000 * (t1 - t0):.3f} ms")

lc = ring.read_lightcurve()
flux = lc + np.random.default_rng(0).normal(0, 1e-4, size=lc.shape)
flux_error = np.full_like(flux, 1e-4)
ring.put_observed_lc(flux, flux_error)

for jitter, label in [(0.0, "jitter=0.0"), (5e-4, "jitter=5e-4")]:
    # warm up
    for _ in range(5):
        ring.get_loglikelihood(jitter=jitter)

    drv.Context.synchronize()
    t0 = time()
    for _ in range(N_EVALS):
        ll = ring.get_loglikelihood(jitter=jitter)
    drv.Context.synchronize()
    t1 = time()

    print(f"{label}: {N_EVALS} chisq_reduce evals in {t1 - t0:.4f}s "
          f"({1000 * (t1 - t0) / N_EVALS:.5f} ms/eval), last ll={ll:.6f}")

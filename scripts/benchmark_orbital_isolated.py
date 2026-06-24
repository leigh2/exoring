#!/usr/bin/env python3
"""
Isolate the added cost of the get_xy Kepler-solve kernel itself, holding
everything else in the pipeline fixed: both cases use the *same* resulting
(x,y) trajectory and ring geometry per iteration, so occult_star's
data-dependent branching does identical work in both - only the presence
of the on-device orbital-position recompute differs.
"""
import numpy as np
from time import time
import pycuda.driver as drv
from exoring import ExoRing

N_EVALS = 2000

orbit_params = dict(t0=0.0, period=10.0, a=15.0, inc=1.5, ecc=0.1, w=0.3)
times = np.linspace(-0.05, 0.05, 2001)


def _make_ring():
    ring = ExoRing(planet_scale=200, img_array_shape=(512, 1024))
    ring.build_image(inner_ring_radius=1.5, outer_ring_radius=2.5,
                      ring_optical_depth=1.0, gamma=0.6)
    return ring


def _bench(label, use_orbit_kernel):
    ring = _make_ring()
    ring.put_times_array(times)
    ring.compute_xy_array(**orbit_params)
    x_fixed = ring.x_array.get()
    y_fixed = ring.y_array.get()

    if not use_orbit_kernel:
        ring.put_xy_array(x_fixed, y_fixed)

    ring.occult_star(planet_radius=0.1, obliquity=0.3, ld_params=(0.3, 0.2))
    lc = ring.read_lightcurve()
    flux = lc + np.random.default_rng(0).normal(0, 1e-4, size=lc.shape)
    flux_error = np.full_like(flux, 1e-4)
    ring.put_observed_lc(flux, flux_error)

    def _step():
        if use_orbit_kernel:
            ring.compute_xy_array(**orbit_params)
        ring.build_image(inner_ring_radius=1.5, outer_ring_radius=2.5,
                          ring_optical_depth=1.0, gamma=0.6)
        ring.occult_star(planet_radius=0.1, obliquity=0.3, ld_params=(0.3, 0.2))
        return ring.get_loglikelihood()

    for _ in range(10):
        _step()

    drv.Context.synchronize()
    t0 = time()
    for _ in range(N_EVALS):
        ll = _step()
    drv.Context.synchronize()
    t1 = time()

    ms_per_eval = 1000 * (t1 - t0) / N_EVALS
    print(f"{label}: {N_EVALS} evals in {t1 - t0:.3f}s ({ms_per_eval:.4f} ms/eval), "
          f"last ll={ll}")
    return ms_per_eval


ms_fixed = _bench("fixed xy (put_xy_array)", use_orbit_kernel=False)
ms_orbit = _bench("recomputed xy (compute_xy_array)", use_orbit_kernel=True)

print(f"\nadded cost of get_xy kernel: {ms_orbit - ms_fixed:.4f} ms/eval "
      f"({100 * (ms_orbit - ms_fixed) / ms_fixed:.2f}%)")

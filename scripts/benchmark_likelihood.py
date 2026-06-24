#!/usr/bin/env python3
"""
Benchmark repeated build_image -> occult_star -> get_loglikelihood calls,
representative of an MCMC inner loop where ring geometry changes every step
but the observed light curve and orbit positions stay resident on the device.
"""
import numpy as np
from time import time
import pycuda.driver as drv
from exoring import ExoRing

N_EVALS = 2000

ring = ExoRing(planet_scale=200, img_array_shape=(512, 1024))
ring.build_image(inner_ring_radius=1.5, outer_ring_radius=2.5,
                  ring_optical_depth=1.0, gamma=0.6)

t = np.linspace(-0.05, 0.05, 2001)
x = t * 50.0
y = np.full_like(x, 0.1)
ring.put_xy_array(x, y)
ring.occult_star(planet_radius=0.1, obliquity=0.3, ld_params=(0.3, 0.2))
lc = ring.read_lightcurve()
flux = lc + np.random.normal(0, 1e-4, size=lc.shape)
flux_error = np.full_like(flux, 1e-4)
ring.put_observed_lc(flux, flux_error)

# warm up
for _ in range(10):
    ring.build_image(inner_ring_radius=1.5, outer_ring_radius=2.5,
                      ring_optical_depth=1.0, gamma=0.6)
    ring.occult_star(planet_radius=0.1, obliquity=0.3, ld_params=(0.3, 0.2))
    ring.get_loglikelihood()

drv.Context.synchronize()
t0 = time()
for _ in range(N_EVALS):
    ring.build_image(inner_ring_radius=1.5, outer_ring_radius=2.5,
                      ring_optical_depth=1.0, gamma=0.6)
    ring.occult_star(planet_radius=0.1, obliquity=0.3, ld_params=(0.3, 0.2))
    ll = ring.get_loglikelihood()
drv.Context.synchronize()
t1 = time()

print(f"{N_EVALS} full likelihood evals in {t1 - t0:.3f}s "
      f"({1000 * (t1 - t0) / N_EVALS:.4f} ms/eval)")
print("last loglikelihood:", ll)

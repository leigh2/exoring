#!/usr/bin/env python3
import numpy as np
from exoring import ExoRing

ring = ExoRing()

ring.build_image(
    inner_ring_radius=1.5, outer_ring_radius=2.5,
    ring_optical_depth=1.0, gamma=0.6
)

k = ring.get_k()
print("k =", k)
assert np.isfinite(k) and k > 1.0

img = ring.read_image()
print("image shape:", img.shape, "nonzero:", np.count_nonzero(img))
assert img.shape == (1024, 2048)
assert np.count_nonzero(img) > 0

t = np.linspace(-0.05, 0.05, 2001)
x = t * 50.0
y = np.full_like(x, 0.1)
ring.put_xy_array(x, y)

ring.occult_star(planet_radius=0.1, obliquity=0.3, ld_params=(0.3, 0.2))
lc = ring.read_lightcurve()
print("lc shape:", lc.shape, "min:", lc.min(), "max:", lc.max())
assert lc.shape == x.shape
assert lc.min() < 1.0
assert lc.max() <= 1.0001

flux = lc + np.random.normal(0, 1e-4, size=lc.shape)
flux_error = np.full_like(flux, 1e-4)
ring.put_observed_lc(flux.astype(np.float32), flux_error.astype(np.float32))
ll = ring.get_loglikelihood()
print("loglikelihood:", ll)
assert np.isfinite(ll)

print("SMOKE TEST PASSED")

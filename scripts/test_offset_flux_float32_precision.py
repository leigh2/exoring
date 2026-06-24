#!/usr/bin/env python3
"""
Test whether storing offset_flux/flux_error in float32 (instead of the
current float64) loses significant precision, now that chisq_reduce compares
them directly against lc_accum's 0-baseline form (offset_flux = 1 - flux)
rather than against a near-1.0 model value.

This only tests the *storage* precision loss from quantizing to float32 (a
round trip: float64 -> float32 -> float64) - lc_accum and the chisq
arithmetic/reduction itself stay float64 either way (consistent with
img_array already being float32 while pixel_contrib_accumulate reduces in
double), so this isolates whether halving offset_flux/flux_error's memory
footprint is free.
"""
import numpy as np
from exoring import ExoRing

RING_PARAMS = dict(
    inner_ring_radius=1.5, outer_ring_radius=2.5,
    ring_optical_depth=1.0, gamma=0.6
)
OCCULT_PARAMS = dict(
    planet_radius=0.1, obliquity=0.3, ld_params=(0.3, 0.2)
)

t = np.linspace(-0.05, 0.05, 2001)
x = t * 50.0
y = np.full_like(x, 0.1)
flux_error = np.full((x.size,), 1e-4)

ring = ExoRing(planet_scale=200, img_array_shape=(512, 1024))
ring.build_image(**RING_PARAMS)
ring.put_xy_array(x, y)
ring.occult_star(**OCCULT_PARAMS)
model_flux = ring.read_lightcurve()

rng = np.random.default_rng(0)
flux = model_flux + rng.normal(0, 1e-4, size=model_flux.shape)

ring.put_observed_lc(flux, flux_error)
loglik_ref = ring.get_loglikelihood()

# replicate chisq_reduce's arithmetic on the host: offset_flux/lc_accum
# comparison, but with offset_flux and flux_error round-tripped through
# float32 storage first
lc_accum = 1.0 - model_flux
offset_flux = 1.0 - flux

offset_flux_32 = offset_flux.astype(np.float32).astype(np.float64)
flux_error_32 = flux_error.astype(np.float32).astype(np.float64)

resid_ref = (offset_flux - lc_accum) / flux_error
resid_32 = (offset_flux_32 - lc_accum) / flux_error_32

chisq_ref = np.sum(resid_ref ** 2)
chisq_32 = np.sum(resid_32 ** 2)

loglik_32 = -0.5 * chisq_32

print(f"loglikelihood (float64 storage): {loglik_ref:.6f}")
print(f"loglikelihood (float32 storage): {loglik_32:.6f}")
print(f"d(loglikelihood): {loglik_32 - loglik_ref:.3e}")
print(f"max|d(offset_flux)| from float32 round-trip: "
      f"{np.max(np.abs(offset_flux_32 - offset_flux)):.3e}")
print(f"max|d(flux_error)| from float32 round-trip: "
      f"{np.max(np.abs(flux_error_32 - flux_error)):.3e}")
print(f"max|d(resid)|: {np.max(np.abs(resid_32 - resid_ref)):.3e}")

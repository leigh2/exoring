#!/usr/bin/env python3
"""
Establish the impact of planet_scale (opacity-image pixels per planet
radius) on numerical precision of the model light curve and the resulting
log-likelihood. img_array_shape is scaled proportionally at each
planet_scale so the field of view in planet radii - and hence whether the
ring fits - stays fixed; see the build_image bounds checks in exoring.py.

The finest planet_scale in PLANET_SCALES is treated as ground truth.
"""
import numpy as np
from exoring import ExoRing

# pixels per planet radius to sweep, finest last (treated as reference)
PLANET_SCALES = [25, 50, 100, 200, 400, 800]

# field of view in planet radii, fixed across the sweep (matches the
# default img_array_shape=(512, 1024) at planet_scale=200)
FOV_ROWS = 512 / 200
FOV_COLS = 1024 / 200

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


def model_lightcurve(planet_scale):
    img_array_shape = (
        int(np.ceil(FOV_ROWS * planet_scale)),
        int(np.ceil(FOV_COLS * planet_scale)),
    )
    ring = ExoRing(planet_scale=planet_scale, img_array_shape=img_array_shape)
    ring.build_image(**RING_PARAMS)
    ring.put_xy_array(x, y)
    ring.occult_star(**OCCULT_PARAMS)
    return ring.read_lightcurve()


fluxes = {ps: model_lightcurve(ps) for ps in PLANET_SCALES}

ref_scale = PLANET_SCALES[-1]
ref_flux = fluxes[ref_scale]

# log-likelihood of each resolution's model against the reference-resolution
# flux as "observed" data, holding flux_error fixed
logliks = {}
for ps in PLANET_SCALES:
    img_array_shape = (
        int(np.ceil(FOV_ROWS * ps)),
        int(np.ceil(FOV_COLS * ps)),
    )
    ring = ExoRing(planet_scale=ps, img_array_shape=img_array_shape)
    ring.build_image(**RING_PARAMS)
    ring.put_xy_array(x, y)
    ring.put_observed_lc(ref_flux, flux_error)
    ring.occult_star(**OCCULT_PARAMS)
    logliks[ps] = ring.get_loglikelihood()

ref_loglik = logliks[ref_scale]

print(f"{'planet_scale':>12}  {'max|dflux|':>12}  {'rms(dflux)':>12}  "
      f"{'loglikelihood':>15}  {'d(loglik) vs ref':>17}")
for ps in PLANET_SCALES:
    dflux = fluxes[ps] - ref_flux
    max_abs = np.max(np.abs(dflux))
    rms = np.sqrt(np.mean(dflux ** 2))
    print(f"{ps:>12}  {max_abs:>12.3e}  {rms:>12.3e}  "
          f"{logliks[ps]:>15.6f}  {logliks[ps] - ref_loglik:>17.6f}")

# exoring
Exoring transit simulation, GPU-accelerated via pycuda/CUDA C.

![Tests](https://github.com/leigh2/exoring/actions/workflows/tests.yml/badge.svg)

## installation
Requires a CUDA-capable GPU and `pycuda`. Clone the repository, navigate to
it, then run pip install.
```sh
git clone git@github.com:leigh2/exoring.git
cd exoring
pip install .
```

## usage examples
### generate the opacity image of a ringed exoplanet
```python
from exoring import ExoRing
import matplotlib.pyplot as plt

ring = ExoRing(planet_scale=200)
ring.build_image(
    inner_ring_radius=1.5, outer_ring_radius=1.9,
    ring_optical_depth=0.2, gamma=0.35
)
image = ring.read_image()

plt.imshow(image)
plt.show()
```
This builds and shows the opacity image of a planet with a ring of optical
depth 0.2, inner and outer radii of 1.5 and 1.9 planetary radii, and a tilt
of 0.35 radians relative to the line of sight to the observer. `planet_scale`
sets the resolution in pixels per planet radius. The image is built on the
GPU; `read_image()` copies it back to the host, mirroring the single
computed quadrant into the full image by default.

### generate a transit light curve of the ringed exoplanet
Extending the above example, we can generate the transit light curve with:
```python
import numpy as np

x_offsets = np.linspace(-2, 2, 1000)
y_offsets = np.full_like(x_offsets, 0.3)
ring.put_xy_array(x_offsets, y_offsets)
ring.occult_star(
    planet_radius=0.03, obliquity=0.2, ld_params=(0.395, 0.295)
)
light_curve = ring.read_lightcurve()

plt.plot(x_offsets, light_curve)
plt.show()
```
This simulates and shows the transit of the above ringed exoplanet in front
of a star. The planet has 3% of the stellar radius and transits with a
minimum separation of 0.3 stellar radii (the `y_offsets`), sweeping from -2
to 2 stellar radii in `x_offsets`. The tilt of the planet with respect to its
orbital axis (direction of motion) is 0.2 radians. Quadratic limb darkening
parameters are (0.395, 0.295), which are roughly appropriate for the Sun in
the Kepler K band (according to https://exoctk.stsci.edu/limb_darkening).

### fitting a light curve
`ExoRing` also supports computing a log-likelihood against an observed light
curve directly on the GPU, for use in an MCMC inner loop:
```python
ring.put_observed_lc(flux, flux_error)
ring.build_image(inner_ring_radius=1.5, outer_ring_radius=1.9,
                 ring_optical_depth=0.2, gamma=0.35)
ring.occult_star(planet_radius=0.03, obliquity=0.2, ld_params=(0.395, 0.295))
loglikelihood = ring.get_loglikelihood()
```
`put_xy_array` and `put_observed_lc` only need to be called once per run, since
the orbit positions and observed light curve stay resident on the device;
`build_image`/`occult_star`/`get_loglikelihood` are the hot path, called once
per step as the ring geometry changes.

### notes

* `ExoRing` requires a CUDA-capable GPU; there is no CPU fallback.
* Kernels are compiled once per process the first time an `ExoRing` instance
is constructed, so the first construction in a process is slower than
subsequent ones.


## Acknowledgements
LCS acknowledges support from PLATO grant UKSA ST/R004838/1

# exoring
Exoring transit simulation using numerical integration.

![Tests](https://github.com/leigh2/exoring/actions/workflows/tests.yml/badge.svg)

## installation
Simply clone the repository, navigate to it, then run pip install.
```sh
git clone git@github.com:leigh2/exoring.git
cd exoring
pip install .
```

## usage examples
### generate the opacity image of a ringed exoplanet
```python
from exoring import build_exoring_image
import matplotlib.pyplot as plt

image, x_grid, y_grid, area = build_exoring_image(
    200, 1.5, 1.9, 0.2, 0.35, full_output=True
)

plt.imshow(image)
plt.show()
```
This will produce and show the opacity image of a planet with a ring of opacity 
0.2, inner and outer radii of 1.5 and 1.9 planetary radii, and with a tilt of 
0.35 radians relative to the line of sight to the observer. The code will 
generate the image at a resolution of 200 pixels per planet radius, and will 
return the full 2d grid of opacity values and x,y coordinates.

### generate a transit light curve of the ringed exoplanet
Extending the above example, we can generate the transit light curve with:
```python
from exoring import occult_star
import numpy as np

x_offsets = np.linspace(-2, 2, 1000)
light_curve = occult_star(
    image, x_grid, y_grid, area,
    0.03, x_offsets, 0.3, 0.2,
    (0.395, 0.295)
)

plt.plot(x_offsets, light_curve)
plt.show()
```
This will simulate and show the transit of the above ringed exoplanet in front 
of a star. The image is scaled such that the planet has 3% of the stellar 
radius. The planet transits the star with a minimum separation of 0.3 stellar 
radii, in the other dimension it passes with values between -2 and 2 stellar 
radii. The tilt of the planet with respect to it's orbital axis (direction of 
motion) is 0.2 radians. Quadratic limb darkening parameters are (0.395, 0.295), 
which are roughly appropriate for the Sun in the Kepler K band (according to 
https://exoctk.stsci.edu/limb_darkening).

### notes

* Unless you want a pretty silhouette picture of a ringed exoplanet 
there is no need to run `build_exoring_image()` with `full_output=True`, passing
the full 2d image and grid to `occult_star` results in unnecessary computational
expense.
* The code is compiled jit by numba, meaning the first run of each of the above
methods is relatively slow but subsequent executions are significantly faster.


## Acknowledgements
LCS acknowledges support from PLATO grant UKSA ST/R004838/1

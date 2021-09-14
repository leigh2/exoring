exoring
=======

**exoring** simulates transits of ringed exoplanets using numerical integration.

.. image:: https://img.shields.io/badge/Github-leigh2%2Fexoring-blue
    :target: https://github.com/leigh2/exoring
.. image:: https://github.com/leigh2/exoring/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/leigh2/exoring/actions?query=workflow%3ATests


.. toctree::
  :maxdepth: 2
  :caption: API:

  modules


Installation
------------

Simply clone the repository, navigate to it, then run pip install.

.. code-block:: console

   git clone git@github.com:leigh2/exoring.git
   cd exoring
   pip install .

Usage
-----

In use you will generate the opacity image of a ringed exoplanet with
:code:`exoring.build_exoring_image()` and then simulate the transit of this in
front of a star with :code:`exoring.occult_star()`. E.g.:

.. code-block:: python

   import numpy as np
   from exoring import build_exoring_image, occult_star

   # integration and ring parameters
   ngrid = 200  # number of grid points per planet radius
   ring_inner_outer_rad = 1.5, 1.9  # ring radii in units of planet radii
   ring_opacity = 0.2  # 0 -> 1 is fully transparent -> fully opaque
   ring_angle = 0.35  # inclination of ring relative to LOS in radians

   # build the opacity image
   image, x_grid, y_grid, area = build_exoring_image(
       ngrid, *ring_inner_outer_rad, ring_opacity, ring_angle
   )

   # positional offsets of planet relative to star in the direction of orbital
   # motion in units of stellar radii
   x_offsets = np.linspace(-2, 2, 1000)
   # and in orthogonal direction
   y_offset = 0.3

   # planet and star parameters
   planet_radius = 0.03  # units of stellar radii
   ld_params = (0.395, 0.295)  # quadratic limb darkening parameters of star
   # inclination of ring relative to direction of orbital motion
   theta = 0.2  # in radians

   light_curve = occult_star(
       image, x_grid, y_grid, area,
       planet_radius, x_offsets, y_offset, theta, ld_params
   )


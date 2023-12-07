# geomfitty
A python library for fitting 3D geometric shapes

[![Build Status](https://travis-ci.org/mark-boer/geomfitty.svg?branch=master)](https://travis-ci.org/mark-boer/geomfitty)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mark-boer/geomfitty/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

## Geometries
This library performs least square fits on a series of 3D geometries. It will try to minimize the sum of the squares of the distance to these objects. For several geometric shapes this can be done algebraicly, but for most this is performed iteravely using [scipy](https://www.scipy.org/).

It also supports the use of a __weights__ vector to give certain points more importance, it will then minize the following formula, where `w_i` is the weight vector and `d_i` the distance from the geometric shape to point `i`.

<img src="https://render.githubusercontent.com/render/math?math=\Sigma_i w_i * d_i^2">

__The following geometries are currently supported:__

 * __line__: An infinitely long line, parameterized by an anchor_point on the line and a direction.

 * __plane__: An infite plane parameterized by an anchor_point and a plane normal.

 * __sphere__: A sphere defined by the center and a positve radius.

 * __circle__: A circle positioned in 3D. It is given by a center and direction. The direction is the normal of the plane that this circle lies in. The third parameter is the radius

 * __cylinder__: A inifintely long cylinder, parameterized by an anchor_point and direction in the same way as the line is. And a radius of the cylinder.

 * __torus__: Similar to a circle in 3D, but with an extra tube radius. The radius of the circle is called major_radius, the radius of the tube is called minor_radius.


## Examples
For some notebook examples see [fit_example](doc/examples/fit_example.ipynb), [plot_o3d](doc/examples/plot_o3d.ipynb).

Create some random data along a circle:
```
>>> points = np.random.uniform(low=-1, high=1, size=(3, 100))
>>> points[2] /= 10
>>> points[:2] /= np.linalg.norm(points[:2], axis=0) * np.random.uniform(
    low=0.9, high=1.1, size=(100,)
)
>>> points = points.T
```

Perform a least square fit:
```
initial_guess = geom3d.Circle3D([0, 0, 0], [0, 0, 1], 1)
circle = fit3d.circle3D_fit(points, initial_guess=initial_guess)
circle
```

Plot the results using Open3D, if you have that installed:
```
geomfitty.plot.plot([point, circle])
```

![Example of a circle fit](./doc/images/circle3d_fit.PNG)


## Development
First clone the repository
```
git clone git@github.com:mark-boer/geomfitty.git
cd geomfitty
```

Install the package as in editable mode and install the dev requirements.
```
poetry install
```

Run the tests
```
poetry run pytest
poetry run mypy .
```

Run the code formatter
```
poetry run black .
poetry run isort .
```

## Future plans
 - [ ] Add cone
 - [ ] Allow fits to run without initial guess
     - [ ] Cylinder
     - [ ] Circle3D
     - [ ] Torus
 - [ ] Add Coordinate transformations
 - [ ] Add 2D geometries
     - [ ] Line
     - [ ] Circle
     - [ ] Ellipse
- [ ] Add Sphinx documentation

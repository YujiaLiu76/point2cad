import numpy as np
from abc import ABC, abstractmethod

from ._descriptor import Direction, Position, PositiveNumber
from ._util import distance_line_point, distance_plane_point, distance_point_point


class GeometricShape(ABC):
    @abstractmethod
    def distance_to_point(self, point):
        """Calculates the smallest distance from a point to the shape"""

    # @abstractmethod
    # def project_point(self, point):
    # pass


class Line(GeometricShape):
    anchor_point = Position(3)
    direction = Direction(3)

    def __init__(self, anchor_point, direction):
        self.anchor_point = anchor_point
        self.direction = direction

    def __repr__(self):
        return f"Line(anchor_point={self.anchor_point.tolist()}, direction={self.direction.tolist()})"

    def distance_to_point(self, point):
        return distance_line_point(self.anchor_point, self.direction, point)


class Plane(GeometricShape):
    anchor_point = Position(3)
    normal = Direction(3)

    def __init__(self, anchor_point, normal):
        self.anchor_point = anchor_point
        self.normal = normal

    def __repr__(self):
        return f"Plane(anchor_point={self.anchor_point.tolist()}, normal={self.normal.tolist()})"

    def distance_to_point(self, point):
        return np.abs(distance_plane_point(self.anchor_point, self.normal, point))


class Sphere(GeometricShape):
    center = Position(3)
    radius = PositiveNumber()

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __repr__(self):
        return f"Sphere(center={self.center.tolist()}, radius={self.radius})"

    def distance_to_point(self, point):
        return np.abs(distance_point_point(point, self.center) - self.radius)


class Cylinder(Line):
    radius = PositiveNumber()

    def __init__(self, anchor_point, direction, radius):
        super().__init__(anchor_point, direction)
        self.radius = radius

    def __repr__(self):
        return f"Cylinder(anchor_point={self.anchor_point.tolist()}, direction={self.direction.tolist()}, radius={self.radius})"

    def distance_to_point(self, point):
        return np.abs(super().distance_to_point(point) - self.radius)


class Circle3D(GeometricShape):
    center = Position(3)
    direction = Direction(3)
    radius = PositiveNumber()

    def __init__(self, center, direction, radius):
        self.center = center
        self.direction = direction
        self.radius = radius

    def __repr__(self):
        return f"Circle3D(center={self.center.tolist()}, direction={self.direction.tolist()}, radius={self.radius})"

    def distance_to_point(self, point):
        delta_p = point - self.center
        x1 = np.matmul(
            np.expand_dims(np.dot(delta_p, self.direction), axis=-1),
            np.atleast_2d(self.direction),
        )
        x2 = delta_p - x1
        return np.sqrt(
            np.linalg.norm(x1, axis=-1) ** 2
            + (np.linalg.norm(x2, axis=-1) - self.radius) ** 2
        )


class Torus(Circle3D):
    minor_radius = PositiveNumber()

    def __init__(self, center, direction, major_radius, minor_radius):
        super().__init__(center, direction, major_radius)
        self.minor_radius = minor_radius

    def __repr__(self):
        return f"Torus(center={self.center.tolist()}, direction={self.direction.tolist()}, major_radius={self.major_radius}, minor_radius={self.minor_radius})"

    @property
    def major_radius(self):
        return self.radius

    def distance_to_point(self, point):
        return np.abs(super().distance_to_point(point) - self.minor_radius)

import numpy as np
import scipy.spatial  # type: ignore


def vector_equal(v1, v2):
    return v1.shape == v2.shape and np.allclose(
        v1, v2, rtol=1e-12, atol=1e-12, equal_nan=False
    )


def distance_point_point(p1, p2):
    """Calculates the euclidian distance between two points or sets of points
    >>> distance_point_point(np.array([1, 0]), np.array([0, 1]))
    1.4142135623730951
    >>> distance_point_point(np.array([[1, 1], [0, 0]]), np.array([0, 1]))
    array([1., 1.])
    >>> distance_point_point(np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, -3]]))
    array([1., 3.])
    """
    return scipy.spatial.minkowski_distance(p1, p2)


def distance_plane_point(plane_point, plane_normal, point):
    """Calculates the signed distance from a plane to one or more points
    >>> distance_plane_point(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([2, 2, 2]))
    1
    >>> distance_plane_point(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([[2, 2, 2], [2, 2, 3]]))
    array([1, 2])
    """
    assert np.allclose(
        np.linalg.norm(plane_normal), 1.0, rtol=1e-12, atol=1e-12, equal_nan=False
    )
    return np.dot(point - plane_point, plane_normal)


def distance_line_point(line_point, line_direction, point):
    """Calculates the distance from a line to a point
    >>> distance_line_point(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([1, 1, 2]))
    1.4142135623730951
    >>> distance_line_point(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([[1, 0, 1], [0, 2, 3]]))
    array([1., 2.])
    """
    assert np.allclose(
        np.linalg.norm(line_direction), 1.0, rtol=1e-12, atol=1e-12, equal_nan=False
    )
    delta_p = point - line_point
    return distance_point_point(
        delta_p,
        np.matmul(
            np.expand_dims(np.dot(delta_p, line_direction), axis=-1),
            np.atleast_2d(line_direction),
        ),
    )

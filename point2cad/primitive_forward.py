import numpy as np
import torch

from point2cad.fitting_utils import LeastSquares, customsvd
from point2cad.primitive_utils import fitcone, fitcylinder
from point2cad.utils import regular_parameterization, guard_sqrt, rotation_matrix_a_to_b, get_rotation_matrix

EPS = np.finfo(np.float32).eps


class Fit:
    def __init__(self):
        """
        Defines fitting and sampling modules for geometric primitives.
        """
        LS = LeastSquares()
        self.lstsq = LS.lstsq
        self.parameters = {}

    def sample_torus(self, r_major, r_minor, center, axis):
        d_theta = 60
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta

        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1) * r_minor

        circle = np.concatenate([np.zeros((circle.shape[0], 1)), circle], 1)
        circle[:, 1] += r_major

        d_theta = 100
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta
        theta = np.concatenate([theta, np.zeros(1)])

        torus = []
        for i in range(d_theta):
            R = get_rotation_matrix(theta[i])
            torus.append((R @ circle.T).T)

        torus = np.concatenate(torus, 0)
        R = rotation_matrix_a_to_b(np.array([0, 0, 1.0]), axis)
        torus = (R @ torus.T).T
        torus = torus + center
        return torus

    def sample_plane(self, d, n, mean):
        regular_parameters = regular_parameterization(120, 120)
        n = n.reshape(3)
        r1 = np.random.random()
        r2 = np.random.random()
        a = (d - r1 * n[1] - r2 * n[2]) / (n[0] + EPS)
        x = np.array([a, r1, r2]) - d * n

        x = x / np.linalg.norm(x)
        n = n.reshape((1, 3))

        # let us find the perpendicular vector to a lying on the plane
        y = np.cross(x, n)
        y = y / np.linalg.norm(y)

        param = 1 - 2 * np.array(regular_parameters)
        param = param * 0.75

        gridded_points = param[:, 0:1] * x + param[:, 1:2] * y
        gridded_points = gridded_points + mean
        return gridded_points

    def sample_cone_trim(self, c, a, theta, points):
        """
        Trims the cone's height based points. Basically we project
        the points on the axis and retain only the points that are in
        the range.
        """
        if c is None:
            return None, None
        c = c.reshape((3))
        a = a.reshape((3))
        norm_a = np.linalg.norm(a)
        a = a / norm_a
        proj = (points - c.reshape(1, 3)) @ a
        proj_max = np.max(proj) + 0.2 * np.abs(np.max(proj))
        proj_min = np.min(proj) - 0.2 * np.abs(np.min(proj))

        # find one point on the cone
        k = np.dot(c, a)
        x = (k - a[1] - a[2]) / (a[0] + EPS)
        y = 1
        z = 1
        d = np.array([x, y, z])
        p = a * (np.linalg.norm(d)) / (np.sin(theta) + EPS) * np.cos(theta) + d

        # This is a point on the surface
        p = p.reshape((3, 1))

        # Now rotate the vector p around axis a by variable degree
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        points = []
        normals = []
        c = c.reshape((3, 1))
        a = a.reshape((3, 1))
        rel_unit_vector = p - c
        rel_unit_vector = (p - c) / np.linalg.norm(p - c)
        rel_unit_vector_min = rel_unit_vector * (proj_min) / (np.cos(theta) + EPS)
        rel_unit_vector_max = rel_unit_vector * (proj_max) / (np.cos(theta) + EPS)

        for j in range(100):
            # p_ = (p - c) * (0.01) * j
            p_ = (
                rel_unit_vector_min
                + (rel_unit_vector_max - rel_unit_vector_min) * 0.01 * j
            )

            d_points = []
            d_normals = []
            for d in range(50):
                degrees = 2 * np.pi * 0.01 * d * 2
                R = np.eye(3) + np.sin(degrees) * K + (1 - np.cos(degrees)) * K @ K
                rotate_point = R @ p_
                d_points.append(rotate_point + c)
                d_normals.append(
                    rotate_point
                    - np.linalg.norm(rotate_point) / np.cos(theta) * a / norm_a
                )

            # repeat the points to close the circle
            d_points.append(d_points[0])
            d_normals.append(d_normals[0])

            points += d_points
            normals += d_normals

        points = np.stack(points, 0)[:, :, 0]
        normals = np.stack(normals, 0)[:, :, 0]
        normals = normals / (np.expand_dims(np.linalg.norm(normals, axis=1), 1) + EPS)

        # projecting points to the axis to trim the cone along the height.
        proj = (points - c.reshape((1, 3))) @ a
        proj = proj[:, 0]
        indices = np.logical_and(proj < proj_max, proj > proj_min)
        # project points on the axis, remove points that are beyond the limits.
        return points[indices], normals[indices]

    def sample_sphere(self, radius, center, N=1000):
        center = center.reshape((1, 3))
        d_theta = 100
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta
        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1)
        lam = np.linspace(
            -radius + 1e-7, radius - 1e-7, 100
        )  # np.linspace(-1 + 1e-7, 1 - 1e-7, 100)
        radii = np.sqrt(radius**2 - lam**2)  # radius * np.sqrt(1 - lam ** 2)
        circle = np.concatenate([circle] * lam.shape[0], 0)
        spread_radii = np.repeat(radii, d_theta, 0)
        new_circle = circle * spread_radii.reshape((-1, 1))
        height = np.repeat(lam, d_theta, 0)
        points = np.concatenate([new_circle, height.reshape((-1, 1))], 1)
        points = points - np.mean(points, 0)
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)
        points = points + center
        return points, normals

    def sample_cylinder_trim(self, radius, center, axis, points, N=1000):
        """
        :param center: center of size 1 x 3
        :param radius: radius of the cylinder
        :param axis: axis of the cylinder, size 3 x 1
        """
        center = center.reshape((1, 3))
        axis = axis.reshape((3, 1))

        d_theta = 60
        d_height = 100

        R = rotation_matrix_a_to_b(np.array([0, 0, 1]), axis[:, 0])

        # project points on to the axis
        points = points - center

        projection = points @ axis
        arg_min_proj = np.argmin(projection)
        arg_max_proj = np.argmax(projection)

        min_proj = np.squeeze(projection[arg_min_proj]) - 0.1
        max_proj = np.squeeze(projection[arg_max_proj]) + 0.1

        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta

        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1)
        circle = np.concatenate([circle] * 2 * d_height, 0) * radius

        normals = np.concatenate([circle, np.zeros((circle.shape[0], 1))], 1)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        height = np.expand_dims(np.linspace(min_proj, max_proj, 2 * d_height), 1)
        height = np.repeat(height, d_theta, axis=0)
        points = np.concatenate([circle, height], 1)
        points = R @ points.T
        points = points.T + center
        normals = (R @ normals.T).T

        return points, normals

    def fit_plane_torch(self, points, normals, weights, ids=0, show_warning=False):
        """
        Fits plane
        :param points: points with size N x 3
        :param weights: weights with size N x 1
        """
        weights_sum = torch.sum(weights) + EPS

        X = points - torch.sum(weights * points, 0).reshape((1, 3)) / weights_sum

        weighted_X = weights * X
        np_weighted_X = weighted_X.data.cpu().numpy()
        if np.linalg.cond(np_weighted_X) > 1e5:
            if show_warning:
                print("condition number is large in plane!", np.sum(np_weighted_X))
                print(torch.sum(points), torch.sum(weights))

        U, s, V = customsvd(weighted_X)
        a = V[:, -1]
        a = torch.reshape(a, (1, 3))
        d = torch.sum(weights * (a @ points.permute(1, 0)).permute(1, 0)) / weights_sum
        return a, d

    def fit_sphere_torch(self, points, normals, weights, ids=0, show_warning=False):
        N = weights.shape[0]
        sum_weights = torch.sum(weights) + EPS
        A = 2 * (-points + torch.sum(points * weights, 0) / sum_weights)

        dot_points = weights * torch.sum(points * points, 1, keepdim=True)

        normalization = torch.sum(dot_points) / sum_weights

        Y = dot_points - normalization
        Y = Y.reshape((N, 1))
        A = weights * A
        Y = weights * Y

        if np.linalg.cond(A.data.cpu().numpy()) > 1e8:
            if show_warning:
                print("condition number is large in sphere!")

        center = -self.lstsq(A, Y, 0.01).reshape((1, 3))
        radius_square = (
            torch.sum(weights[:, 0] * torch.sum((points - center) ** 2, 1))
            / sum_weights
        )
        radius_square = torch.clamp(radius_square, min=1e-3)
        radius = guard_sqrt(radius_square)
        return center, radius

    def fit_cylinder(self, points, normals, weights, ids=0, show_warning=False):
        w_fit, C_fit, r_fit, fit_err = fitcylinder(points.detach().cpu().numpy())
        return w_fit, C_fit, r_fit

    def fit_cone(self, points, normals, weights, ids=0, show_warning=False):
        c, a, theta, err, failure = fitcone(points.detach().cpu().numpy())
        return c, a, theta, err, failure

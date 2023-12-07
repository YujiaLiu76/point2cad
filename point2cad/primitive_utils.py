import numpy as np
from scipy import optimize
from scipy.optimize import minimize

from dependencies.geomfitty.geomfitty import geom3d
from dependencies.geomfitty.geomfitty._descriptor import Direction, Position
from dependencies.geomfitty.geomfitty._util import distance_line_point
from dependencies.geomfitty.geomfitty.fit3d import _check_input


class Cone(geom3d.GeometricShape):
    vertex = Position(3)
    axis = Direction(3)

    def __init__(self, theta, axis, vertex):
        self.vertex = vertex
        self.axis = axis
        self.theta = theta

    def __repr__(self):
        return f"Cone (vertex={self.vertex}, axis={self.axis}, theta={self.theta}"

    def distance_to_point(self, point):
        a = distance_line_point(self.vertex, self.axis, point)
        k = a * np.tan(self.theta)
        b = k + np.abs(np.dot((point - self.vertex), self.axis))
        l = b * np.sin(self.theta)
        d = a / np.cos(self.theta) - l  # np.abs

        return np.abs(d)


def fitcone(points, weights=None, initial_guess: Cone = None):
    """Fits a cone through a set of points"""
    _check_input(points, weights)
    initial_guesses = [
        Cone(0.0, np.array([1.0, 0, 0]), np.zeros(3)),
        Cone(0.0, np.array([0, 1.0, 0]), np.zeros(3)),
        Cone(0.0, np.array([0, 0, 1.0]), np.zeros(3)),
    ]
    if initial_guesses is None:
        raise NotImplementedError

    def cone_fit_residuals(cone_params, points, weights):
        cone = Cone(cone_params[0], cone_params[1:4], cone_params[4:7])

        distances = cone.distance_to_point(points)

        if weights is None:
            return distances

        return distances * np.sqrt(weights)

    best_fit = None
    best_score = float("inf")
    failure = False

    for initial_guess in initial_guesses:
        x0 = np.concatenate(
            [np.array([initial_guess.theta]), initial_guess.axis, initial_guess.vertex]
        )
        results = optimize.least_squares(
            cone_fit_residuals, x0=x0, args=(points, weights), ftol=1e-10
        )

        if not results.success:
            # return RuntimeError(results.message)
            continue

        if results.fun.sum() < best_score:
            best_score = results.fun.sum()
            best_fit = results

    try:
        apex = best_fit.x[4:7]
        axis = best_fit.x[1:4]
        theta = best_fit.x[0]
        err = best_fit.fun.mean()
    except:
        return None, None, None, None, True

    for iter in range(5):
        # check if the cone is valid
        c = apex.reshape((3))
        a = axis.reshape((3))
        norm_a = np.linalg.norm(a)
        a = a / norm_a
        proj = (points - c.reshape(1, 3)) @ a
        if np.max(proj) * np.min(proj) > 0:
            break
        else:
            r_max = distance_line_point(c, a, points[np.argmax(proj)])
            r_min = distance_line_point(c, a, points[np.argmin(proj)])
            h = np.max(proj) - np.min(proj)
            tan_theta2 = (r_max - r_min) / h
            r0 = distance_line_point(c, a, points[np.argmin(proj**2)])
            if tan_theta2 < 0:
                tan_theta2 = (r_min - r_max) / h
                vertex2 = c + a * (r0 / tan_theta2 + iter * 0.5)
            else:
                vertex2 = c - a * (r0 / tan_theta2 + iter * 0.5)

            initial_guess_2 = Cone(np.arctan(tan_theta2), a, vertex2)
            x0 = np.concatenate(
                [
                    np.array([initial_guess_2.theta]),
                    initial_guess_2.axis,
                    initial_guess_2.vertex,
                ]
            )
            results = optimize.least_squares(
                cone_fit_residuals, x0=x0, args=(points, weights), ftol=1e-10
            )

            if not results.success and iter != 4:
                continue
            if not results.success and iter == 4:
                failure = True
                # print('failure!')

            apex = results.x[4:7]
            axis = results.x[1:4]
            theta = results.x[0]
            err = results.fun.mean()

    return apex, axis, theta, err, failure


def fitcylinder(data, guess_angles=None):
    """Fit a list of data points to a cylinder surface. The algorithm implemented
    here is from David Eberly's paper "Fitting 3D Data with a Cylinder" from
    https://www.geometrictools.com/Documentation/CylinderFitting.pdf
    Arguments:
        data - A list of 3D data points to be fitted.
        guess_angles[0] - Guess of the theta angle of the axis direction
        guess_angles[1] - Guess of the phi angle of the axis direction

    Return:
        Direction of the cylinder axis
        A point on the cylinder axis
        Radius of the cylinder
        Fitting error (G function)
    """

    def direction(theta, phi):
        """Return the direction vector of a cylinder defined
        by the spherical coordinates theta and phi.
        """
        return np.array(
            [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
        )

    def projection_matrix(w):
        """Return the projection matrix  of a direction w."""
        return np.identity(3) - np.dot(np.reshape(w, (3, 1)), np.reshape(w, (1, 3)))

    def skew_matrix(w):
        """Return the skew matrix of a direction w."""
        return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    def calc_A(Ys):
        """Return the matrix A from a list of Y vectors."""
        return sum(np.dot(np.reshape(Y, (3, 1)), np.reshape(Y, (1, 3))) for Y in Ys)

    def calc_A_hat(A, S):
        """Return the A_hat matrix of A given the skew matrix S"""
        return np.dot(S, np.dot(A, np.transpose(S)))

    def preprocess_data(Xs_raw):
        """Translate the center of mass (COM) of the data to the origin.
        Return the prossed data and the shift of the COM"""
        n = len(Xs_raw)
        Xs_raw_mean = sum(X for X in Xs_raw) / n

        return [X - Xs_raw_mean for X in Xs_raw], Xs_raw_mean

    def G(w, Xs):
        """Calculate the G function given a cylinder direction w and a
        list of data points Xs to be fitted."""
        n = len(Xs)
        P = projection_matrix(w)
        Ys = [np.dot(P, X) for X in Xs]
        A = calc_A(Ys)
        A_hat = calc_A_hat(A, skew_matrix(w))

        u = sum(np.dot(Y, Y) for Y in Ys) / n
        v = np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(
            np.dot(A_hat, A)
        )

        return sum((np.dot(Y, Y) - u - 2 * np.dot(Y, v)) ** 2 for Y in Ys)

    def C(w, Xs):
        """Calculate the cylinder center given the cylinder direction and
        a list of data points.
        """
        n = len(Xs)
        P = projection_matrix(w)
        Ys = [np.dot(P, X) for X in Xs]
        A = calc_A(Ys)
        A_hat = calc_A_hat(A, skew_matrix(w))

        return np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(
            np.dot(A_hat, A)
        )

    def r(w, Xs):
        """Calculate the radius given the cylinder direction and a list
        of data points.
        """
        n = len(Xs)
        P = projection_matrix(w)
        c = C(w, Xs)

        return np.sqrt(sum(np.dot(c - X, np.dot(P, c - X)) for X in Xs) / n)

    Xs, t = preprocess_data(data)

    # Set the start points

    start_points = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]
    if guess_angles:
        start_points = guess_angles

    # Fit the cylinder from different start points

    best_fit = None
    best_score = float("inf")

    for sp in start_points:
        fitted = minimize(
            lambda x: G(direction(x[0], x[1]), Xs), sp, method="Powell", tol=1e-6
        )

        if fitted.fun < best_score:
            best_score = fitted.fun
            best_fit = fitted

    w = direction(best_fit.x[0], best_fit.x[1])

    return w, C(w, Xs) + t, r(w, Xs), best_fit.fun

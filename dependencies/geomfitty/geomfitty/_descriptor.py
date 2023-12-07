import numpy as np
from weakref import WeakKeyDictionary

DTYPE = np.float64


class Position:
    def __init__(self, dim: int):
        self.dim = dim
        self._instance_data: WeakKeyDictionary[str, np.ndarray] = WeakKeyDictionary()

    def __get__(self, instance, owner):
        try:
            view = self._instance_data[instance].view()
        except KeyError as e:
            raise AttributeError() from e
        view.flags.writeable = False
        return view

    def __set__(self, instance, value):
        value = np.array(value, dtype=DTYPE, copy=True)  # TODO copy?
        if value.shape != (self.dim,):
            raise ValueError("Could not construct a 3D point")
        self._instance_data[instance] = value


class Direction:
    def __init__(self, dim: int):
        self.dim = dim
        self._instance_data: WeakKeyDictionary[str, np.ndarray] = WeakKeyDictionary()

    def __get__(self, instance, owner):
        try:
            view = self._instance_data[instance].view()
        except KeyError as e:
            raise AttributeError() from e
        view.flags.writeable = False
        return view

    def __set__(self, instance, value):
        value = np.array(value, dtype=DTYPE, copy=True)
        value /= np.linalg.norm(value)
        if value.shape != (self.dim,):
            raise ValueError("Could not construct a 3D point")
        self._instance_data[instance] = value


class PositiveNumber:
    def __init__(self):
        self._instance_data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        try:
            return self._instance_data[instance]
        except KeyError as e:
            raise AttributeError() from e

    def __set__(self, instance, value):
        value = DTYPE(value)
        if value < 0:
            raise ValueError(
                "{} must be initialized with a positive number".format(
                    self.__class__.__name__
                )
            )
        self._instance_data[instance] = value

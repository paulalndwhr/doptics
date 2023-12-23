import numpy as np
from attrs import define
import scipy.integrate as sci
import scipy as sp
from typing import Callable, List, Union
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt


class Triangle:
    """

    """
    points: {'a': ArrayLike(float), 'b': ArrayLike(float), 'c': ArrayLike(float)}


class Mesh:
    points: ArrayLike(float)
    cells: NDArray(dtype=int)  # i want shape (:,3)  -> unspecified rows, three cols


def transform(triangle: Triangle, target: Triangle = {'a': (0, 0), 'b': (1, 0), 'c': (0, 1)}):
    """

    :param triangle:
    :param target:
    :return:
    """
    A_T = np.array(((triangle.points.b[0] - triangle.points.a[0]), (triangle.points.c[0] - triangle.points.a[0])),
                   ((triangle.points.b[1] - triangle.points.a[1]), (triangle.points.c[1] - triangle.points.a[1]))
                   )
    b_T = np.array((triangle.points.a[1]), (triangle.points.a[1]))


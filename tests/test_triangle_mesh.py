import doptics.functions as func
import doptics.target as tg
import doptics.triangle_mesh as tm
import numpy as np
from typing import Callable
from icecream import ic
import pytest
from time import perf_counter


def test_det_2x2():
    """
    asserting that the direct implementation of the 2x2 determinant is close to the np.linalg.det() and that it is faster
    :return:
    """
    a = [np.random.rand(2, 2) for x in range(100)]
    start_dir = perf_counter()
    det_dir = [tm.det_2x2(x) for x in a]
    end_dir = perf_counter()
    det_np = [np.linalg.det(x) for x in a]
    end_np = perf_counter()
    np.testing.assert_allclose(det_dir, det_np)
    assert end_np - end_dir > end_dir - start_dir


def test_inv_2x2():
    """
    asserting that the direct implementation of the 2x2 inverse is close to the np.linalg.inv() and that it is faster.
    Also asserts that an exception is raised appropriately if det(A) = 0 <=> the determinant cannot be calculated.
    :return:
    """
    a = [np.random.rand(2, 2) for x in range(100)]
    start_dir = perf_counter()
    inv_dir = [tm.inv_2x2(x) for x in a]
    end_dir = perf_counter()
    inv_np = [np.linalg.inv(x) for x in a]
    end_np = perf_counter()
    np.testing.assert_allclose(inv_dir, inv_np)
    assert end_np - end_dir > end_dir - start_dir
    # non-invertible
    with pytest.raises(ValueError) as e:
        x = tm.inv_2x2(np.zeros([2,2]))
    assert str(e.value.args[0]) == "det(A) = 0 => Matrix [[0. 0.]\n [0. 0.]] is not invertible"
    assert e.type is ValueError


def test_Triangles():
    """

    :return:
    """


def test_transformation():
    """
    The affine transformation of any triangle to the triangle ((0,0), (1, 0), (0,1)) must work - at the right speed.
    :return:
    """


def test_back_transformation():
    """
    The affine backtransformation must bring the triangle back to the original position - or at least close enough.
    Numpy has a good framework for testing if calculations are "close enough"
    :return:
    """
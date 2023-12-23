import doptics.functions as func
import doptics.target as tg
import doptics.triangle_mesh as tm
import numpy as np
from typing import Callable
from icecream import ic


def test_Traingles():
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
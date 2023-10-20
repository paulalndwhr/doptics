import doptics.functions as func
import doptics.target as tg
import numpy as np
from typing import Callable
from icecream import ic


def test_point_target_construction():
    angle_density = func.uniform
    small_angle = -0.9
    large_angle = -0.2
    midpoint = (12, 2.3)
    
    targ = tg.PointTarget(
        angle_density=angle_density,
        small_angle=small_angle,
        large_angle=large_angle,
        midpoint=midpoint
    )

    y2_density, y1_density, y1_span, y2_span, l1, l2 = func.construct_target_density_intervals_from_angular(
        angle_density=angle_density,
        small_angle=small_angle,
        large_angle=large_angle,
        midpoint=midpoint
    )
    
    # assert targ.y1_density == y1_density
    # assert targ.y2_density == y2_density
    assert np.array_equiv(targ.y1_span, y1_span)
    assert np.array_equiv(targ.y2_span, y2_span)
    assert (targ.y1_span == y1_span).all()
    assert (targ.y2_span == y2_span).all()
    assert targ.l1 == l1
    assert targ.l2 == l2

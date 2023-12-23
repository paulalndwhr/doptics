from attrs import define
from typing import Callable
import numpy as np
import doptics

@define
class Target:
    y1_span: [float, float]
    y2_span: [float, float]
    l1: float
    l2: float
    y1_density: Callable
    y2_density: Callable
    ray_crossing: []


@define
class PointTarget(Target):
    midpoint: [float, float]
    angle_density: Callable

    def __init__(self, angle_density, small_angle, large_angle, midpoint):
        abs_small_angle = small_angle if np.abs(np.sin(small_angle)) < np.abs(np.sin(large_angle)) else large_angle
        abs_large_angle = small_angle if np.abs(np.sin(small_angle)) > np.abs(np.sin(large_angle)) else large_angle
        l1 = round(np.sin(abs_small_angle), 5)
        l2 = round(np.sin(abs_large_angle), 5)
        y1_span, y2_span = doptics.functions.construct_y_spans(small_angle, large_angle)
        if l1 == l2:
            l2 = 1.4 * l2
            y2_span[0] = 1.4 * y2_span[0]
            y2_span[1] = 1.4 * y2_span[1]
        self.y1_span = y1_span + midpoint[0]
        self.y2_span = y2_span + midpoint[0]
        self.l1 = l1 + midpoint[1]
        self.l2 = l2 + midpoint[1]
        self.midpoint = midpoint
        self.angle_density = angle_density

        # create y1_density and y2_density

        def y1_density(y1):
            return (angle_density(np.arccos(y1 / np.linalg.norm(np.array(
                [y1 - midpoint[0], l1 - midpoint[1]], dtype=object)))) /
                    np.linalg.norm(np.array([y1 - midpoint[0], l1 - midpoint[1]], dtype=object))
                    )

        def y2_density(y2):
            return (angle_density(np.arccos(y2 / np.linalg.norm(np.array(
                [y2 - midpoint[0], l2 - midpoint[1]], dtype=object)))) /
                    np.linalg.norm(np.array([y2 - midpoint[0], l2 - midpoint[1]], dtype=object))
                    )
        self.y1_density = y1_density
        self.y2_density = y2_density

@define
class IntervallTarget(Target):

    def __init__(self, y1_span, y2_span, l1, l2, source_density, l_source):
        self.y1_span = y1_span
        self.y2_span = y2_span
        self.l1 = l1
        self.l2 = l2



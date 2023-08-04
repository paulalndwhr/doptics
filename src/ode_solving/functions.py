import scipy as sc
import numpy as np
import scipy as sp
from typing import Callable, List
from numpy.typing import ArrayLike
from main import YL, YR, SIGMA, XL, XR


uniform = lambda x: 1
triangle = lambda x: x
normal_10 = lambda x: np.exp(-0.5*((x-0)/SIGMA)**2) + 0.1
G1 = lambda y1: 1
G2 = lambda y2: 1 - np.abs(y2 - 12) if 11 < y2 < 13 else 0
E = lambda x: 1 / (np.exp(10 * (x - 0.5)) + np.exp(-10 * (x - 0.5)))
normal_target = lambda x: np.exp(-0.5*((x-((YR+YL)*0.5))/SIGMA)**2) + 0.01
normal_source = lambda x: np.exp(-0.5*((x-((XR+XL)*0.5))/SIGMA)**2) + 0.01


def cdf_sampling_source(source_density: Callable, linear_samples: ArrayLike, total_samples: int = 1000):
    """
    transforms a equidistant array into an "equi-probable" array respecting a source_density
    :param source_density: Callable, density function which should be respected
    :param linear_samples: ArrayLike,
    :param total_samples: a parameter which increases precision as it becomes larger at linear runtime cost
    :return: an array of the same length as linear_samples with equi-probable distance between its values
    """
    min_ = linear_samples[0]  # =1
    max_ = linear_samples[-1]  # =2
    # delta = (max_ - min_)/(len(linear_samples)-1)
    delta = (max_ - min_) / (len(linear_samples))
    fine_ladder = np.linspace(min_, max_, total_samples)
    probability_density = lambda x: source_density(x) / (sp.integrate.quad(source_density,
                                                                           linear_samples[0],
                                                                           linear_samples[-1])[0])

    phi_arr = [probability_density(sample)*(1/total_samples) for sample in fine_ladder]
    print(phi_arr)
    print('das war phi_arr')
    print(sum(phi_arr))

    # abusing that Phi_arr is initialized as np.zeros(len(phi_arr)), the following speed-up can be achieved:
    Phi_arr = np.zeros(len(phi_arr))
    for i, elt in enumerate(phi_arr):
        Phi_arr[i] = Phi_arr[i - 1] + elt  # abuses np.zeros()
    Phi_arr = np.roll(Phi_arr, 1)
    Phi_arr[0] = 0
    # return Phi_arr
    print(Phi_arr)

    distr_samples = np.zeros(len(linear_samples))
    inverted_Phi = reversed(Phi_arr)
    quantiles = np.linspace(0, 1, len(linear_samples))
    for i in range(len(distr_samples)):
        # ladder_index = next(x[0] for x in enumerate(inverted_Phi) if x[1] < delta*i)
        # ladder_index = len([x for x in Phi_arr if x <= + i*delta]) - 1
        ladder_index = len([x for x in Phi_arr if x <= + quantiles[i]]) - 1
        distr_samples[i] = fine_ladder[ladder_index]
    distr_samples[-1] = max_
    return distr_samples


def construct_target_density_intervals_from_angular(angle_density: Callable,
                                                    small_angle: float,
                                                    large_angle: float,
                                                    center: List[float] = [10, 10],
                                                    precision: float = 1000) -> (Callable, Callable,
                                                                                 List[float], List[float],
                                                                                 float, float):
    """
    Angles should be negative if light is supposed to hit from below. You must not include 0 inn D for the moment
    :param angle_denstiy: defined on Radians $D = [d_l, d_r] \subsetneq (-\pi, pi)$ with $\lambda(D) < \pi$
    :param small_angle: float, in radians
    :param large_angle: float, in radians
    :param precision: int, number of evaluations of the density
    :return:
    """
    # center = (0, 0)

    # find largest angle and construct levels
    abs_small_angle = small_angle if np.abs(np.sin(small_angle)) < np.abs(np.sin(large_angle)) else large_angle
    abs_large_angle = small_angle if np.abs(np.sin(small_angle)) > np.abs(np.sin(large_angle)) else large_angle
    l1 = round(np.sin(abs_small_angle), 5)
    l2 = round(np.sin(abs_large_angle), 5)
    print(f'l1 = {l1}')
    print(f'l2 = {l2}')
    print(np.cos(small_angle))
    print(np.cos(large_angle))

    y1_span = np.array([np.cos(small_angle), np.cos(large_angle) * np.sin(abs_small_angle)/np.sin(abs_large_angle)])
    y2_span = np.sort(
        np.array([np.cos(large_angle), np.cos(small_angle) * np.sin(abs_large_angle)/np.sin(abs_small_angle)])
    )

    print(y1_span)
    print(y2_span)

    if l1 == l2:
        l2 = 1.4 * l2
        y2_span[0] = 1.4 * y2_span[0]
        y2_span[1] = 1.4 * y2_span[1]

    print(y2_span)

    y1_density = lambda y1: angle_density(np.arccos(y1 / np.linalg.norm(np.array([y1-center[0], l1-center[1]], dtype=object)))) / np.linalg.norm(np.array([y1-center[0], l1-center[1]], dtype=object))
    y2_density = lambda y2: angle_density(np.arccos(y2 / np.linalg.norm(np.array([y2-center[0], l2-center[1]], dtype=object)))) / np.linalg.norm(np.array([y2-center[0], l2-center[1]], dtype=object))
    for i in np.linspace(small_angle, large_angle, 100):
        print(f'y2({i}) = {y2_density(i)}')
    y1_span = y1_span + center[0]
    y2_span = y2_span + center[0]
    for j in np.linspace(y2_span[0], y2_span[1], 100):
        print(f'arg to take arccos from = {j / np.linalg.norm([j - center[0], l2 - center[1]])}')
    print(f'{y1_span[0]}pppppp{y1_density(y1_span[1])}')
    print(f'qqqqqqq{y2_density(y2_span[0])}')
    return y1_density, y2_density, y1_span, y2_span, l1 + center[1], l2 + center[1]


def f(x):
    return 1


def g(x):
    return x**2


def rescaling_target_distribution():
    xl = -10
    xr = 4
    #f is the density on x
    yl = 0
    yr = 3
    integral_x = sc.integrate.quad(f, xl, xr)
    integral_y = sc.integrate.quad(g, yl, yr)
    appropriate_g_factor = integral_x[0] / integral_y[0]
    integral_y1 = appropriate_g_factor * sc.integrate.quad(g, yl, yr)[0]
    print(integral_x[0])
    print(integral_y1)



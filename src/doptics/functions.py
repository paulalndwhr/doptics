import scipy as sc
import numpy as np
import scipy as sp
from typing import Callable, List, Tuple
from numpy.typing import ArrayLike
from icecream import ic

YL = 0
YR = 2
SIGMA = 1
XL = 1
XR = 9


def uniform(x): return 1
def triangle(x): return x
def normal_source(x, xl=XL, xr=XR, sigma=SIGMA): return np.exp(-0.5*((x-((xl+xr)*0.5))/sigma)**2) + 0.01
def normal_target(x, yl=YL, yr=YR, sigma=SIGMA): return np.exp(-0.5*((x-((yl+yr)*0.5))/sigma)**2) + 0.01
def normal_10(x, sigma=SIGMA): return np.exp(-0.5 * ((x - 0) / sigma) ** 2) + 0.1
def G1(y1): return 1
def G2(y2): return 1 - np.abs(y2 - 12) if 11 < y2 < 13 else 0
def E(x): return 1 / (np.exp(10 * (x - 0.5)) + np.exp(-10 * (x - 0.5)))


def normalize(func: Callable, domain: ArrayLike) -> Callable:
    """
    returns the normalized value of a function f evaluated in
    :param func:
    :return:
    """
    return lambda x: func(x) / sp.integrate.quad(func, domain[0], domain[1])[0]


def u_prime(x: float, u: float, l: float, m: Callable):
    """
    
    :param x:
    :param u:
    :param l:
    :param m:
    :return:
    """
    return (m(x) - x) / (((m(x) - x) ** 2 + (-l + u) ** 2) ** .5 - l + u)


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
    # delta = (max_ - min_) / (len(linear_samples))
    fine_ladder = np.linspace(min_, max_, total_samples)

    def probability_density(x): return source_density(x) / (sp.integrate.quad(source_density,
                                                                              linear_samples[0],
                                                                              linear_samples[-1])[0])
    phi_arr = [probability_density(sample)*(1/total_samples) for sample in fine_ladder]
    ic(phi_arr)

    # abusing that Phi_arr is initialized as np.zeros(len(phi_arr)), the following speed-up can be achieved:
    Phi_arr = np.zeros(len(phi_arr))
    for i, elt in enumerate(phi_arr):
        Phi_arr[i] = Phi_arr[i - 1] + elt  # abuses np.zeros()
    Phi_arr = np.roll(Phi_arr, 1)
    Phi_arr[0] = 0
    ic(Phi_arr)

    distr_samples = np.zeros(len(linear_samples))
    # reversed_Phi = reversed(Phi_arr)
    quantiles = np.linspace(0, 1, len(linear_samples))
    for i in range(len(distr_samples)):
        # ladder_index = next(x[0] for x in enumerate(reversed_Phi) if x[1] < delta*i)
        # ladder_index = len([x for x in Phi_arr if x <= + i*delta]) - 1
        ladder_index = len([x for x in Phi_arr if x <= + quantiles[i]]) - 1
        distr_samples[i] = fine_ladder[ladder_index]
    distr_samples[-1] = max_
    return distr_samples


def construct_y_spans(small_angle: float, large_angle: float) -> Tuple[ArrayLike]:
    r"""
    Construct the interval targets for point targets with an angular lighting distribution.
    
    Make sure that the absolute value $\vert small_angle - large_angle \vert < \pi$
    
    :param small_angle: in radians, somewhere between $-\pi$ and $\pi$.
    :param large_angle: in radians, somewhere between $-\pi$ and $\pi$.
    :return: intervals y1_span and y2_span which are hit by the light rays which hit the target
    """
    abs_small_angle = small_angle if np.abs(np.sin(small_angle)) < np.abs(np.sin(large_angle)) else large_angle
    abs_large_angle = small_angle if np.abs(np.sin(small_angle)) > np.abs(np.sin(large_angle)) else large_angle

    y1_span = np.array([np.cos(small_angle), np.cos(large_angle) * np.sin(abs_small_angle) / np.sin(abs_large_angle)])
    y2_span = np.sort(
        np.array([np.cos(large_angle), np.cos(small_angle) * np.sin(abs_large_angle) / np.sin(abs_small_angle)])
    )
    return (y1_span, y2_span)


def construct_target_density_intervals_from_angular(angle_density: Callable,
                                                    small_angle: float,
                                                    large_angle: float,
                                                    midpoint: List[float] = [10, 10],
                                                    precision: float = 1000) -> (Callable, Callable,
                                                                                 List[float], List[float],
                                                                                 float, float):
    r"""
    Angles should be negative if light is supposed to hit from below. You must not include 0 inn D for the moment
    :param angle_density: defined on Radians $D = [d_l, d_r] \subsetneq (-\pi, pi)$ with $\lambda(D) < \pi$
    :param small_angle: float, in radians
    :param large_angle: float, in radians
    :param midpoint: List[float] (length=2), the midpoint 
    :param precision: int, number of evaluations of the density
    :return:
    """
    # midpoint = (0, 0)

    # find largest angle and construct levels
    abs_small_angle = small_angle if np.abs(np.sin(small_angle)) < np.abs(np.sin(large_angle)) else large_angle
    abs_large_angle = small_angle if np.abs(np.sin(small_angle)) > np.abs(np.sin(large_angle)) else large_angle
    l1 = round(np.sin(abs_small_angle), 5)
    l2 = round(np.sin(abs_large_angle), 5)
    ic(f'l1 = {l1}')
    ic(f'l2 = {l2}')
    ic(np.cos(small_angle))
    ic(np.cos(large_angle))

    y1_span, y2_span = construct_y_spans(small_angle, large_angle)

    # y1_span = np.array([np.cos(small_angle), np.cos(large_angle) * np.sin(abs_small_angle)/np.sin(abs_large_angle)])
    # y2_span = np.sort(
    #     np.array([np.cos(large_angle), np.cos(small_angle) * np.sin(abs_large_angle)/np.sin(abs_small_angle)])
    # )

    ic(y1_span)
    ic(y2_span)

    if l1 == l2:
        l2 = 1.4 * l2
        y2_span[0] = 1.4 * y2_span[0]
        y2_span[1] = 1.4 * y2_span[1]

    ic(y2_span)

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

    # for i in np.linspace(small_angle, large_angle, 100):
    #     print(f'y2({i}) = {y2_density(i)}')

    y1_span = y1_span + midpoint[0]
    y2_span = y2_span + midpoint[0]
    for j in np.linspace(y2_span[0], y2_span[1], 100):
        print(f'arg to take arccos from = {j / np.linalg.norm([j - midpoint[0], l2 - midpoint[1]])}')
    print(f'{y1_span[0]}pppppp{y1_density(y1_span[1])}')
    print(f'qqqqqqq{y2_density(y2_span[0])}')
    return y1_density, y2_density, y1_span, y2_span, l1 + midpoint[1], l2 + midpoint[1]


def f(x): return 1


def g(x, mu=0): return (x-mu)**2


def rescaling_target_distribution() -> None:
    xl = -10
    xr = 4
    # f is the density on x
    yl = 0
    yr = 3
    integral_x = sc.integrate.quad(f, xl, xr)
    integral_y = sc.integrate.quad(g, yl, yr)
    appropriate_g_factor = integral_x[0] / integral_y[0]
    integral_y1 = appropriate_g_factor * sc.integrate.quad(g, yl, yr)[0]
    print(integral_x[0])
    print(integral_y1)

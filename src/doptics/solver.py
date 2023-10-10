import scipy.integrate as sci
import scipy as sp
from typing import Callable, List, Union
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import numpy as np
# from main import FILENAME


def solve_single_mirror_parallel_source(starting_distribution: Callable, target_distribution: Callable,
                                        x_span: Union[ArrayLike, List[float]], y_span: Union[ArrayLike, List[float]],
                                        u0: float, L: float,
                                        number_rays=15):
    """

    :param starting_distribution: Callable
    :param target_distribution: Callable
    :param x_span: ArrayLike of type float
    :param y_span: ArrayLike of type float
    :param u0: float, values as specified in https://doi.org/10.25368/2023.148
    :param L: float, values as specified in https://doi.org/10.25368/2023.148
    :param number_rays:
    :return:
    """
    xs = np.linspace(x_span[0], x_span[1], number_rays)
    for result_type in [1, -1]:
        a = sp.integrate.quad(starting_distribution, x_span[0], x_span[1])[0] / \
            sp.integrate.quad(target_distribution, y_span[0], y_span[1])[0]
        target_denisty_fixed = lambda y: a * target_distribution(y)

        m_prime = lambda x, m: result_type * starting_distribution(x) / target_denisty_fixed(m)

        y0 = y_span[0] if result_type == 1 else y_span[1]

        m_solved = sp.integrate.solve_ivp(m_prime, x_span, [y0], t_eval=xs, dense_output=True)
        m = m_solved.sol

        u_prime = lambda x, u: (m(x) - x) / (((m(x) - x)**2 + (-L + u)**2)**.5 - L + u)
        u_solved = sp.integrate.solve_ivp(u_prime, x_span, [u0], t_eval=xs).y.reshape(-1)

        plt.clf()
        plt.plot(xs, u_solved, "k")
        for i in range(len(xs)):
            # plt.plot([xs[i], xs[i], m_solved[i]], [0, u_solved[i], +L])
            plt.plot([xs[i], xs[i], m_solved.y.reshape(-1)[i]], [0, u_solved[i], +L])
            plt.savefig(f'solution-mirror-{result_type}.png')
        # plt.savefig(f'solution-mirror-{FILENAME[result_type]}.png')


def f(x: float, u: float) -> float:
    return u**2


def solve():
    x_arr = np.linspace(2, 5, num=50)
    x0 = 0
    y0 = 2
    ret = sci.odeint(f, y0, x_arr)
    return ret

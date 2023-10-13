import scipy.integrate as sci
import scipy as sp
from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
# from main import FILENAME
BEAMPROPERTIES = {'preserve': {'sign': 1, 'index': 0}, 'cross': {'sign': -1, 'index': -1}}


def solve_single_mirror_parallel_source(starting_distribution: Callable, target_distribution: Callable,
                                        x_span: List[float], y_span: List[float],
                                        u0: float, l: float,
                                        number_rays=15):
    xs = np.linspace(x_span[0], x_span[1], number_rays)
    # for result_type in [1, -1]:
    for result_type in ['preserve', 'cross']:
        a = sp.integrate.quad(starting_distribution, x_span[0], x_span[1])[0] / \
            sp.integrate.quad(target_distribution, y_span[0], y_span[1])[0]
        target_density_rescaled = lambda y: a * target_distribution(y)


        m_prime = lambda x, m: BEAMPROPERTIES[result_type]['sign'] * starting_distribution(x) / target_density_rescaled(m)

        # y0 = y_span[0] if result_type == 1 else y_span[1]
        y0 = y_span[BEAMPROPERTIES[result_type]['index']]

        m_solved = sp.integrate.solve_ivp(m_prime, x_span, [y0], t_eval=xs, dense_output=True)
        m = m_solved.sol

        u_prime = lambda x, u: (m(x) - x) / (((m(x) - x)**2 + (-l + u)**2)**.5 - l + u)

        def u_prime(x, u):
            return (m(x) - x) / (((m(x) - x)**2 + (-l + u)**2)**.5 - l + u)

        u_solved = sp.integrate.solve_ivp(u_prime, x_span, [u0], t_eval=xs).y.reshape(-1)

        plt.clf()
        plt.plot(xs, u_solved, "k")
        for i in range(len(xs)):
            plt.plot([xs[i], xs[i], m_solved.y.reshape(-1)[i]], [0, u_solved[i], +l])
        plt.savefig(f'solution-mirror-{result_type}.png')

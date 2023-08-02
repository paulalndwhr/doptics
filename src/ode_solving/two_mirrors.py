import scipy.integrate as sci
import scipy as sp
from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np
from src.ode_solving.symbolic import u_prime_mv


# from main import FILENAME
BEAMPROPERTIES = {'preserve': {'sign': 1, 'index': 0}, 'cross': {'sign': -1, 'index': -1}}


def solve_two_mirrors_parallel_source_two_targets(starting_distribution: Callable, target_distribution_1: Callable,
                                                  target_distribution_2, x_span: List[float], y1_span: List[float],
                                                  y2_span: List[float],
                                                  u0: float, w0: float, l1: float, l2: float,
                                                  number_rays=15):

    x_discrete = np.linspace(x_span[0], x_span[1], number_rays)
    for result_type in [['preserve', 'cross'], ['preserve', 'preserve'], ['cross', 'preserve'], ['cross', 'cross']]:
        print(f'result_type {result_type}')
        a1 = sp.integrate.quad(starting_distribution, x_span[0], x_span[1])[0] / \
             sp.integrate.quad(target_distribution_1, y1_span[0], y1_span[1])[0]
        a2 = sp.integrate.quad(starting_distribution, x_span[0], x_span[1])[0] / \
             sp.integrate.quad(target_distribution_1, y2_span[0], y2_span[1])[0]
        target_density_1_rescaled = lambda y: a1 * target_distribution_1(y)
        target_density_2_rescaled = lambda y: a2 * target_distribution_2(y)

        m1_prime = lambda x, m: BEAMPROPERTIES[result_type[0]]['sign'] \
                                * starting_distribution(x) / target_density_1_rescaled(m)
        m2_prime = lambda x, m: BEAMPROPERTIES[result_type[1]]['sign'] \
                                * starting_distribution(x) / target_density_1_rescaled(m)

        # ==========================================
        """
        code adapted until here
        """

        y1_0 = y1_span[BEAMPROPERTIES[result_type[0]]['index']]
        m1_solved = sp.integrate.solve_ivp(m1_prime, x_span, [y1_0], dense_output=1)
        m1 = lambda x: m1_solved.sol(x)[0]

        y2_0 = y2_span[BEAMPROPERTIES[result_type[1]]['index']]
        m2_solved = sp.integrate.solve_ivp(m2_prime, y1_span, [y2_0], dense_output=1)
        m2 = lambda y1: m2_solved.sol(y1)[0]

        t = lambda y1: np.array([m2(y1) - y1, l2 - l1])
        t_hat = lambda y1: t(y1) / np.linalg.norm(t(y1))

        V0 = u0 + w0 + np.linalg.norm(np.array([m1(x_span[0]), l1]) - t_hat(m1(x_span[0])) * w0)

        V_y1 = lambda y1, _: t_hat(y1)[0]
        V_solved = sp.integrate.solve_ivp(V_y1, y1_span, [V0], dense_output=1)
        V = lambda y1: V_solved.sol(y1)[0]

        up, w = u_prime_mv()

        u_prime = lambda x, u: up(x, m1(x), u, V(m1(x)), t_hat(m1(x))[0], t_hat(m1(x))[1], l1)
        u_solved = sp.integrate.solve_ivp(u_prime, x_span, [u0], t_eval=x_discrete, dense_output=1)
        u_discrete = u_solved.y.reshape(-1)

        B1 = np.zeros(number_rays)
        B2 = np.zeros(number_rays)

        A = np.empty((number_rays, 2))
        B = np.empty((number_rays, 2))

        for i in range(number_rays):
            wi = w(x_discrete[i], m1(x_discrete[i]), u_discrete[i], V(m1(x_discrete[i])), t_hat(m1(x_discrete[i]))[0],
                   t_hat(m1(x_discrete[i]))[1], l1)
            B1[i] = m1(x_discrete[i]) - t_hat(m1(x_discrete[i]))[0] * wi
            B2[i] = l1 - t_hat(m1(x_discrete[i]))[1] * wi

        # For ray tracing
        # rtn = 1000
        # Ar = np.empty((rtn, 2))
        # Br = np.empty((rtn, 2))
        # Ar[:, 0] = np.linspace(x_span[0], x_span[1], rtn)
        # for i in range(rtn):
        #     Ar[i, 1] = u_solved.sol(Ar[i, 0])[0]
        #     wi = w(Ar[i, 0], m1(Ar[i, 0]), Ar[i, 1], V(m1(Ar[i, 0])), t_hat(m1(Ar[i, 0]))[0], t_hat(m1(Ar[i, 0]))[1], l1)
        #     Br[i, 0] = m1(Ar[i, 0]) - t_hat(m1(Ar[i, 0]))[0] * wi
        #     Br[i, 1] = l1 - t_hat(m1(Ar[i, 0]))[1] * wi
        #
        # # np.savez("result.npz", A=Ar, B=Br, l1=l1, l2=l2, x_span=x_span, y1_span=y1_span, y2_span=y2_span)

        # Color test
        # """
        xis = np.arange(0, 1, 0.02)
        Es = starting_distribution(xis)
        # col = Es / np.max(Es) * 0.2
        col = np.ones(xis.shape)
        """
        xis = cdf_sampling_source(E, np.arange(0, 1, 0.05))
        col = np.ones(xis.shape)
        #"""

        for i, xi in enumerate(xis):
            y1i = m1(xi)
            ui = u_solved.sol(xi)[0]
            t1i = t_hat(y1i)[0]
            t2i = t_hat(y1i)[1]
            wi = w(xi, y1i, ui, V(y1i), t1i, t2i, l1)
            y2i = m2(y1i)

            plt.plot([xi, xi, y1i - wi * t1i, y2i],
                     [0, ui, l1 - wi * t2i, l2],
                     "r", alpha=col[i], linewidth=1.5)

        plt.plot(x_discrete, u_discrete, "g")
        plt.plot(B1, B2, "g")
        plt.plot(x_span, [0, 0], "k")
        plt.plot(y1_span, [l1, l1], "k")
        plt.plot(y2_span, [l2, l2], "k")

        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    return None



# xs = np.linspace(x_span[0], x_span[1], number_rays)
# # for result_type in [1, -1]:
# for result_type in ['preserve', 'cross']:
#     a = sp.integrate.quad(starting_distribution, x_span[0], x_span[1])[0] / \
#         sp.integrate.quad(target_distribution, y_span[0], y_span[1])[0]
#     target_density_rescaled = lambda y: a * target_distribution(y)
# 
#     m_prime = lambda x, m: BEAMPROPERTIES[result_type]['sign'] * starting_distribution(x) / target_density_rescaled(m)
# 
#     # y0 = y_span[0] if result_type == 1 else y_span[1]
#     y0 = y_span[BEAMPROPERTIES[result_type]['index']]
# 
#     m_solved = sp.integrate.solve_ivp(m_prime, x_span, [y0], t_eval=xs, dense_output=True)
#     m = m_solved.sol
# 
#     u_prime = lambda x, u: (m(x) - x) / (((m(x) - x)**2 + (-l + u)**2)**.5 - l + u)
#     u_solved = sp.integrate.solve_ivp(u_prime, x_span, [u0], t_eval=xs).y.reshape(-1)
# 
#     plt.clf()
#     plt.plot(xs, u_solved, "k")
#     for i in range(len(xs)):
#         plt.plot([xs[i], xs[i], m_solved.y.reshape(-1)[i]], [0, u_solved[i], +l])
#     plt.savefig(f'solution-mirror-{result_type}.png')


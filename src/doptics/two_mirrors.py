import scipy.integrate as sci
import scipy as sp
from typing import Callable, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from doptics.symbolic import u_prime_mv
from doptics.functions import cdf_sampling_source
import doptics.functions as func


# from main import FILENAME
BEAMPROPERTIES = {'preserve': {'sign': 1, 'index': 0}, 'cross': {'sign': -1, 'index': -1}}
COLORS = {'szegedblue': '#3f3fa6'}


def solve_two_mirrors_parallel_source_two_targets(starting_density: Callable, target_distribution_1: Callable,
                                                  target_distribution_2, x_span: List[float], y1_span: List[float],
                                                  y2_span: List[float],
                                                  u0: float, w0: float, l1: float, l2: float,
                                                  number_rays=15,
                                                  color: str = 'szegedblue') -> List[dict]:
    """

    :param starting_density:
    :param target_distribution_1:
    :param target_distribution_2:
    :param x_span:
    :param y1_span:
    :param y2_span:
    :param u0:
    :param w0:
    :param l1:
    :param l2:
    :param number_rays:
    :param color:
    :return:
    """
    x_discrete = np.linspace(x_span[0], x_span[1], number_rays)
    ax = sp.integrate.quad(starting_density, x_span[0], x_span[1])[0]
    def starting_density_rescaled(x): return 1 / ax * starting_density(x)

    res = []

    for result_type in [['preserve', 'cross'], ['preserve', 'preserve'], ['cross', 'preserve'], ['cross', 'cross']]:
        print(f'result_type {result_type}')
        a1 = 1 / sp.integrate.quad(target_distribution_1, y1_span[0], y1_span[1])[0]
        a2 = 1 / sp.integrate.quad(target_distribution_1, y2_span[0], y2_span[1])[0]
        target_density_1_rescaled = lambda y: a1 * target_distribution_1(y)
        target_density_2_rescaled = lambda y: a2 * target_distribution_2(y)

        m1_prime = lambda x, m: BEAMPROPERTIES[result_type[0]]['sign'] \
                                * starting_density_rescaled(x) / target_density_1_rescaled(m)
        m2_prime = lambda x, m: BEAMPROPERTIES[result_type[1]]['sign'] \
                                * target_density_1_rescaled(x) / target_density_2_rescaled(m)

        # ==========================================

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

        for i in range(number_rays):
            wi = w(x_discrete[i], m1(x_discrete[i]), u_discrete[i], V(m1(x_discrete[i])), t_hat(m1(x_discrete[i]))[0],
                   t_hat(m1(x_discrete[i]))[1], l1)
            B1[i] = m1(x_discrete[i]) - t_hat(m1(x_discrete[i]))[0] * wi
            B2[i] = l1 - t_hat(m1(x_discrete[i]))[1] * wi

        xis = np.arange(0, 1, 0.02)
        xis = cdf_sampling_source(starting_density_rescaled, xis)

        # Color ray with preset from COLORS dictionary if the key exists, else try color string as a color
        try:
            raycolor = COLORS[color]
        except KeyError:
            raycolor = color

        for i, xi in enumerate(xis):
            y1i = m1(xi)
            ui = u_solved.sol(xi)[0]
            t1i = t_hat(y1i)[0]
            t2i = t_hat(y1i)[1]
            wi = w(xi, y1i, ui, V(y1i), t1i, t2i, l1)
            y2i = m2(y1i)

            plt.plot([xi, xi, y1i - wi * t1i, y2i],
                     [0, ui, l1 - wi * t2i, l2],
                     raycolor,
                     # alpha=col[i],
                     linewidth=1.5)

        # better plotting of the reflectors
        precision = 1000
        x_arr_u = np.linspace(x_span[0], x_span[1], precision)
        spline_u = sp.interpolate.CubicSpline(x_discrete, u_discrete)
        plt.plot(x_arr_u, spline_u(x_arr_u), "r")
        plt.plot(x_arr_u, u_solved.sol(x_arr_u)[0], "k")
        if BEAMPROPERTIES[result_type[0]]['sign'] == -1:  # * BEAMPROPERTIES[result_type[1]]['sign'] == -1:
            B1 = np.flipud(B1)
            B2 = np.flipud(B2)
        plt.plot(B1, B2, "g")

        x_arr_w = np.linspace(B1.min(), B1.max(), precision)
        spline_w = sp.interpolate.CubicSpline(B1, B2)
        plt.plot(x_arr_w, spline_w(x_arr_w), "r")

        # plt.plot(x_discrete, u_discrete, "g")
        # plt.plot(B1, B2, "g")
        plt.plot(x_span, [0, 0], "k")
        plt.plot(y1_span, [l1, l1], "k")
        plt.plot(y2_span, [l2, l2], "k")

        plt.axis("equal")
        plt.tight_layout()
        plt.show()

        res.append({'solution_type': result_type,
                    'u': u_solved.sol, 'domain_u': (x_arr_u[0], x_arr_u[-1]), 'u_spline': spline_u,
                    # 'w': w_solved.sol,
                    'domain_w': (x_arr_w[0], x_arr_w[-1]), 'w_spline': spline_w}
                   )

    return res


def solve_two_mirrors_parallel_source_point_target(starting_density: Callable, target_distribution_1: Callable,
                                                   target_distribution_2,
                                                   x_span: Union[List[float], Tuple[float, float]],
                                                   # y1_span: List[float],
                                                   # y2_span: List[float],
                                                   u0: float, w0: float, l1: float, l2: float,
                                                   number_rays=15,
                                                   color: str = 'szegedblue') -> List[dict]:
    def E(x): return np.exp(-0.5 * ((x - ((1 + 0) * 0.5)) / 0.25) ** 2) + 0.01
    G2, G1, y2_span, y1_span, l2, l1 = func.construct_target_density_intervals_from_angular(
        angle_density=starting_density,
        small_angle=-0.7 * np.pi,
        large_angle=-0.3 * np.pi)
    # new stuff ends
    print('back in main program')
    print(G1(y1_span[1]))
    x_discrete = np.linspace(x_span[0], x_span[1], number_rays)

    BEAMPROPERTIES = {'preserve': {'sign': 1, 'index': 0}, 'cross': {'sign': -1, 'index': -1}}
    for result_type in [['preserve', 'preserve'], ['cross', 'preserve']]:  # ['preserve', 'cross'], ['cross', 'cross']]:
        print(f'result_type {result_type}')

        G1_int = sp.integrate.quad(G1, y1_span[0], y1_span[1])[0]
        print(f'G1_int = {G1_int}')
        G2_int = sp.integrate.quad(G2, y2_span[0], y2_span[1])[0]
        print(f'G2_int = {G2_int}')
        a2 = G1_int / sp.integrate.quad(G2, y2_span[0], y2_span[1])[0]
        print(f'calculated a2 as {a2}')
        G2_fixed = lambda y2: a2 * G2(y2)
        a1 = G1_int / sp.integrate.quad(E, x_span[0], x_span[1])[0]
        E_fixed = lambda x: a1 * E(x)

        # pm1 = - 1  # +-1
        # pm2 = 1  # +-1
        m1_prime = lambda x, m1: E_fixed(x) / G1(m1) * BEAMPROPERTIES[result_type[0]]['sign']
        m2_prime = lambda y, m2: G1(y) / G2_fixed(m2) * BEAMPROPERTIES[result_type[1]]['sign']

        # y1_0 = y1_span[0] if pm1 == 1 else y1_span[1]
        y1_0 = y1_span[BEAMPROPERTIES[result_type[0]]['index']]
        m1_solved = sp.integrate.solve_ivp(m1_prime, x_span, [y1_0], dense_output=1)
        m1 = lambda x: m1_solved.sol(x)[0]

        # y2_0 = y2_span[0] if pm2 == 1 else y2_span[1]
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
        # Ar[:, 0] = np.linspace(x_span[0], x_span[1], rtnumber_rays)
        # for i in range(rtnumber_rays):
        #     Ar[i, 1] = u_solved.sol(Ar[i, 0])[0]
        #     wi = w(Ar[i, 0], m1(Ar[i, 0]), Ar[i, 1], V(m1(Ar[i, 0])), t_hat(m1(Ar[i, 0]))[0], t_hat(m1(Ar[i, 0]))[1], l1)
        #     Br[i, 0] = m1(Ar[i, 0]) - t_hat(m1(Ar[i, 0]))[0] * wi
        #     Br[i, 1] = l1 - t_hat(m1(Ar[i, 0]))[1] * wi
        #
        # # np.savez("result.npz", A=Ar, B=Br, l1=l1, l2=l2, x_span=x_span, y1_span=y1_span, y2_span=y2_spanumber_rays)

        # Color test
        # """
        xis = np.arange(0, 1, 0.02)
        xis = func.cdf_sampling_source(E, xis)
        print(xis)
        print('das waren die xis')
        # Es = E(xis)
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
                     "r", alpha=0.6, linewidth=1.5)
            plt.plot([y2i, y2i - .2 * (y2i - y1i - wi * t1i)], [l2, l2 - .2 * (l2 - l1 - wi * t2i)], 'g', linewidth=1.5)
        plt.plot([10 for i in range(100)], np.linspace(0, 12, 100))
        plt.plot(np.linspace(0, 12, 100), [10 for i in range(100)])

        plt.plot(x_discrete, u_discrete, "g")
        # better plotting of the reflectors using splines
        precision = 1000
        x_arr_u = np.linspace(x_span[0], x_span[1], precision)
        spline_u = sp.interpolate.CubicSpline(x_discrete, u_discrete)
        plt.plot(x_arr_u, spline_u(x_arr_u), "r")

        if BEAMPROPERTIES[result_type[0]]['sign'] * BEAMPROPERTIES[result_type[1]]['sign'] == -1:
            B1 = np.flipud(B1)
            B2 = np.flipud(B2)
        plt.plot(B1, B2, "g")

        x_arr_w = np.linspace(B1.min(), B1.max(), precision)
        spline_w = sp.interpolate.CubicSpline(B1, B2)
        plt.plot(x_arr_w, spline_w(x_arr_w), "r")

        plt.plot(x_span, [0, 0], "k")
        plt.plot(y1_span, [l1, l1], "k")
        plt.plot(y2_span, [l2, l2], "k")

        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    return None

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from src.ode_solving.symbolic import u_prime_mv
import src.ode_solving.functions as func

import src.ode_solving.two_mirrors as tms
from src.ode_solving.functions import cdf_sampling_source

# Enable ctrl+c
# import signal
# signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == '__main__':
    l1 = 5
    l2 = 10
    u0 = 3
    w0 = 5

    y1_span = (6, 8)
    y2_span = (11.1, 12.9)
    x_span = (0, 1)

    n = 20

    # G2 = lambda y2: 1 - np.abs(y2 - sum(y2_span)*0.5) if y2_span[0] < y2 <= y2_span[1] + 0.0001 else 0.0001
    # this function gives cursed outputs!
    # tms.solve_two_mirrors_parallel_source_two_targets(starting_density=func.E, target_distribution_1=func.G1,
    #                                                   target_distribution_2=func.G1, x_span=x_span, y1_span=y1_span,
    #                                                   y2_span=y2_span,
    #                                                   u0=u0, w0=w0, l1=l1, l2=l2,
    #                                                   # color='#a69f3f',
    #                                                   number_rays=15
    #                                                   )

    # G1 = lambda y1: 1
    # G2 = lambda y2: 1 - np.abs(y2 - 12) if 11 < y2 < 13 else 0
    # E = lambda x: 1 / (np.exp(10 * (x - 0.5)) + np.exp(-10 * (x - 0.5)))

    # new stuff
    # E = func.uniform
    E = lambda x: np.exp(-0.5*((x-((1+0)*0.5))/0.25)**2) + 0.01
    G2, G1, y2_span, y1_span, l2, l1 = func.construct_target_density_intervals_from_angular(angle_density=func.normal_10,  #lambda x: x,
                                                                                            small_angle=-0.7 * np.pi,
                                                                                            large_angle=-0.3 * np.pi)
    # new stuff ends
    print(f'back in main program')
    print(G1(y1_span[1]))
    x_discrete = np.linspace(x_span[0], x_span[1], n)

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

        B1 = np.zeros(n)
        B2 = np.zeros(n)

        A = np.empty((n, 2))
        B = np.empty((n, 2))

        for i in range(n):
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
            plt.plot([y2i, y2i - .2*(y2i-y1i - wi * t1i)], [l2, l2 - .2*(l2 - l1 - wi * t2i)], 'g', linewidth=1.5)
        plt.plot([10 for i in range(100)], np.linspace(0, 12, 100))
        plt.plot(np.linspace(0, 12, 100), [10 for i in range(100)])

        plt.plot(x_discrete, u_discrete, "g")
        plt.plot(B1, B2, "g")
        plt.plot(x_span, [0, 0], "k")
        plt.plot(y1_span, [l1, l1], "k")
        plt.plot(y2_span, [l2, l2], "k")

        plt.axis("equal")
        plt.tight_layout()
        plt.show()

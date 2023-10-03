import doptics.single_mirror as ssm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from doptics import functions as func

FILENAME = {1: 'normal_target', -1: 'crossing-beams'}
XL = 1
XR = 2
YL = 2
YR = 10
SIGMA = 1/5

if __name__ == '__main__':
    # res = solver.solve()
    # print(res)

    number_rays = 15
    x_span = [XL, XR]
    y_span = [YL, YR]
    xs = np.linspace(x_span[0], x_span[1], number_rays)

    # result_type = 1  # 1 or -1

    L = -1
    y0 = 1/10
    u0 = 1

    E = func.E  # lambda x: 1
    G = func.G1  # lambda x: 10

    ssm.solve_single_mirror_parallel_source(starting_distribution=E, target_distribution=G,
                                            x_span=x_span, y_span=y_span,
                                            u0= u0, l=L,
                                            number_rays=15)


    # G = func.normal_target
    # result_type = 1
    for result_type in [1, -1]:
        a = sp.integrate.quad(E, x_span[0], x_span[1])[0] / sp.integrate.quad(G, y_span[0], y_span[1])[0]
        G_fixed = lambda y: a * G(y)

        m_prime = lambda x, m: result_type * E(x) / G_fixed(m)

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
        plt.savefig(f'solution-mirror-{FILENAME[result_type]}.png')
        # plt.show()

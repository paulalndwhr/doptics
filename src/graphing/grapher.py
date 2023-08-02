import matplotlib.pyplot as plt
from typing import List
import numpy as np
import numpy.typing as npt
import scipy as sc





def plot_rays(rays: List[Ray]):
    for ray in rays:
        plt.plot(ray.x, ray.y)
    plt.show()


def graph_single_mirror_parallel_source(x_l: float, x_r: float, y_l: float, y_r: float, rays: List[Ray],  # u: npt.NDArray[float],
                                        x_offset: float = 0, y_offset: float = -5
                                        ):
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(xdata=[x_l, x_r], ydata=[x_offset, x_offset], marker='x', color='g')
    # ax.plot(xdata=[y_l, y_r], ydata=[y_offset, y_offset], marker='x', color='g')
    # plt.show()
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), tight_layout=True)

    # # single point
    # ax[0].plot(105, 110, '-ro', label='line & marker - no line because only 1 point')
    # ax[0].plot(200, 210, 'go', label='marker only')  # use this to plot a single point
    # ax[0].plot(160, 160, label='no marker - default line - not displayed- like OP')
    # ax[0].set(title='Markers - 1 point')
    # ax[0].legend()

    # two points
    ax[0].plot([x_l, x_r], [x_offset, x_offset], '-rx', label='line & marker')
    ax[0].plot([y_l, y_r], [y_offset, y_offset], '-gx', label='marker only')
    for ray in rays:
        print(ray.x)
        print(ray.y)
        ax[0].plot(ray.x, ray.y, '-m')
    # ax[0].plot([105, 110], [190, 200], label='no marker - default line')
    ax[0].set(title='Line & Markers - 2 points')
    ax[0].legend()

    # scatter plot
    ax[1].scatter(x=105, y=110, c='r', label='One Point')  # use this to plot a single point
    ax[1].scatter(x=[80, 85, 90], y=[85, 90, 95], c='g', label='Multiple Points')
    ax[1].set(title='Single or Multiple Points with using .scatter')
    ax[1].legend()
    plt.show(block=True)


# def E(x):
#     if isinstance(x, np.ndarray):
#         return np.ones(len(x))
#     else:
#         return 1
E = lambda x: 1
G = lambda x: 1
#
# def G(y):
#     return y


def solve_ode(xl, xr, yl, yr, steps=20):
    Fm = lambda x, m: E(x) / G(m)
    solm = sc.integrate.solve_ivp(Fm, [xl, xr], [yl], t_eval=np.linspace(xl, xr, steps), dense_output=True)
    print(solm)
    print(solm.y[0])
    return solm


if __name__ == '__main__':
    plotting_rays = [Ray(x=[4, 6, 3, 12], y=[0, 3, 4, -1]), Ray(x=[4, 7], y=[1, 1])]
    xl = 1
    xr = 2
    yl = 0.1  # np.sqrt(2)
    yr = 2
    x_offset = 0
    y_offset = -5
    steps=30

    plot_rays(plotting_rays)

    # solm = solve_ode(xl=xl, xr=xr, yl=yl, yr=yr,  steps=steps)
    #
    # L = 2  # z: minus value of target distribution
    # # F = lambda x, u: (m(x) - x) / (np.sqrt((m(x) - x) ** 2 + (L + u) ** 2) + L + u)  # rhs of ODE
    # F2 = lambda x, u: (solm.sol(x) - x) / (np.sqrt((solm.sol(x) - x) ** 2 + (L + u) ** 2) + L + u)  # rhs of ODE
    #
    # # erg = solve_ode(xl=xl, xr=xr, yl=yl, yr=yr, steps=steps)
    #
    # sol = sc.integrate.solve_ivp(F2, [xl, xr], [1], t_eval=np.linspace(xl, xr, steps))
    # plt.plot(sol.t, sol.y.T, color='b')
    # # plt.plot([sol.t[0], sol.t[0], m(sol.t[0])], [0, sol.y.T[0], -L], color='r')
    # # plt.plot([sol.t[25], sol.t[25], m(sol.t[25])], [0, sol.y.T[25], -L], color='r')
    # # print(m(y))
    # for i, t in enumerate(sol.t):
    #     plt.plot([t, t, solm.y.T[i]], [0, sol.y.T[i], -L], color='r')
    # plt.grid(True)
    # plt.show(block=True)
    #
    # graph_single_mirror_parallel_source(x_l=xl, x_r=xr, y_l=yl, y_r=yr, rays=plotting_rays)

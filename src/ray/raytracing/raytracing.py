#!/usr/bin/env python3
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib as mpl
from functools import lru_cache
mpl.rcParams['figure.dpi'] = 100
COLORS = {'szegedblue': '#3f3fa6'}


def line(p1, p2):
    a_ = (p1[1] - p2[1])
    b_ = (p2[0] - p1[0])
    c_ = (p1[0]*p2[1] - p2[0]*p1[1])
    return a_, b_, -c_


def intersection(l1, l2):
    d = l1[0] * l2[1] - l1[1] * l2[0]
    dx = l1[2] * l2[1] - l1[1] * l2[2]
    dy = l1[0] * l2[2] - l1[2] * l2[0]
    if d != 0:
        x = dx / d
        y = dy / d
        return x, y
    else:
        return np.nan


"""
Example of ray tracing
Variable names are same as in the report
"""

L1 = 8
L2 = 14
y1_SPAN = (10.5, 13.5)
y2_SPAN = (11, 13)
x_SPAN = (0, 3)
u0 = 4
w0 = 6
# E = lambda x: np.exp(-((x - 1.5) / 1)**2 / 2) / (2 * np.pi)**.5
def G1(y1): return 1
def G2(y2): return 1


E = G2
PM1 = -1
PM2 = 1


@lru_cache(maxsize=100)
def integrate(f_, left: float, right: float) -> float:
    return sp.integrate.quad(f_, left, right)[0]


if __name__ == '__main__':
    # calculate theoretical target from the mapping between emitting density E and target density G1
    def G1_fixed(y1, y1_span=y1_SPAN): return G1(y1) / integrate(G1, y1_span[0], y1_span[1])
    def E_fixed(x, x_span=x_SPAN): return E(x) / integrate(x, x_span[0], x_span[1])
    def m1_prime(x, m, pm1=PM1): return pm1 * E_fixed(x) / G1_fixed(m)

    y1_0 = y1_SPAN[1]
    m1_solved = sp.integrate.solve_ivp(m1_prime, x_SPAN, [y1_0], dense_output=1)
    def m1(x): return m1_solved.sol(x)[0]
    # m1 = lambda x: m1_solved.sol(x)[0]

    # absue that solution was calculated already by importing mirror points
    reflectors = np.load("reflectors.npz")
    A = reflectors["arr_0"]  # A is discrete reflector as solved by ODE
    B = reflectors["arr_1"]  # B is ...
    print(B)
    # List of (x, y) pairs
    B_relevant = np.flipud(B)
    print(B_relevant)
    spline_A = sp.interpolate.CubicSpline(A[:, 0], A[:, 1])
    spline_B = sp.interpolate.CubicSpline(B_relevant[:, 0], B_relevant[:, 1], extrapolate=True)
    # spline_B = sp.interpolate.CubicSpline(np.flipud(B[:][0]), np.flipud(B[:][1]))
    # spline_B = sp.interpolate.CubicSpline(np.flipud(B[:][0]), B[:][1])

    # x_arr_u = np.linspace(x_SPAN[0], x_SPAN[1], 1000)
    # x_arr_w = np.linspace(B[0][0], B[-1][0], 1000)
    # plt.plot(x_arr_u, spline_A(x_arr_u), "r")
    # plt.plot(A[:, 0], A[:, 1], "k")
    # plt.plot(x_arr_w, spline_B(x_arr_w), "g")
    # plt.plot(B[:, 0], B[:, 1], "k")
    # plt.show()

    RUNS = 1000
    errn = np.zeros(RUNS)
    for n_rays in range(1, RUNS):
        # n_rays = 490
        print(n_rays)
        xis = np.linspace(0, 3, n_rays)
        targets = m1(xis)
        # print(f'target x-coords are {targets}')

        # First reflector
        Ps = np.zeros((n_rays, 2))  # what is Ps
        As = np.zeros((n_rays, 2))  # As is reflector A?
        us = np.zeros((n_rays, 2))
        for i, x in enumerate(xis):
            for j in range(A.shape[0] - 1):
                a = np.array([x, 0])
                b = np.array([x, 5])
                c = A[j, :]
                d = A[j + 1, :]
                e = b - a
                f = d - c
                p = np.array([-e[1], e[0]])
                h = np.dot(a - c, p) / np.dot(f, p)
                if 0 <= h <= 1:
                    o = c + f * h
                    n = np.array([-f[1], f[0]])
                    n /= np.linalg.norm(n)
                    us[i] = e - 2 * np.dot(e, n) * n
                    As[i] = o
                    Ps[i] = a
                    break
            # find intersection with spline reflector
            # implementation for parallel rays
        hits_U = spline_A(xis)
        hits_U_prime = spline_A(xis, 1)
        U_reflection = - np.sin(np.arccos(hits_U_prime))
        # plt.plot(xis, hits_U)
        # plt.plot([xis, xis, xis + 10], [np.zeros(len(xis)), hits_U, hits_U + 10*U_reflection])
        # plt.show()

        # Second reflector
        Bs = np.zeros((n_rays, 2))
        u2s = np.zeros((n_rays, 2))

        spline_intersections = np.zeros(len(xis))
        for i in range(n_rays):
            for j in range(B.shape[0] - 1):
                a = As[i]
                c = B[j, :]
                d = B[j + 1, :]
                e = us[i]
                f = d - c
                p = np.array([-e[1], e[0]])
                h = np.dot(a - c, p) / np.dot(f, p)
                if 0 <= h <= 1:
                    o = c + f * h
                    n = np.array([-f[1], f[0]])
                    n /= np.linalg.norm(n)
                    u2s[i] = e - 2 * np.dot(e, n) * n
                    Bs[i] = o
                    break
            # calculate intersection with spline reflector
            # ray_AB = lambda x: spline_A(xis[i]) - spline_A(xis[i], 1) * x
            # diff = lambda x: spline_B(x) - ray_AB(x)
            # spline_intersections[i] = sp.optimize.newton(diff, m1(xis[i]))  # , fprime2=lambda x: 6 * x)
                # sp.optimize.bisect(diff, a=-20, b=20)
        # spline_B
        # hits_W = spline_A(xis)
        # hits_W_prime = spline_A(xis, 1)
        # W_reflection = np.sin(np.arccos(hits_U_prime))
        # plt.plot(xis, hits_U)
        # plt.plot([xis, xis, xis + 10], [np.zeros(len(xis)), hits_U, hits_U - 10 * U_reflection])
        # plt.show()

        # Plotting
        # for i in range(n_rays):
        #     plt.plot([Ps[i][0], As[i][0], Bs[i][0], Bs[i][0] + 4 * u2s[i][0]],
        #              [Ps[i][1], As[i][1], Bs[i][1], Bs[i][1] + 4 * u2s[i][1]])
        # plt.plot(A[:, 0], A[:, 1], "k")
        # plt.plot(B[:, 0], B[:, 1], "k")
        #
        # plt.axis("equal")
        # plt.tight_layout()
        # plt.show()

        # ray tracing error
        # Bs[i][0], Bs[i][0] + 4 * u2s[i][0]  # x-coords
        # Bs[i][1], Bs[i][1] + 4 * u2s[i][1]  # y-coords
        error = np.zeros(n_rays)
        for i in range(n_rays):
            ray_low = (Bs[i][0], Bs[i][1])
            ray_upp = (Bs[i][0] + 4 * u2s[i][0], Bs[i][1] + 4 * u2s[i][1])
            ray = line(ray_low, ray_upp)
            target_line = line((y1_SPAN[0], L1), (y1_SPAN[1], L1))
            intersec = intersection(target_line, ray)
            # print(intersec)
            try:
                error[i] = np.abs(targets[i] - intersec[0]) * E(xis[i])
                # error[i] = (targets[i] - intersec[0]) * E(xis[i])
            except TypeError:
                error[i] = np.nan
            # print(error)
        # error = error / n_rays
        # print(f'error: {error}\nsum(error): {sum(error)}')
        indices_to_drop = np.argwhere(error == np.nan)
        error = np.delete(error, indices_to_drop)
        error = error / len(error)
        # print(f'error: {error}\nsum(error): {sum(error)}')
        # print(f'targets: {targets}')
        errn[n_rays] = np.nansum(error)
    plt.plot(np.arange(1, RUNS+1, 1), errn, COLORS['szegedblue'])
    plt.title('ERTE, Affine Reflector')
    plt.show()

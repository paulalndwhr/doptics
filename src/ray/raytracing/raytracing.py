#!/usr/bin/env python3
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp



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
        return False




"""
Example of ray tracing
Variable names are same as in the report
"""

L1 = 8
L2 = 14
y1_span = (10.5, 13.5)
y2_span = (11, 13)
x_span = (0, 3)
u0 = 4
w0 = 6
E = lambda x: np.exp(-((x - 1.5) / 1)**2 / 2) / (2 * np.pi)**.5
G1 = lambda y1: 1
G2 = lambda y2: 1
pm1 = -1
pm2 = 1


if __name__ == '__main__':
    # calculate theoretical target from the mapping between emitting density E and target density G1
    m1_prime = lambda x, m: pm1 * E(x) / G1(m)
    y1_0 = y1_span[1]
    m1_solved = sp.integrate.solve_ivp(m1_prime, x_span, [y1_0], dense_output=1)
    m1 = lambda x: m1_solved.sol(x)[0]

    # absue that solution was calculated already by importing mirror points
    reflectors = np.load("reflectors.npz")
    A = reflectors["arr_0"]  # A is discrete reflector as solved by ODE
    B = reflectors["arr_1"]  # B is ...
    # List of (x, y) pairs

    errn = np.zeros(1000)
    for n_rays in range(1, 1000):
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

        # Second reflector
        Bs = np.zeros((n_rays, 2))
        u2s = np.zeros((n_rays, 2))
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
            target_line = line((y1_span[0], L1), (y1_span[1], L1))
            intersec = intersection(target_line, ray)
            # print(intersec)
            try:
                error[i] = np.abs(targets[i] - intersec[0]) * E(xis[i])
                # error[i] = (targets[i] - intersec[0]) * E(xis[i])
            except TypeError:
                error[i] = -1
            # print(error)
        # error = error / n_rays
        # print(f'error: {error}\nsum(error): {sum(error)}')
        error = error / n_rays
        # print(f'error: {error}\nsum(error): {sum(error)}')
        # print(f'targets: {targets}')
        errn[n_rays] = sum(error)
    plt.plot(np.arange(1, 1001, 1), errn)
    plt.show()

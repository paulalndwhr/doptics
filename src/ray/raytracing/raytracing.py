#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


"""
Example of ray tracing
Variable names are same as in the report
"""

if __name__ == '__main__':

    reflectors = np.load("reflectors.npz")
    A = reflectors["arr_0"]  # A is discrete shape of
    B = reflectors["arr_1"]
    print(A)

    n_rays = 100
    xis = np.linspace(0, 3, n_rays)

    # First reflector
    Ps = np.zeros((n_rays, 2))
    As = np.zeros((n_rays, 2))
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
    for i in range(n_rays):
        plt.plot([Ps[i][0], As[i][0], Bs[i][0], Bs[i][0] + 4 * u2s[i][0]],
                 [Ps[i][1], As[i][1], Bs[i][1], Bs[i][1] + 4 * u2s[i][1]])
    plt.plot(A[:, 0], A[:, 1], "k")
    plt.plot(B[:, 0], B[:, 1], "k")

    plt.axis("equal")
    plt.tight_layout()
    plt.show()

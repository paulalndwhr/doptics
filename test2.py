import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from doptics.symbolic import u_prime_mv
import doptics.functions as func

import doptics.two_mirrors as tms
from doptics.functions import cdf_sampling_source

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
    result = tms.solve_two_mirrors_parallel_source_two_targets(starting_density=func.uniform,
                                                               target_distribution_1=func.G1,
                                                               target_distribution_2=func.G1, x_span=x_span,
                                                               y1_span=y1_span, y2_span=y2_span, u0=u0, w0=w0,
                                                               l1=l1, l2=l2,
                                                               # color='#a69f3f',
                                                               number_rays=15
                                                               )

    def G1(y1): return 1
    def G2(y2): return 1 - np.abs(y2 - 12) if 11 < y2 < 13 else 0
    def E(x): return 1 / (np.exp(10 * (x - 0.5)) + np.exp(-10 * (x - 0.5)))

    # new stuff
    # E = func.uniform

    angle_result = tms.solve_two_mirrors_parallel_source_point_target(starting_density=func.uniform,
                                                                      target_distribution_1=G1,
                                                                      target_distribution_2=G2,
                                                                      x_span=x_span,
                                                                      y1_span=y1_span, y2_span=y2_span, u0=u0, w0=w0,
                                                                      l1=l1, l2=l2,
                                                                      # color='#a69f3f',
                                                                      number_rays=16)

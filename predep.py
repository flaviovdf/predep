from astropy.stats import bayesian_blocks

from statsmodels.distributions import ECDF

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import statsmodels.api as sm

def predep(X, Y, n_boot=10000):
    BX1 = np.random.choice(X, n_boot)
    BX2 = np.random.choice(X, n_boot)
    DX = BX1 - BX2
    s_x = ss.gaussian_kde(DX).pdf(0)

    edges_y = bayesian_blocks(Y)
    ecdf_y = ECDF(Y)

    s_x_mid_y = 0
    if edges_y.shape[0] > 1:
        for i in range(1, edges_y.shape[0]):
            bg = edges_y[i - 1]
            ed = edges_y[i]
            X_mid_Y = X[(Y >= bg) & (Y < ed)]
            if X_mid_Y.shape[0] == 0:
                continue

            BX_mid_Y1 = np.random.choice(X_mid_Y, n_boot)
            BX_mid_Y2 = np.random.choice(X_mid_Y, n_boot)
            DX_mid_Y = BX_mid_Y1 - BX_mid_Y2

            p_range = ecdf_y(ed) - ecdf_y(bg)

            p_x_mid_y = ss.gaussian_kde(DX_mid_Y).pdf(0)
            s_x_mid_y += p_range * (p_x_mid_y)
        alpha_est = (s_x_mid_y - s_x) / s_x_mid_y
        return alpha_est
    else:
        return np.nan

if __name__ == '__main__':
    rho = 0.65
    mean = [0, 0]
    cov = [
        [1, rho],
        [rho, 1]
    ]
    alpha = 1 - np.sqrt(1 - rho ** 2)
    print(alpha)

    XY = np.random.multivariate_normal(mean, cov, 500)
    X = XY[:, 0]
    Y = XY[:, 1]
    print(predep_bb(X, Y))

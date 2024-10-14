import math

import numpy as np
from numpy.random import uniform

from dp2p import dp2p, LambdaEstimates


def sample_R_x_horizontal():

    r_y = uniform(0, 2 * math.pi)
    s = math.sin(r_y)
    c = math.cos(r_y)
    R_y = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])

    r_x = uniform(0, 2 * math.pi)
    s = math.sin(r_x)
    c = math.cos(r_x)
    R_x = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])

    R_xy = R_x @ R_y
    assert R_xy[0, 1] == 0
    return R_xy


def sample_input(exact):

    C_gt = uniform(-10.0, 10.0, 3)
    R_gt = sample_R_x_horizontal()

    x1 = uniform(-1, 1, 3)
    x1[2] = 1.0
    lambda1 = uniform(0.2, 100, 1)
    x1 *= lambda1

    x2 = uniform(-1, 1, 3)
    x2[2] = 1.0
    lambda2 = uniform(0.2, 100, 1)
    x2 *= lambda2

    # R(X - C) = x
    # X = R.T.x + C
    X1 = R_gt.T @ x1 + C_gt
    X2 = R_gt.T @ x2 + C_gt

    assert np.allclose(R_gt @ (X1 - C_gt), x1)
    assert np.allclose(R_gt @ (X2 - C_gt), x2)

    if not exact:
        err_low = 0.5
        err_high = 2.0
        lmbd1_err = uniform(err_low, err_high,1)
        x1 *= lmbd1_err
        lmbd2_err = uniform(err_low, err_high,1)
        x2 *= lmbd2_err

    return R_gt, C_gt, x1, x2, X1, X2


def check_solution(R, C, x1, x2, X1, X2, exact, estimates: LambdaEstimates):

    def check_point(x, X):
        dilate = R @ (X - C) / x
        assert np.isclose(dilate.min(), dilate.max())
        if exact:
            assert np.isclose(dilate[0], 1.0)
        return dilate[0]

    assert R[0, 1] == 0.0
    dilate1 = check_point(x1, X1)
    dilate2 = check_point(x2, X2)
    if estimates == LambdaEstimates.FIX_ANGLE:
        assert np.isclose(dilate1, dilate2)


def check_solver():

    np.random.seed()
    synthetic_samples = 100

    for estimates in [LambdaEstimates.FIX_LAMBDA, LambdaEstimates.FIX_ANGLE, LambdaEstimates.TRY_BOTH]:

        for exact in [True, False]:
            sols_c = 0
            for i in range(synthetic_samples):
                R_gt, C_gt, x1, x2, X1, X2 = sample_input(exact)
                sols = dp2p(x1, x2, X1, X2, estimates=estimates)
                sols_c += len(sols)
                for R, C in sols:
                    check_solution(R, C, x1, x2, X1, X2, exact, estimates)
            print(f"{estimates}, lambdas exact - {exact}: {sols_c} solutions in {synthetic_samples} instances")


if __name__ == '__main__':
    check_solver()

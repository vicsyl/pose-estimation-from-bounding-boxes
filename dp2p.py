import math
from enum import Enum

import numpy as np
from numpy import ndarray as arr


def solve_quadratic_equation(a, b, c):
    D = b ** 2 - 4 * a * c
    if D > 0:
        return [(-b + math.sqrt(D)) / (2 * a), (-b - math.sqrt(D)) / (2 * a)]
    elif D == 0:
        return [-b / (2 * a)]
    else:
        return []


def solve_for_2nd_row(X_d, x_d, R_out):
    """
    :param X_d: numpy.ndarray[3]: X2 - X1
    :param x_d: numpy.ndarray[3]: x2 - x1
    :param R_out: numpy.ndarray[3, 3] - array to write the solution to
    :return: True iff there is a valid solution
    """

    r11 = R_out[0, 0]
    r13 = R_out[0, 2]
    assert r11 != 0 or r13 != 0

    # NOTE FOR ALL COMMENTS: symbolics (r_ij) are one-based, arrays (x_d, X_d) are zero based
    # I: r11 * r21 + r13 * r23 = 0
    #    => second(r11 or r13) = a * first(r31 or r11)

    # II: X_d[0] * r_21 + X_d[1] * r_22 + X_d[2] * r_23 = x_d[1]
    #    => coeff * first + X_d[1] * r_22 = x_d[1]

    # r11 * r21 + r13 * r23 = 0
    if r13 == 0:
        solve_first_r21 = False
        # r21 = 0 * r23
        a = 0
        coeff = X_d[2]
    else:
        solve_first_r21 = True
        # r23 = (-r31/r33) * r21 = a * r21
        a = -r11 / r13
        # (a * X_d[2] + X_d[0]) * r21 = x_d[1]
        coeff = a * X_d[2] + X_d[0]

    # coeff * first = x_d[1]
    if X_d[1] == 0:
        if coeff == 0:
            return False
        else:
            # coeff * first = x_d[1]
            first = x_d[1] / coeff
            second = a * first
            r_21 = first if solve_first_r21 else second
            r_23 = second if solve_first_r21 else first

            A = 1
            C = r_21 ** 2 + r_23 ** 2 - 1

            if C > 0:
                return False
            elif C == 0:
                roots = [0.0]
            else:
                root = math.sqrt(-C / A)
                roots = [root, -root]

            for i in range(len(roots)):
                r_22 = roots[i]
                R_out[1] = [r_21, r_22, r_23]
                R_out[2] = np.cross(R_out[0], R_out[1])
                diff = R_out[2] @ X_d - x_d[2]
                if math.fabs(diff) < 1e-6:
                    return True
            return False

    # coeff * first + X_d[1] * r_22 = x_d[1]
    # X_d[1] != 0
    else:
        # r_22 = r22_a * first + r22_b
        r22_a = -coeff / X_d[1]
        r22_b = x_d[1] / X_d[1]

        # r21 ** 2 + r22 ** 2 + r23 ** 2 = 1
        # r21 ** 2 + (r22_a * first + r22_b) ** 2 + r23 ** 2 = 1
        # (a * first) ** 2 + (r22_a * first + r22_b) ** 2 + first ** 2 = 1 => solve for first
        # A * first ** 2 + B * first + C = 0
        A = 1 + r22_a ** 2 + a ** 2
        B = 2 * r22_a * r22_b
        C = r22_b ** 2 - 1.0
        roots = solve_quadratic_equation(A, B, C)

        for root in roots:
            first = root
            second = a * first

            r_22 = r22_a * first + r22_b
            r_21 = first if solve_first_r21 else second
            r_23 = second if solve_first_r21 else first
            R_out[1] = [r_21, r_22, r_23]
            R_out[2] = np.cross(R_out[0], R_out[1])

            # only at most one root solves the third equation
            diff = R_out[2] @ X_d - x_d[2]
            if math.fabs(diff) < 1e-6:
                return True

        return False


def dp2p_exact(x1: arr, x2: arr, X1: arr, X2: arr):
    """
    :param x1: numpy.ndarray[3] - the 1st 2D correspondence times depth
    :param x2: numpy.ndarray[3] - the 2nd 2D correspondence times depth
    :param X1: numpy.ndarray[3] - the 1st 3D correspondence
    :param X2: numpy.ndarray[3] - the 2nd 3D correspondence

    Camera pre-rotated so that camera x-axis is on the x-z world plane:
    R.T @ [1, 0, 0].T = [x, 0, z].T => r_12 = 0
    :return: list of solutions for camera poses [(R, C)]
    """

    X_d = X2 - X1
    x_d = x2 - x1

    # NOTE FOR ALL COMMENTS: symbolics (r_ij) are one-based, arrays (x_d, X_d) are zero based
    # eq: X_d[0] * r11 + X_d[2] * r13 = x_d[0]
    # => r_second = a * r_first + b
    if X_d[0] != 0.0:
        # r11 = (x_d[0] - X_d[2] * r13) / X_d[0]
        # r11 = a * r13 + b
        # solve for r13
        solve_r11 = False
        a = -X_d[2] / X_d[0]
        b = x_d[0] / X_d[0]
        if not np.isclose((a + b) * X_d[0] + 1 * X_d[2], x_d[0]):
            assert np.isclose((a + b) * X_d[0] + 1 * X_d[2], x_d[0])
    else:
        # r13 = (x_d[2] - X_d[0] * r11) / X_d[2]
        # solve for r31
        solve_r11 = True
        a = -X_d[0] / X_d[2]
        b = x_d[0] / X_d[2]
        assert np.isclose((a + b) * X_d[2] + 1 * X_d[0], x_d[0])

    # unit circle: r11 ** 2 + r13 ** 2 = 1
    # r_first ** 2 + (a * r_first + b) ** 2 = 1
    # (1 + a ** 2) * r_first ** 2 + 2 * a * b * r_first + b ** 2 - 1 = 0
    # A * r_first ** 2 + B * r_first + C = 0
    A = 1 + a * a
    B = 2 * a * b
    C = b * b - 1.0
    roots = solve_quadratic_equation(A, B, C)

    ret = []
    for root in roots:

        first = root
        second = a * first + b
        r11 = first if solve_r11 else second
        r13 = second if solve_r11 else first
        zeros = [0.0] * 3
        R = np.array([[r11, 0.0, r13], zeros, zeros])

        assert np.isclose((r11 ** 2 + r13 ** 2).item(), 1.0)
        assert np.isclose(r11 * X_d[0] + r13 * X_d[2], x_d[0])

        add = solve_for_2nd_row(X_d, x_d, R)
        if add:
            cam_C = -R.T @ x1 + X1
            assert np.allclose(R @ (X1 - cam_C), x1)
            assert np.allclose(R @ (X2 - cam_C), x2)
            ret.append((R, cam_C))

    return ret


def compute_other_length(c, a, cos_gamma, orig_b):
    """
    :param c: the distance between 3D correspondences
    :param a: the length to fix
    :param cos_gamma: cosine of the angle between the rays
    :param orig_b: the original length to be updated
    :return: the new length or None if there is no feasible solution
    """

    # -b ** 2 - a ** 2 + 2 * cos_gamma * a * b + c ** 2 = 0 -> solve for b
    a_cos_gamma = a * cos_gamma
    quarter_D = a_cos_gamma ** 2 + c ** 2 - a ** 2
    if quarter_D < 0:
        return None
    elif quarter_D == 0.0:
        return a_cos_gamma
    else:
        half_sq_D = math.sqrt(quarter_D)
        r1 = a_cos_gamma - half_sq_D
        r2 = a_cos_gamma + half_sq_D
        if r1 < 0:
            return r2
        else:
            r1_d = math.fabs(r1 - orig_b)
            r2_d = math.fabs(r2 - orig_b)
            return r1 if r1_d < r2_d else r2


class LambdaEstimates(Enum):
    FIX_LAMBDA = 1
    FIX_ANGLE = 2
    TRY_BOTH = 3


def dp2p(x1: arr, x2: arr, X1: arr, X2: arr, estimates: LambdaEstimates):
    """
    :param x1: numpy.ndarray[3] - the 1st 2D correspondence with depth
    :param x2: numpy.ndarray[3] - the 2nd 2D correspondence with depth
    :param X1: numpy.ndarray[3] - the 1st 3D correspondence
    :param X2: numpy.ndarray[3] - the 2nd 3D correspondence
    :param estimates: LambdaEstimates - one of
        LambdaEstimates.FIX_LAMBDA - fix one of the lambdas at a time: returns at most 2 * 2 = 4 valid solutions
        LambdaEstimates.FIX_ANGLE - fix the angle between the 2D correspondences: returns at most 2 valid solutions
        LambdaEstimates.TRY_BOTH - try both strategies: returns at most 4 + 2 = 6 valid solutions


    Camera pre-rotated so that camera x-axis is on the x-z world plane:
    R.T @ [1, 0, 0].T = [x, 0, z].T
    :return: list of solutions [(R, C)]
    """

    ret = []
    c = np.linalg.norm(X2 - X1)
    if estimates == LambdaEstimates.FIX_ANGLE or estimates == LambdaEstimates.TRY_BOTH:
        dilate = c / math.sqrt(x1 @ x1 + x2 @ x2 - 2 * x1 @ x2)
        xin1 = dilate * x1
        xin2 = dilate * x2
        ret.extend(dp2p_exact(xin1, xin2, X1, X2))

    if estimates == LambdaEstimates.FIX_LAMBDA or estimates == LambdaEstimates.TRY_BOTH:

        x1_l = np.linalg.norm(x1)
        x2_l = np.linalg.norm(x2)
        cos_gamma = x1 @ x2 / (np.linalg.norm(x1) * np.linalg.norm(x2))

        new_x1_l = compute_other_length(c, x2_l, cos_gamma, x1_l)
        if new_x1_l:
            new_x1 = x1.copy() * new_x1_l / np.linalg.norm(x1)
            ret.extend(dp2p_exact(new_x1, x2, X1, X2))

        new_x2_l = compute_other_length(c, x1_l, cos_gamma, x2_l)
        if new_x2_l:
            new_x2 = x2.copy() * new_x2_l / np.linalg.norm(x2)
            ret.extend(dp2p_exact(x1, new_x2, X1, X2))

    return ret

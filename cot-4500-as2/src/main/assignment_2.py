import numpy as np
import math


# 1. Neville’s Method
def neville_interpolation(x, y, x_target):
    n = len(x)
    Q = np.zeros((n, n))
    Q[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            Q[i, j] = (
                (x_target - x[i + j]) * Q[i, j - 1]
                - (x_target - x[i]) * Q[i + 1, j - 1]
            ) / (x[i] - x[i + j])

    return Q[0, n - 1]


# 2. Newton’s Forward Method
def newton_forward_table(x_vals, y_vals):
    n = len(x_vals)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_vals

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = diff_table[i + 1, j - 1] - diff_table[i, j - 1]

    return diff_table


def newton_forward_coefficients(x_vals, y_vals):
    diff_table = newton_forward_table(x_vals, y_vals)
    h = x_vals[1] - x_vals[0]

    coefficients = [diff_table[0, i] / (h ** (i)) for i in range(1, len(x_vals))]

    return coefficients


# 3. Approximate f(7.3)
def newton_forward_interpolation(x_vals, y_vals, x_target):
    diff_table = newton_forward_table(x_vals, y_vals)
    h = x_vals[1] - x_vals[0]
    s = (x_target - x_vals[0]) / h

    interpolation_result = y_vals[0]
    term = 1

    for i in range(1, len(x_vals)):
        term *= (s - (i - 1)) / i
        interpolation_result += term * diff_table[0, i]

    return interpolation_result


# 4. Hermite Polynomial
def hermite_divided_difference_table(x_values, y_values, dy_values):
    n = len(x_values)
    table = np.zeros((2 * n, 5))

    for i in range(n):
        table[2 * i][0] = table[2 * i + 1][0] = x_values[i]
        table[2 * i][1] = table[2 * i + 1][1] = y_values[i]
        table[2 * i + 1][2] = dy_values[i]

    for i in range(1, 2 * n, 2):
        table[i, 2] = dy_values[i // 2]
        table[i - 1, 2] = dy_values[i // 2]

    for col in range(3, 5):
        for row in range(2 * n - col):
            denominator = table[row + col - 2, 0] - table[row, 0]
            if abs(denominator) > 1e-12:
                table[row, col] = (
                    table[row + 1, col - 1] - table[row, col - 1]
                ) / denominator

    return table


# 5. Cubic Spline Interpolation
def cubic_spline_matrix(x_values, y_values):
    n = len(x_values) - 1
    h = np.diff(x_values)
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = (3 / h[i]) * (y_values[i + 1] - y_values[i]) - (3 / h[i - 1]) * (
            y_values[i] - y_values[i - 1]
        )

    return A, b


def solve_cubic_spline(A, b):
    x_vector = np.linalg.solve(A, b)
    return x_vector

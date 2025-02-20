import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../main")))

from assignment_2 import (
    neville_interpolation,
    newton_forward_coefficients,
    newton_forward_interpolation,
    hermite_divided_difference_table,
    cubic_spline_matrix,
    solve_cubic_spline,
)


# 1. Neville’s Method
def test_neville_interpolation():
    x = np.array([3.6, 3.8, 3.9])
    y = np.array([1.675, 1.436, 1.318])
    x_target = 3.7
    result = neville_interpolation(x, y, x_target)
    print(result)


# 2. Newton’s Forward Method
def test_newton_forward_coefficients():
    x_vals = np.array([7.2, 7.4, 7.5, 7.6])
    y_vals = np.array([23.5492, 25.3913, 26.8224, 27.4589])

    coefficients = newton_forward_coefficients(x_vals, y_vals)

    for coef in coefficients:
        print(coef)


# 3. Approximate f(7.3)
def test_newton_forward_interpolation():
    x_vals = np.array([7.2, 7.4, 7.5, 7.6])
    y_vals = np.array([23.5492, 25.3913, 26.8224, 27.4589])
    x_target = 7.3

    interpolated_value = newton_forward_interpolation(x_vals, y_vals, x_target)
    print(interpolated_value)


# 4. Hermite Polynomial
def test_hermite_divided_difference():
    x_values = np.array([3.6, 3.8, 3.9])
    y_values = np.array([1.675, 1.436, 1.318])
    dy_values = np.array([-1.195, -1.188, -1.182])

    table = hermite_divided_difference_table(x_values, y_values, dy_values)

    for row in table:
        print(" ".join(f"{val:.6f}" for val in row))


# 5. Cubic Spline Interpolation
def test_cubic_spline():
    x_values = np.array([2, 5, 8, 10])
    y_values = np.array([3, 5, 7, 9])

    A, b = cubic_spline_matrix(x_values, y_values)
    x_vector = solve_cubic_spline(A, b)

    for row in A:
        print("[", " ".join(f"{val:.6f}" for val in row), "]")
    print("[", " ".join(f"{val:.6f}" for val in b), "]")
    print("[", " ".join(f"{val:.6f}" for val in x_vector), "]")


def main():
    print("Run Tests")
    test_neville_interpolation()
    print()
    test_newton_forward_coefficients()
    print()
    test_newton_forward_interpolation()
    print()
    test_hermite_divided_difference()
    print()
    test_cubic_spline()
    print("Tests passed")


if __name__ == "__main__":
    main()

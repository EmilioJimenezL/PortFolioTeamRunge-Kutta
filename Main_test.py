"""
Numerical Analysis: Calculus Interpolation, Derivation and Integration
Team Participants:
Emilio Ivan Jimenez Lopez
Brandon
Andre
Javi
"""
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import Proyecto2

"""
Excercise 1:
Construct the quadratic Lagrange interpolating polynomial for the following data points:
(1,2), (3,4), (4,3)
"""

def f_lagrange1(x):
    return -(2/3)*x**2 + (11/3)*x - 1

x = np.linspace(0,5,500)
y = f_lagrange1(x)

plt.plot(x,y, label = "Lagrange polynomial", color = "blue")

# Add the points used for interpolation
points_x = [1, 3, 4]
points_y = [2, 4, 3]
plt.scatter(points_x, points_y, color='red', label='Points', zorder=5)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Excercise 1")
plt.legend()
plt.grid()
plt.show()

"""
Excercise 2:
Find the polynomial of degree 2 that interpolates at the points:
(0,1), (1,2), (4,2)
"""

x_vals = [0, 1, 4]  # x coordinates
y_vals = [1, 2, 2]  # y coordinates

# Get the interpolation polynomial
polynomial = Proyecto2.lagrange_interpolation(x_vals, y_vals)

# Convert sympy expression to a numpy-compatible function
f_lagrange2 = sp.lambdify('x', polynomial, 'numpy')

# Create points for plotting
x = np.linspace(-0.5, 4.5, 500)
y = f_lagrange2(x)

# Plot the polynomial
plt.plot(x, y, label="Lagrange Polynomial", color="blue")

# Add the interpolation points
plt.scatter(x_vals, y_vals, color='red', label='Points', zorder=5)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Excercise 2")
plt.legend()
plt.grid()
plt.show()

"""
Find the polynomial of degree 2 that interpolates to y = x³ at the nodes x0 = 0,
x1 = 1, and x2 = 2.
"""
#List of x points

x_vals = [0, 1, 2]

#Define the function
def function_excercise3(x):
    return x**3

#Get the y points

y_vals = [function_excercise3(0), function_excercise3(1), function_excercise3(2)]

#Get and simplify the polynomials

polynomial = Proyecto2.lagrange_interpolation(x_vals, y_vals)

f_lagrange3 = sp.lambdify('x', polynomial, 'numpy')

#plotting

x = np.linspace(1/4, 5/4, 500)
y_lagrange = f_lagrange3(x)
y_function = function_excercise3(x)

plt.plot(x, y_lagrange, label="Lagrange Polynomial", color="blue")
plt.plot(x, y_function, label="x³", color="green", linestyle='--')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Exercise 3")
plt.legend()
plt.grid()
plt.show()
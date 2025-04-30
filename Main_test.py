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

x_vals = [1, 3, 4]
y_vals = [2, 4, 3]

polynomial = Proyecto2.lagrange_interpolation(x_vals, y_vals)
f_lagrange1 = sp.lambdify('x', polynomial, 'numpy')

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
Excercise 3:
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

"""
Excercise 4
Determine the Lagrange interpolation polynomial of degree two for the function
f (x) = 1/x with x0 = 2, x1 = 2.75, and x2 = 4. Use your result to approximate
f (3) = 1/3 . Plot the function and the polynomial.
"""

def function_excercise4(x):
    return 1/x

x_vals = [2, 2.75, 4]
y_vals = [function_excercise4(2), function_excercise4(2.75), function_excercise4(4)]

polynomial = Proyecto2.lagrange_interpolation(x_vals, y_vals)
f_lagrange4 = sp.lambdify('x', polynomial, 'numpy')

x = np.linspace(1.5, 4.5, 500)
y_lagrange = f_lagrange4(x)
y_function = function_excercise4(x)

plt.plot(x, y_lagrange, label="Lagrange Polynomial", color="blue")
plt.plot(x, y_function, label="x³", color="green", linestyle='--')

plt.scatter(3, 1/3, color='red', label='Points', zorder=5)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Exercise 4")
plt.legend()
plt.grid()
plt.show()

"""
Excercise 5
f(x) = ln(x)
x = 1.8
h = 0.1, 0.05, 0.01
"""
def function_excercise5(x):
    return np.log(x)

x = 1.8
h = [0.1, 0.05, 0.01]
exact_derivative = 1/x

# Prepare data for table
derivatives = []
errors = []
error_bounds = []  # New list for error bounds
for i in h:
    numerical_derivative = Proyecto2.forward(function_excercise5, x, i)
    derivatives.append(numerical_derivative)
    errors.append(abs(numerical_derivative - exact_derivative))
    # Calculate error bound using ξ = 1.8
    error_bound = i/(2 * 1.8**2)
    error_bounds.append(error_bound)

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 3))  # Increased figure width to accommodate new column

# Hide axes
ax.axis('off')

# Create table data
table_data = []
for i in range(len(h)):
    table_data.append([
        f'{h[i]:.2f}',
        f'{derivatives[i]:.6f}',
        f'{errors[i]:.2e}',
        f'{error_bounds[i]:.2e}',  # New column for error bound
    ])

# Create table
table = ax.table(
    cellText=table_data,
    colLabels=['Step size (h)', 'Numerical', 'Error', 'Error Bound'],
    loc='center',
    cellLoc='center'
)

# Customize table appearance
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)  # Adjust table size

# Add title
plt.title(f'Forward Difference Results for ln(x) at x = {x}\nExact Derivative = {exact_derivative:.6f}', pad=20)

plt.tight_layout()
plt.show()

"""
Excercise 6
Use the first order formulas to approximate the derivative of f (x) = ln(x) at x0 = 1.8 with h = 0.1, 0.5, 0.01, 0.001, and 0.0001. Determine the limits of the error in
your approximation and compare the results.
"""

def function_excercise6(x):
    return np.log(x)

x0 = 1.8
h = [0.1, 0.5, 0.01, 0.001, 0.0001]
exact_derivative = 1/x0
forward_derivatives = []
backward_derivatives = []
central_derivatives = []
forward_errors = []
backward_errors = []
central_errors = []

for i in h:
    fwd = Proyecto2.forward(function_excercise6, x0, i)
    bwd = Proyecto2.backward(function_excercise6, x0, i)
    cnt = Proyecto2.central(function_excercise6, x0, i)
    
    forward_derivatives.append(fwd)
    backward_derivatives.append(bwd)
    central_derivatives.append(cnt)
    
    forward_errors.append(abs(fwd - exact_derivative))
    backward_errors.append(abs(bwd - exact_derivative))
    central_errors.append(abs(cnt - exact_derivative))

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 4))

# Hide axes
ax.axis('off')

# Create table data
table_data = []
for i in range(len(h)):
    table_data.append([
        f'{h[i]:.4f}',
        f'{forward_derivatives[i]:.6f}',
        f'{forward_errors[i]:.2e}',
        f'{backward_derivatives[i]:.6f}',
        f'{backward_errors[i]:.2e}',
        f'{central_derivatives[i]:.6f}',
        f'{central_errors[i]:.2e}'
    ])

# Create table
table = ax.table(
    cellText=table_data,
    colLabels=['h', 'Forward', 'Error', 'Backward', 'Error', 'Central', 'Error'],
    loc='center',
    cellLoc='center'
)

# Customize table appearance
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Add title
plt.title(f'Derivative Approximations for ln(x) at x = {x0}\nExact Derivative = {exact_derivative:.6f}', pad=20)

plt.tight_layout()
plt.show()

"""
Excercise 7
Let f (x) = x³4x. Define an equally spaced set of nodes {xi} with separation
h = 0.1. Construct the derivative data set applying the forward difference formula
at every point. Plot the derivative data set and the derivative function and compare.
"""

# definie la funcion f(x)
def function_excercise7(x):
    return (x**3) * (4*x)

#derivada exacta de f(x)
def df_exact(x):
    return 16 * x**3 + 12 * x**2

h = 0.1
x_start = 0
x_end = 2
x_nodes = np.arange(x_start, x_end + h, h)  # nodos de 0 a 2 con paso 0.1

f_values = function_excercise7(x_nodes)

#formula de forward
fwd_diff = (f_values[1:] - f_values[:-1]) / h
x_midpoints = x_nodes[:-1]  # los forward diffs corresponden a los primeros puntos

df_exact_values = df_exact(x_midpoints)

# graficamos
plt.figure(figsize=(10,6))
plt.plot(x_midpoints, df_exact_values, label="Exact derivative", color="blue", linewidth=2)
plt.plot(x_midpoints, fwd_diff, '--', label="Numerical derivative (forward diff)", color="red")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.title("Exact vs Numerical derivative of f(x)=x^3(4x)")
plt.legend()
plt.grid(True)
plt.show()

"""
Excercise 8

"""
"""
Numerical Analysis Project: Calculus Interpolation, Derivation and Integration
==========================================================================

This project implements various numerical methods for:
- Lagrange Interpolation
- Numerical Differentiation
- Numerical Integration

Team Participants:
- Emilio Ivan Jimenez Lopez
- Brandon
- Andre
- Javi

Dependencies:
- sympy
- numpy
- matplotlib
- Proyecto2 (custom module)

Structure:
1. Exercises 1-4: Lagrange Interpolation
2. Exercises 5-7: Numerical Differentiation
3. Exercises 8-10: Numerical Integration
4. Practical Problems: Real-world applications

Note: This code requires the custom module 'Proyecto2' which contains implementations 
of the following functions:
- lagrange_interpolation()
- forward()
- backward()
- central()
- trapezoidal_simple()
- simpson_simple()
- trapezoidal()
- simpson()
- riemann_sum()
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import Proyecto2

def validate_inputs(x_vals, y_vals):
    """
    Validates input arrays for interpolation functions.
    
    Args:
        x_vals (list/array): x coordinates
        y_vals (list/array): y coordinates
        
    Raises:
        ValueError: If inputs are invalid
    """
    if len(x_vals) != len(y_vals):
        raise ValueError("x_vals and y_vals must have the same length")
    if len(x_vals) < 2:
        raise ValueError("At least 2 points are required for interpolation")

#=============================================================================
# SECTION 1: LAGRANGE INTERPOLATION EXERCISES
#=============================================================================

"""
Exercise 1: Quadratic Lagrange Interpolation
==========================================
Construct the quadratic Lagrange interpolating polynomial for the points:
(1,2), (3,4), (4,3)

Method: Lagrange interpolation polynomial of degree 2
"""

x_vals = [1, 3, 4]
y_vals = [2, 4, 3]

validate_inputs(x_vals, y_vals)
polynomial = Proyecto2.lagrange_interpolation(x_vals, y_vals)
f_lagrange1 = sp.lambdify('x', polynomial, 'numpy')

x = np.linspace(0,5,500)
y = f_lagrange1(x)

plt.plot(x,y, label = "Lagrange polynomial", color = "blue")
plt.scatter(x_vals, y_vals, color='red', label='Points', zorder=5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exercise 1: Quadratic Lagrange Interpolation")
plt.legend()
plt.grid()
plt.show()

"""
Exercise 2: Quadratic Interpolation
================================
Find the polynomial of degree 2 that interpolates at the points:
(0,1), (1,2), (4,2)

Method: Lagrange interpolation polynomial of degree 2
"""

x_vals = [0, 1, 4]
y_vals = [1, 2, 2]

validate_inputs(x_vals, y_vals)
polynomial = Proyecto2.lagrange_interpolation(x_vals, y_vals)
f_lagrange2 = sp.lambdify('x', polynomial, 'numpy')

x = np.linspace(-0.5, 4.5, 500)
y = f_lagrange2(x)

plt.plot(x, y, label="Lagrange Polynomial", color="blue")
plt.scatter(x_vals, y_vals, color='red', label='Points', zorder=5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exercise 2: Quadratic Interpolation")
plt.legend()
plt.grid()
plt.show()

"""
Exercise 3: Cubic Function Interpolation
====================================
Find the polynomial of degree 2 that interpolates to y = x³ at the nodes:
x₀ = 0, x₁ = 1, and x₂ = 2

Method: Lagrange interpolation polynomial comparing with original function
"""

def function_exercise3(x):
    """
    Cubic function for Exercise 3.
    
    Args:
        x (float): Input value
    Returns:
        float: x³
    """
    return x**3

x_vals = [0, 1, 2]
y_vals = [function_exercise3(x) for x in x_vals]

validate_inputs(x_vals, y_vals)
polynomial = Proyecto2.lagrange_interpolation(x_vals, y_vals)
f_lagrange3 = sp.lambdify('x', polynomial, 'numpy')

x = np.linspace(1/4, 5/4, 500)
y_lagrange = f_lagrange3(x)
y_function = function_exercise3(x)

plt.plot(x, y_lagrange, label="Lagrange Polynomial", color="blue")
plt.plot(x, y_function, label="x³", color="green", linestyle='--')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exercise 3: Cubic Function Interpolation")
plt.legend()
plt.grid()
plt.show()

"""
Exercise 4: Reciprocal Function Interpolation
=========================================
Determine the Lagrange interpolation polynomial of degree two for f(x) = 1/x with:
x₀ = 2, x₁ = 2.75, and x₂ = 4
Use result to approximate f(3) = 1/3

Method: Lagrange interpolation polynomial comparing with original function
"""

def function_exercise4(x):
    """
    Reciprocal function for Exercise 4.
    
    Args:
        x (float): Input value
    Returns:
        float: 1/x
    """
    return 1/x

x_vals = [2, 2.75, 4]
y_vals = [function_exercise4(x) for x in x_vals]

validate_inputs(x_vals, y_vals)
polynomial = Proyecto2.lagrange_interpolation(x_vals, y_vals)
f_lagrange4 = sp.lambdify('x', polynomial, 'numpy')

x = np.linspace(1.5, 4.5, 500)
y_lagrange = f_lagrange4(x)
y_function = function_exercise4(x)

plt.plot(x, y_lagrange, label="Lagrange Polynomial", color="blue")
plt.plot(x, y_function, label="1/x", color="green", linestyle='--')
plt.scatter(3, 1/3, color='red', label='f(3)', zorder=5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exercise 4: Reciprocal Function Interpolation")
plt.legend()
plt.grid()
plt.show()

#=============================================================================
# SECTION 2: NUMERICAL DIFFERENTIATION EXERCISES
#=============================================================================

"""
Exercise 5: Forward Difference Method
=================================
Calculate numerical derivative of f(x) = ln(x) at x = 1.8
using forward difference with step sizes h = 0.1, 0.05, 0.01

Method: Forward difference approximation
"""

def function_exercise5(x):
    """
    Natural logarithm function for Exercise 5.
    
    Args:
        x (float): Input value
    Returns:
        float: ln(x)
    """
    return np.log(x)

x = 1.8
h = [0.1, 0.05, 0.01]
exact_derivative = 1/x

derivatives = []
errors = []
error_bounds = []

for i in h:
    numerical_derivative = Proyecto2.forward(function_exercise5, x, i)
    derivatives.append(numerical_derivative)
    errors.append(abs(numerical_derivative - exact_derivative))
    error_bound = i/(2 * 1.8**2)
    error_bounds.append(error_bound)

# Prepare data for table
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')

table_data = []
for i in range(len(h)):
    table_data.append([
        f'{h[i]:.2f}',
        f'{derivatives[i]:.6f}',
        f'{errors[i]:.2e}',
        f'{error_bounds[i]:.2e}',
    ])

table = ax.table(
    cellText=table_data,
    colLabels=['Step size (h)', 'Numerical', 'Error', 'Error Bound'],
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

plt.title(f'Forward Difference Results for ln(x) at x = {x}\nExact Derivative = {exact_derivative:.6f}', pad=20)

plt.tight_layout()
plt.show()

"""
Exercise 6: First Order Formulas Comparison
=======================================
Approximate the derivative of f(x) = ln(x) at x₀ = 1.8 using first order formulas
with h = 0.1, 0.5, 0.01, 0.001, 0.0001
Determine the error limits and compare results

Method: Forward, backward, and central difference approximations
"""

def function_exercise6(x):
    """
    Natural logarithm function for Exercise 6.
    
    Args:
        x (float): Input value
    Returns:
        float: ln(x)
    """
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
    fwd = Proyecto2.forward(function_exercise6, x0, i)
    bwd = Proyecto2.backward(function_exercise6, x0, i)
    cnt = Proyecto2.central(function_exercise6, x0, i)
    
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
Exercise 7: Derivative of f(x) = x³ * 4x
=======================================
Approximate the derivative of f(x) = x³ * 4x using forward difference
with equally spaced nodes {xi} separated by h = 0.1
Compare the numerical derivative with the exact derivative

Method: Forward difference approximation, comparison with exact derivative
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

#=============================================================================
# SECTION 3: NUMERICAL INTEGRATION EXERCISES
#=============================================================================

"""
Exercise 8: Simple Trapezoidal and Simpson's Rule
==============================================
Approximate the integrals of the following functions from 0 to 2
using the single Trapezoidal and Simpson’s rule:
(a) f(x) = x²
(b) f(x) = x⁴
(c) f(x) = 1/(x+1)
(d) f(x) = √(1 + x²)
(e) f(x) = sin(x)
(f) f(x) = e^x

Method: Simple Trapezoidal and Simpson's rule
"""

a = 0
b = 2

def f8_a(x):
    return x**2

def f8_b(x):
    return x**4

def f8_c(x):
    return 1/(x + 1)

def f8_d(x):
    return np.sqrt(1 + x**2)

def f8_e(x):
    return np.sin(x)

def f8_f(x):
    return np.exp(x)

functions = [
    (f8_a, 2.667, "x²"),
    (f8_b, 6.400, "x⁴"),
    (f8_c, 1.099, "1/(x+1)"),
    (f8_d, 2.958, "√(1 + x²)"),
    (f8_e, 1.416, "sin(x)"),
    (f8_f, 6.389, "e^x")
]

trapezoidal_results = []
simpson_results = []

for f, exact, name in functions:
    trapezoidal_results.append(Proyecto2.trapezoidal_simple(f, a, b))
    simpson_results.append(Proyecto2.simpson_simple(f, a, b))

data = []
for i in range(len(functions)):
    data.append([
        f'{functions[i][2]}',
        f'{trapezoidal_results[i]:.4f}',
        f'{simpson_results[i]:.4f}',
        f'{functions[i][1]:.3f}'
    ])

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')  # Hide axes

# Create table
table = ax.table(
    cellText=data,
    colLabels=['f(x)', 'Trapezoidal Simple', 'Simpson Simple', 'Exact'],
    loc='center',
    cellLoc='center'
)

# Customize table appearance
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Add title
plt.title('Integration Results (from 0 to 2) with simple trapezoidal and simpson', pad=20)

plt.tight_layout()
plt.show()

"""
Exercise 9: Composite Trapezoidal and Simpson's Rule
================================================
Approximate the integrals of the functions from Exercise 8 from 0 to 2
using the Composite Trapezoidal and Simpson’s rule.

Method: Composite Trapezoidal and Simpson's rule with n = 4
"""
n = 4

trapezoidal_results = []
simpson_results = []

for f, exact, name in functions:
    trapezoidal_results.append(Proyecto2.trapezoidal(f, a, b, n))
    simpson_results.append(Proyecto2.simpson(f, a, b, n))

data = []
for i in range(len(functions)):
    data.append([
        f'{functions[i][2]}',
        f'{trapezoidal_results[i]:.4f}',
        f'{simpson_results[i]:.4f}',
        f'{functions[i][1]:.3f}'
    ])

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')  # Hide axes

# Create table
table = ax.table(
    cellText=data,
    colLabels=['f(x)', 'Trapezoidal', 'Simpson', 'Exact'],
    loc='center',
    cellLoc='center'
)

# Customize table appearance
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Add title
plt.title('Integration Results (from 0 to 2) with composite trapezoidal and simpson\nn = 4', pad=20)

plt.tight_layout()
plt.show()

"""
Exercise 10: Riemann Sum
========================
Approximate the integrals of the functions from Exercise 8 from 0 to 2
using the Riemann sum method.

Method: Riemann sum with n = 4
"""

riemann_results = []

for f,exact,name in functions:
    riemann_results.append(Proyecto2.riemann_sum(f, a, b, n))

data = []
for i in range(len(functions)):
    data.append([
        f'{functions[i][2]}',
        f'{riemann_results[i]:.4f}',
        f'{functions[i][1]:.3f}'
    ])

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')  # Hide axes

# Create table
table = ax.table(
    cellText=data,
    colLabels=['f(x)', 'Riemann', 'Exact'],
    loc='center',
    cellLoc='center'
)

# Customize table appearance
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Add title
plt.title('Integration Results (from 0 to 2) with Riemann Sum method\nn = 4', pad=20)

plt.tight_layout()
plt.show()

#=============================================================================
# SECTION 4: PRACTICAL PROBLEMS
#=============================================================================

"""
----------------及ひractical Problems------------------------------
Problem 1: Circuit Analysis
==========================
Analysis of an electrical circuit with:
- Inductance (L) = 0.98 H
- Resistance (R) = 0.142 Ω
Using numerical differentiation to find E(t) = L*di/dt + R*i
"""
# datos
t = np.array([1.00, 1.01, 1.02, 1.03, 1.04])
i = np.array([3.10, 3.12, 3.14, 3.18, 3.24])
h = 0.01
L = 0.98
R = 0.142

# derivada usando forward difference
di_dt = (i[1:] - i[:-1]) / h

# para E(t) solo podemos calcular en los primeros 4 valores (no en el ultimo)
t_di = t[:-1]  # tiempos donde tenemos derivada
i_di = i[:-1]  # corriente en esos tiempos

# calculo de E(t)
E = L * di_dt + R * i_di

# Create figure and axis for the table
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')  # Hide axes

# Prepare data for the table
table_data = []
for ti, ii, didti, Ei in zip(t_di, i_di, di_dt, E):
    table_data.append([f'{ti:.2f}', f'{ii:.2f}', f'{didti:.2f}', f'{Ei:.2f}'])

# Create table
table = ax.table(
    cellText=table_data,
    colLabels=['t', 'i(t)', 'di/dt', 'E(t)'],
    loc='center',
    cellLoc='center'
)

# Customize table appearance
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

plt.title('Circuit Analysis Results', pad=20)
plt.tight_layout()
plt.show()

"""
Problem 2: Distance Calculation
==============================
Calculate total distance traveled based on speed measurements
using composite trapezoidal rule for numerical integration.
"""

# Given data
time_intervals = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
speeds = np.array([124, 134, 148, 156, 147, 133, 121, 109, 99, 85, 78, 89, 104, 116, 123])

# Step size
h = 6

# Composite trapezoidal rule
def composite_trapezoidal_rule(time, speed):
    n = len(time) - 1  # Number of intervals
    result = speed[0] + speed[-1] + 2 * sum(speed[1:n])
    result *= h / 2
    return result

# Calculate total distance
total_distance = composite_trapezoidal_rule(time_intervals, speeds)
print("Practical problem 2 results:")
print(f"The total distance traveled is approximately {total_distance:.2f} feet.")

"""
Problem 3: Normal Distribution
=============================
Calculate probabilities for standard normal distribution
using Simpson's rule for numerical integration.
"""

# Define the function to integrate: e^(-z^2 / 2)
def gaussian_function(z):
    return np.exp(-z**2 / 2)

# Compute P(-mσ ≤ x ≤ mσ) for m = 1, 2, 3
m_values = [1, 2, 3]
results = []

for m in m_values:
    a, b = -m, m  # Integration limits
    n = 100  # Number of subintervals (even)
    integral = Proyecto2.simpson(gaussian_function, a, b, n)
    P_m = (1 / np.sqrt(2 * np.pi)) * integral  # Multiply by 1/sqrt(2π)
    results.append((m, P_m))

# Display results
print("Practical problem 3 results:")
for m, P in results:
    print(f"P(-{m}σ ≤ x ≤ {m}σ) ≈ {P:.6f}")
"""
Practical Problem 1: Circuit Analysis
===================================
Analysis of an electrical circuit with:
- Inductance (L) = 0.98 H
- Resistance (R) = 0.142 Ω
Using numerical differentiation to find E(t) = L*di/dt + R*i

Practical Problem 2: Distance Calculation
=======================================
Calculate total distance traveled based on speed measurements
using composite trapezoidal rule for numerical integration.

Practical Problem 3: Normal Distribution
=====================================
Calculate probabilities for standard normal distribution
using Simpson's rule for numerical integration.
"""
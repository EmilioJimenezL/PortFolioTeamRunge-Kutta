"""
Numerical Analysis: Calculus Interpolation, Derivation and Integration
Team Participants:
Emilio Ivan Jimenez Lopez
Brandon
Andre
Javi
"""

import numpy as np
import matplotlib.pyplot as plt
from Proyecto2 import Lagrange

"""
Excercise 1:
Construct the quadratic Lagrange interpolating polynomial for the following data points:
(1,2), (3,4), (4,3)
"""

def f_lagrange(x):
    return -(2/3)*x**2 + (11/3)*x - 1

x = np.linspace(0,5,500)
y = f_lagrange(x)

plt.plot(x,y, label = "y = lagrange polynomial", color = "blue")

# Add the points used for interpolation
points_x = [1, 3, 4]
points_y = [2, 4, 3]
plt.scatter(points_x, points_y, color='red', label='Points', zorder=5)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Lagrange polynomial")
plt.legend()
plt.grid()
plt.show()
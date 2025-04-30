#paquetes usados
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

#metodo simpson
def simpson(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even to apply Simpson's 1/3 rule.")
    h = (b - a) / n
    f_sum = 0
    for i in range(1, n, 2):
        x = a + i * h
        f_sum += 4 * f(x)
    for i in range(2, n - 1, 2):
        x = a + i * h
        f_sum += 2 * f(x)
    return h / 3 * (f(a) + f_sum + f(b))

def simpson_3_8(f, a, b):
    h = b - a
    c = a + h
    d = a + (2*h)
    return (3 * h/8) * (f(a) + 3 * f(c) + 3 * f(d) + f(b))

def simpson_simple(f, a, b):
    h = (b - a) / 2
    c = (a + b) / 2
    return (h / 3) * (f(a) + 4 * f(c) + f(b))

#Trapezoidal

def trapezoidal(f, a, b, n):
    h = (b - a) / n
    f_sum = 0
    for i in range(1, n, 1):
        x = a + i * h
        f_sum += f(x)
    return h * (0.5 * f(a) + f_sum + 0.5 * f(b))

def trapezoidal_simple(f, a, b):
    h = b - a
    return h * (0.5 * f(a) + (3/8) * f(b))


#Aplicaciones

def application():
    from math import exp
    v = lambda t: 3 * (t ** 2) * exp(t ** 3)
    numerical = simpson_3_8(v, 0, 1)
    # Compare with exact result
    V = lambda t: exp(t ** 3)
    exact = V(1) - V(0)
    error = abs(exact - numerical)
    print("{:.16f}, error: {:g}".format(numerical, error))
    
def application_Simpson_simple():
    from math import exp 
    v = lambda t: 3 * (t ** 2) * exp(t ** 3)
    numerical = simpson_simple(v, 0, 1)
    V = lambda t: exp(t ** 3)
    exact = V(1) - V(0)
    error = abs(exact - numerical)
    print("Simpson 1/3 simple {:.16f}, error: {:g}".format(numerical, error))
    
def application_trapezoidal():
    from math import exp
    v = lambda t: 3 * (t ** 2) * exp(t ** 3)
    n = int(input("n: "))
    numerical = trapezoidal(v, 0, 1, n)
    # Compare with exact result
    V = lambda t: exp(t ** 3)
    exact = V(1) - V(0)
    error = abs(exact - numerical)
    print("n={:d}: {:.16f}, error: {:g}".format(n, numerical, error))

def application_trapezoidal_simple():
    from math import exp
    v = lambda t: 3 * (t ** 2) * exp(t ** 3)
    numerical = trapezoidal_simple(v, 0, 1)
    # Compare with exact result
    V = lambda t: exp(t ** 3)
    exact = V(1) - V(0)
    error = abs(exact - numerical)
    print("{:.16f}, error: {:g}".format(numerical, error))


#Polinomios lagrange

def forward(f, x0, increment): # FORWARD EULER
    dydx_num = (f(x0+increment)-f(x0))/increment
    return dydx_num

def backward(f, x0, decrement):  # BACKWARD EULER
    dydx_num = (f(x0 + decrement) - f(x0)) / decrement
    return dydx_num

def central(f, x0, h):  # CENTRAL
    dydx_num = (f(x0 + h) - f(x0 - h)) / (2 * h)
    return dydx_num

def exact(Df, x0):
    # Exact/Analytical Solution
    dydx_exact = Df(x0)
    return dydx_exact

def Lagrange(x0,increment):
    x1 = x0 + increment
    x = np.array([x0, x1])
    print("x: ", x)
    y = np.log(x)
    print("y: ", y)
    
    # Exact/Analytical Solution
    dydx_exact = 1/x0
    print("dydx_exact: ", dydx_exact)
    
    # Numerical Solution
    dydx_num = np.diff(y)/np.diff(x)
    print("dydx_num: ", dydx_num)
    
    def forward(f, x0, increment): # FORWARD EULER
        dydx_num = (f(x0+increment)-f(x0))/increment
        return dydx_num
    
    def backward(f, x0, decrement): # BACKWARD EULER
        dydx_num = (f(x0+decrement)-f(x0))/decrement
        return dydx_num
    
    def central(f, x0, h): # CENTRAL
        dydx_num = (f(x0+h)-f(x0-h))/(2*h)
        return dydx_num
    
    def exact(Df, x0):
        # Exact/Analytical Solution
        dydx_exact = Df(x0)
        return dydx_exact
    
    def plot_resultsLabels(x0, f, dydx_exact, dydx_num):
        y0 = f(x0)
        # Dominio para graficar la función
        x_vals = np.linspace(x0-1, x0+1, 100)
        y_vals = f(x_vals)
        # Recta tangente (usamos la derivada exacta)
        tangent_line = dydx_exact * (x_vals - x0) + y0
        # Recta tangente (usamos derivada numérica)
        tangent_lineNum = dydx_num * (x_vals - x0) + y0
        fig = plt.figure()
        fig.suptitle('Aproximacion de la derivada', fontsize = 12)
        plt.plot(x_vals, y_vals, 'b', label = f'Función original')
        plt.plot(x0, y0, 'o', color='red')
        plt.plot(x_vals, tangent_line, 'g--', label = f'Recta tangente exacta: {round(dydx_exact, 4)}')
        plt.plot(x_vals, tangent_lineNum, 'r--', label = f'Recta tangente aproximada: {round(dydx_num, 4)}')
        plt.grid()  
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        return None
    
    ##### FUNCTION CALLING #####
    f = lambda X: np.log(X)
    Df = lambda x: 1/x
    x0
    increment
    dydx_exact = exact(Df, x0)
    
    # Derivada numérica
    
    dydx_num = forward(f, x0, increment)
    
    plot_resultsLabels(x0, f, dydx_exact, dydx_num)

def lagrange_interpolation(x_vals, y_vals):
    x = sp.Symbol('x')
    n = len(x_vals)
    p = 0
    for i in range(n):
        L_i = 1
        for j in range(n):
            if i != j:
                L_i *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        p += y_vals[i] * L_i
    return sp.simplify(p)
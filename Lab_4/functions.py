import numpy as np
from numpy import exp, sin, cos, sqrt, pi, array, max
import matplotlib.pyplot as plt
import pandas as pd

def u(x, y):
    return exp(1 - x**2 - y**2)

def f(x, y):
    return -abs((sin(pi*x*y))**3)

def ft(x, y):
    return -4*u(x, y)*(x**2 + y**2 - 1)

def mu1_2(x, y):
    return 1 - y**2

def mu3_4(x, y):
    return abs(sin(pi*x))

def f_lam1(h, k, n, m):
	return (4*sin(pi/(2*n))**2)/(h**2) + (4*sin(pi/(2*m))**2)/(k**2)

def f_lamn(h, k, n, m):
    return (4*cos((pi)/(2*n))**2)/(h**2) + (4*cos((pi)/(2*m))**2)/(k**2)

def tau(S, k, lam1, lamn):
	s = S % k
	return 2/((lam1 + lamn) + (lamn - lam1)*cos((pi*(2*s + 1))/(2*k)))

def graf(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(x, y)
    z = array(z)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='inferno')
    plt.show()

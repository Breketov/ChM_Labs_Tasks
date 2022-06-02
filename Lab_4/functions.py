import numpy as np
from numpy import exp, sin, cos, sqrt, pi, array, max, abs
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
	return 2/((lam1 + lamn) + (lamn - lam1)*cos((pi*(2*s - 1))/(2*k)))

def graf_std(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(x, y)
    z = array(z)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='inferno')
    plt.show()

def graf_nst_Nan(a, c, h, k, m, n, Z):
    x1 = [a + i*h for i in range(0, int(n/2) + 1)]
    y1 = [c + j*k for j in range(0, int(m/2) + 1)]
    x2 = [a + i*h for i in range(0, n + 1)]
    y2 = [c + j*k for j in range(int(m/2), m + 1)]

    z1, z2 = [], []
    for i in range(0, int(m/2) + 1):
        if i == int(m/2):
            z1.append([])
            for j in range(0, int(n/2) + 1):
                z1[i].append(Z[i][j])
        else:
            z1.append(Z[i])
    for i in range(int(m/2), m + 1):
        z2.append(Z[i])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1, y1 = np.meshgrid(x1, y1)
    x2, y2 = np.meshgrid(x2, y2)
    z1 = array(z1)
    z2 = array(z2)
    ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap='inferno')
    ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap='inferno')
    plt.show()

def graf_nst_0(a, c, h, k, m, n, Z):
    x1 = [a + i*h for i in range(0, n + 1)]
    y1 = [c + j*k for j in range(0, m + 1)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1, y1 = np.meshgrid(x1, y1)
    z1 = array(Z)
    ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap='inferno')
    plt.show()
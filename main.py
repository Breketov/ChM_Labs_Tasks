from typing import ValuesView
import numpy as np
from numpy.core.function_base import linspace
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math


n = 5
step = 0.01
x = 0
v = 2

class zadachi():
    def __init__(self, x, u) -> None:
        self.x = x
        self.u = u
        pass
    def test(x, u):
        dudx = -1* 3/2 * u
        return dudx
    
    def osnovnay1(x, u):
        dudx = 1/((1+(x**2))**(1/3))*(u**2) + u - (u**3)*math.sin(10*x)
        return dudx

    """ def osnovnaya2(x, u, u0, u_0): """





for i in range(1, n+1):
    k1 = zadachi.test(x, v)
    k2 = zadachi.test(x + step / 2.0, v + 0.5 * step * k1)
    k3 = zadachi.test(x + step / 2.0, v + 0.5 * step * k2)
    k4 = zadachi.test(x + step, v + step * k3)
    x = x + step
    v = v + step/6 * (k1 + 2*k2 + 2*k3 + k4)
    print(x, '   ', v)

print('___________________________________________')

n = 5
step = 0.01
x = 0
v = 2

for i in range(1, n+1):
    k1 = zadachi.osnovnay1(x, v)
    k2 = zadachi.osnovnay1(x + step / 2.0, v + 0.5 * step * k1)
    k3 = zadachi.osnovnay1(x + step / 2.0, v + 0.5 * step * k2)
    k4 = zadachi.osnovnay1(x + step, v + step * k3)
    x = x + step
    v = v + step/6 * (k1 + 2*k2 + 2*k3 + k4)
    print(x, '   ', v)

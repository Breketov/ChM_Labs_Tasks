import numpy as np
from numpy.core.function_base import linspace
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math


class zadachi():
    def __init__(self) -> None:
        pass
    def test(x, u):
        dudx = -1* 3/2 * u
        return dudx
    
    def osnovnaya1(x, u):
        dudx = 1/((1+(x**2))**(1/3))*(u**2) + u - (u**3)*math.sin(10*x)
        return dudx

    def osnovnaya2(x, u):
        du_ = u[1]
        dx_= -a* (u[1]**2) - b * u[0]
        return [du_, dx_] 


n = 5
step = 0.01
x = 0
v = 2
a = 3
b = 2
 
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
a = 3
b = 2

for i in range(1, n+1):
    k1 = zadachi.osnovnaya1(x, v)
    k2 = zadachi.osnovnaya1(x + step / 2.0, v + 0.5 * step * k1)
    k3 = zadachi.osnovnaya1(x + step / 2.0, v + 0.5 * step * k2)
    k4 = zadachi.osnovnaya1(x + step, v + step * k3)
    x = x + step
    v = v + step/6 * (k1 + 2*k2 + 2*k3 + k4)
    print(x, '   ', v)


print('___________________________________________')


""" 

n = 5
step = 0.01
x = 0.0
a = 3
b = 2
v = [3.0, 2.0]

for i in range(1, n+1):
    k2 = []
    k3 = []
    k4 = []
    k1 = zadachi.osnovnaya2(x, v)
    k2 = zadachi.osnovnaya2(x + step / 2.0, v + 0.5 * step * k1)
    k3 = zadachi.osnovnaya2(x + step / 2.0, v + 0.5 * step * k2)
    k4 = zadachi.osnovnaya2(x + step, v + step * k3)
    x = x + step
    v = v + step/6 * (k1 + 2*k2 + 2*k3 + k4)
    print(x, '   ', v[0], '   ', v[1])

 """
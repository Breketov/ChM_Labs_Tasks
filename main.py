import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class zadachi():
    def __init__(self) -> None:
        pass

    def test(x, u):
        dudx = -1* 3/2 * u
        return dudx
    
    def osnov1(x, u):
        dudx = 1/((1+(x**2))**(1/3))*(u**2) + u - (u**3)*math.sin(10*x)
        return dudx

    def osnov2(x, u):
        du_ = u[1]
        dx_= -a* (u[1]**2) - b * u[0]
        return [du_, dx_] 

class RK4(zadachi):
    def __init__(self) -> None:
        pass

    def RK4_test(x, v, step, n):
        print(x, '   ', v)
        for i in range(1, n+1):
            k1 = zadachi.test(x, v)
            k2 = zadachi.test(x + step / 2, v + 0.5 * step * k1)
            k3 = zadachi.test(x + step / 2, v + 0.5 * step * k2)
            k4 = zadachi.test(x + step, v + step * k3)

            x = x + step
            v = v + step/6 * (k1 + 2*k2 + 2*k3 + k4)
            print(x, '   ', v)
            return v

    def RK4_osnov1(x, v, step, n):
        print(x, '   ', v)
        for i in range(1, n+1):
            k1 = zadachi.osnov1(x, v)
            k2 = zadachi.osnov1(x + step / 2, v + 0.5 * step * k1)
            k3 = zadachi.osnov1(x + step / 2, v + 0.5 * step * k2)
            k4 = zadachi.osnov1(x + step, v + step * k3)

            x = x + step
            v = v + step/6 * (k1 + 2*k2 + 2*k3 + k4)
            print(x, '   ', v)

    def RK4_osnov2(x, v, step, n):
        print(x, '   ', v)
        for i in range(1, n+1):
            v_ = np.array(v)
            k1 = zadachi.osnov2(x, v_)
            k1_ = np.array(k1)
            k2 = zadachi.osnov2(x + step / 2, v_ + 0.5 * step * k1_)
            k2_ = np.array(k2)
            k3 = zadachi.osnov2(x + step / 2, v_ + 0.5 * step * k2_)
            k3_ = np.array(k3)
            k4 = zadachi.osnov2(x + step, v_ + step * k3_)
            k4_ = np.array(k3)

            x = x + step
            v = v_ + step/6 * (k1_ + 2*k2_ + 2*k3_ + k4_)
            print(x, '   ', v[0], '   ', v[1])


a = 3
b = 2



""" 
c = RK4.RK4_osnov2(x0, v0, step, n)
print(c)

d = RK4.RK4_osnov1(0, 2, 0.01, 5)
print(d)
"""
""" 
b = RK4.RK4_test(0, 2, 0.01, 5)
 """

def test(x, u):
    dudx = -1* 3/2 * u
    return dudx

u0 = 0
x = np.linspace(0, 0.05, 6)
u = odeint(test, u0, x)
u = np.array(u).flatten()
plt.plot(x, u, 'o-', linewidth = 2.0)
plt.show()


RK4.RK4_test

plt.plot()
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
        return du_, dx_ 

class RK4(zadachi):
    def __init__(self) -> None:
        pass

    def RK4_test(x, v, step, n):
        x_ = []
        v_ = []
        print(x, '   ', v)
        for i in range(1, n+1):
            k1 = zadachi.test(x, v)
            k2 = zadachi.test(x + step / 2, v + 0.5 * step * k1)
            k3 = zadachi.test(x + step / 2, v + 0.5 * step * k2)
            k4 = zadachi.test(x + step, v + step * k3)

            x = x + step
            v = v + step/6 * (k1 + 2*k2 + 2*k3 + k4)

            x_.append(x)
            v_.append(v)
            print(x, '   ', v)
        return x_, v_

    def RK4_osnov1(x, v, step, n):
        x_ = []
        v_ = []
        print(x, '   ', v)
        for i in range(1, n+1):
            k1 = zadachi.osnov1(x, v)
            k2 = zadachi.osnov1(x + step / 2, v + 0.5 * step * k1)
            k3 = zadachi.osnov1(x + step / 2, v + 0.5 * step * k2)
            k4 = zadachi.osnov1(x + step, v + step * k3)

            x = x + step
            v = v + step/6 * (k1 + 2*k2 + 2*k3 + k4)

            x_.append(x)
            v_.append(v)
            print(x, '   ', v)
        return x_, v_

    def RK4_osnov2(x, v, step, n):
        x_ = []
        v1_ = []
        v2_ = []
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
            k4_ = np.array(k4)

            x = x + step
            v = v_ + step/6 * (k1_ + 2*k2_ + 2*k3_ + k4_)

            x_.append(x)
            v1_.append(v[0])
            v2_.append(v[1])
            print(x, '   ', v[0], '   ', v[1])
        return x_, v1_, v2_


a = 3
b = 2


""" 
v0 = []
x0 = float(input())
v0_0 = float(input())
v0.append(v0_0)
v0_1 = float(input())
v0.append(v0_1)
step = float(input())
n = int(input())

c = RK4.RK4_osnov2(x0, v0, step, n)
 """


""" d = RK4.RK4_osnov1(0, 2, 0.01, 10)
print(d)
 """


""" b = RK4.RK4_test(0, 2, 0.01, 5)
print(b)
 """


#! Графики истинного решения тестовой задачи рисуется и поточечная траектория численного решения тоже существует и рисуется, но их разница незаметна
#! Текущие точки численной траектории возвращаются




""" 
u0 = 2
x = np.linspace(0, 0.05, 6)
i = 0
u_test = []
while i < len(x):
   u_ = u0 * math.exp(-3/2 * x[i])
   u_test.append(u_)
   i = i + 1


e = RK4.RK4_test(0, 2, 0.01, 6)
e_x = e[0]
e_v = e[1]

plt.plot(e_x, e_v, 'r--', linewidth = 0.5)
plt.plot(x, u_test, 'b--', linewidth = 0.5)
plt.axis([0, 0.05, 0, 2])
plt.show() 
"""




""" 
g = RK4.RK4_osnov1(0, 2, 0.01, 6)
g_x = g[0]
g_v = g[1]

plt.plot(g_x, g_v, 'r--', linewidth = 0.5)
plt.show()
 """



""" 
l = RK4.RK4_osnov2(0, [3, 2], 0.01, 6)
l_x = l[0]
l_v1 = l[1]
l_v2 = l[2]

plt.plot(l_v1, l_v2, 'r--', linewidth = 0.5)
plt.show() 
"""
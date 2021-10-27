import numpy as np
import math
import matplotlib.pyplot as plt


class zadachi():
    def __init__(self) -> None:
        pass

    def test(x, u):
        dudx = -1* 3/2 * u
        return dudx
    
    def o1(x, u):
        dudx = 1/((1+(x**2))**(1/3))*(u**2) + u - (u**3)*math.sin(10*x)
        return dudx

    def o2(x, u):
        du_ = u[1]
        dx_= -a* (u[1]**2) - b * u[0]
        return du_, dx_ 

class RK4(zadachi):
    def __init__(self) -> None:
        super().__init__()

    def RK4_test(x, v, step, n):
        x1 = x
        v1 = v
        x_1 = []
        v_1 = []
        x2 = x
        v2 = v
        x_2 = []
        v_2 = []

        print(x1, '   ', v1)
        for i in range(0, n+1):
            k1 = zadachi.test(x1, v1)
            k2 = zadachi.test(x1 + step / 2, v1 + 0.5 * step * k1)
            k3 = zadachi.test(x1 + step / 2, v1 + 0.5 * step * k2)
            k4 = zadachi.test(x1 + step, v1 + step * k3)
            x_1.append(x1)
            v_1.append(v1)

            x1 = x1 + step
            v1 = v1 + step/6 * (k1 + 2*k2 + 2*k3 + k4)
            print(x1, '   ', v1)
        
        print(x2, '   ', v2)
        for i in range(0, 2*(n + 1)):
            k1_2 = zadachi.test(x2, v2)
            k2_2 = zadachi.test(x2 + step / 4, v2 + 0.25 * step * k1_2)
            k3_2 = zadachi.test(x2 + step / 4, v2 + 0.25 * step * k2_2)
            k4_2 = zadachi.test(x2 + step / 2, v2 + 0.5 * step * k3_2)
            x_2.append(x2)
            v_2.append(v2)

            x2 = x2 + step / 2
            v2 = v2 + step/12 * (k1 + 2*k2_2 + 2*k3_2 + k4_2)
            print(x2, '   ', v2)

        for i in range(0, n+1):
            v__1 = np.array(v_1)
            v__2 = np.array(v_2)
            S = v__1[i] - v__2[2*i]

            print(S)
        return x_1, v_1, x_2, v_2, S

    def RK4_o1(x, v, step, n):
        x1 = x
        v1 = v
        x_1 = []
        v_1 = []
        x2 = x
        v2 = v
        x_2 = []
        v_2 = []

        print(x1, '   ', v1)
        for i in range(0, n+1):
            k1 = zadachi.o1(x1, v1)
            k2 = zadachi.o1(x1 + step / 2, v1 + 0.5 * step * k1)
            k3 = zadachi.o1(x1 + step / 2, v1 + 0.5 * step * k2)
            k4 = zadachi.o1(x1 + step, v1 + step * k3)
            x_1.append(x1)
            v_1.append(v1)

            x1 = x1 + step
            v1 = v1 + step/6 * (k1 + 2*k2 + 2*k3 + k4)
            print(x1, '   ', v1)

        print(x2, '   ', v2)
        for i in range(0, 2*(n + 1)):
            k1_2 = zadachi.o1(x2, v2)
            k2_2 = zadachi.o1(x2 + step / 4, v2 + 0.25 * step * k1_2)
            k3_2 = zadachi.o1(x2 + step / 4, v2 + 0.25 * step * k2_2)
            k4_2 = zadachi.o1(x2 + step / 2, v2 + 0.5 * step * k3_2)
            x_2.append(x2)
            v_2.append(v2)

            x2 = x2 + step / 2
            v2 = v2 + step/12 * (k1 + 2*k2_2 + 2*k3_2 + k4_2)
            print(x2, '   ', v2)

        for i in range(0, n+1):
            v__1 = np.array(v_1)
            v__2 = np.array(v_2)
            S = v__1[i] - v__2[2*i]
            
            print(S)
        return x_1, v_1, x_2, v_2

    def RK4_o2(x, v, step, n):
        x1 = x
        v1 = v
        x_1 = []
        v1_1 = []
        v2_1 = []

        x2 = x
        v2 = v
        x_2 = []
        v1_2 = []
        v2_2 = []

        print(x1, '   ', v1[0], '   ', v1[1])
        for i in range(0, n+1):
            v_1 = np.array(v1)
            k1 = zadachi.o2(x1, v_1)
            k1_ = np.array(k1)
            k2 = zadachi.o2(x1 + step / 2, v_1 + 0.5 * step * k1_)
            k2_ = np.array(k2)
            k3 = zadachi.o2(x1 + step / 2, v_1 + 0.5 * step * k2_)
            k3_ = np.array(k3)
            k4 = zadachi.o2(x1 + step, v_1 + step * k3_)
            k4_ = np.array(k4)
            x_1.append(x1)
            v1_1.append(v1[0])
            v2_1.append(v1[1])

            x1 = x1 + step
            v1 = v_1 + step/6 * (k1_ + 2*k2_ + 2*k3_ + k4_)
            print(x1, '   ', v1[0], '   ', v1[1])
        
        print(x2, '   ', v2[0], '   ', v2[1])
        for i in range(0, 2*(n + 1)):
            v_2 = np.array(v2)
            k1 = zadachi.o2(x2, v_2)
            k1_ = np.array(k1)
            k2 = zadachi.o2(x2 + step / 2, v_2 + 0.5 * step * k1_)
            k2_ = np.array(k2)
            k3 = zadachi.o2(x2 + step / 2, v_2 + 0.5 * step * k2_)
            k3_ = np.array(k3)
            k4 = zadachi.o2(x2 + step, v_2 + step * k3_)
            k4_ = np.array(k4)
            x_2.append(x2)
            v1_2.append(v2[0])
            v2_2.append(v2[1])

            x2 = x2 + step
            v2 = v_2 + step/6 * (k1_ + 2*k2_ + 2*k3_ + k4_)
            print(x1, '   ', v2[0], '   ', v2[1])

        for i in range(0, n+1):
            v1__1 = np.array(v1_1)
            v1__2 = np.array(v1_2)
            v2__1 = np.array(v2_1)
            v2__2 = np.array(v2_2)
            S1 = v1__1[i] - v1__2[2*i]
            S2 = v2__1[i] - v2__2[2*i]
            
            print(S1, '   ', S2)
        return x_1, v1_1, v2_1, x_2, v1_2, v2_2, S1, S2

a = 3
b = 2
epsilon = 0.0001


def control(x, v, v_, step, n, epsilon):
    test = RK4.RK4_test(x, v, step, n)
    osnov1 = RK4.RK4_osnov1
    S = v - v_
    if((epsilon / 32) <= abs(S) and abs(S) <= epsilon):
        step = step
        return x, v, v_, step
    if(abs(S) <= (epsilon / 32)):
        step = 2*step
        return x, v, v_, step
    if(abs(S) >= epsilon):
        step = step/2
        return x, v[i-1], v_, step

test = RK4.RK4_test(0, 2, 0.01, 5)



print("world")















































































#! Графики истинного решения тестовой задачи рисуется и поточечная траектория численного решения тоже существует и рисуется, но их разница незаметна
#! Текущие точки численной траектории возвращаются




""" 
u0 = 2
x = np.linspace(0, 0.05, 6)
i = 0
u_test = []
while i < len(x):
   u_ = u0 * math.exp(-3/2 * x[i])                     #! Проверить есть ли разница в math и np, в np.exp написано что может принимать массив
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
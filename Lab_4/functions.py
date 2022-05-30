import numpy as np
from numpy import exp, sin, cos, sqrt, pi, array
import matplotlib.pyplot as plt
import pandas as pd

#* Функции
def u(x, y):
    return exp(1 - x**2 - y**2)

def f(x, y):
    return -abs((sin(pi*x*y))**3)

def ft(x, y):
    return -4*u(x, y)*(x**2 + y**2 - 1)

#* ГУ
def mu1_2(x, y):
    return 1 - y**2

def mu3_4(x, y):
    return abs(sin(pi*x))

#* Параметры метода
def f_lam1(h, k, n, m):
	return 4*sin(pi/(2*n))**2/(h**2) + 4*sin(pi/(2*m))**2/(k**2)

def f_lamn(h, k, n, m):
    return 4*cos((pi)/(2*n))**2/(h**2) + 4*cos((pi)/(2*m))**2/(k**2)

def tau(S, k, lam1, lamn):
	s = S % k
	return 1/((lam1 + lamn)*0.5 + 0.5*((lamn - lam1)*cos(0.5*pi*(1 + 2*s)/k)))

#* Подсчет ошибок
def error_test(m, n, U, V, R):
    max_eps = 0
    nev = 0
    for i in range(0, m + 1):
        for j in range(0, n + 1):
            eps = abs(U[i][j] - V[i][j])
            if(eps >= max_eps):
                max_eps = eps
                error_i = i
                error_j = j
            nev += R[i][j]**2
    return sqrt(nev), R[1][1], [max_eps, error_i, error_j]

def error_main(m, n, V1, V2, R1, R2):
    max_eps = 0
    nev1, nev2 = 0, 0
    for i in range(0, m + 1):
        for j in range(0, n + 1):
            eps = abs(V1[i][j] - V2[2*i][2*j])
            if(eps >= max_eps):
                max_eps = eps
                error_i = i
                error_j = j
            nev1 += R1[i][j]**2
            nev2 += R2[i][j]**2
    return [sqrt(nev1), sqrt(nev2)], [R1[1][1], R2[1][1]], [max_eps, error_i, error_j]

#* Отрисовываем графики
def graf(x, y, u):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(x, y)
    u = array(u)
    ax.plot_surface(x, y, u, rstride=1, cstride=1, cmap='inferno')
    plt.show()

#* Решение тестовой
def solve_test(n , m, eps, Nmax, cheb):
    U, V, R, Z = [], [], [], []
    a, c = -1, -1
    b, d = 1, 1

    h = (b - a)/n
    k = (d - c)/m
    xi = [a + i*h for i in range(0, n + 1)]
    yj = [c + j*k for j in range(0, m + 1)]
    h2 = 1/(h**2)
    k2 = 1/(k**2)
    A = -2*(h2 + k2)

    lam1 = f_lam1(h, k, n, m)
    lamn = f_lamn(h, k, n, m)
    #* Заполняем массивы
    #-----------------------------------------#
    for i in range(0, m + 1):
        U.append([])
        for j in range(0, n + 1):
            U[i].append(0)
            U[i][j] = u(xi[j], yj[i])

    for i in range(0, m + 1):
        V.append([])
        R.append([])
        Z.append([])
        for j in range(0, n + 1):
            V[i].append(0)
            R[i].append(0)
            Z[i].append(0)

    #* Заполняем граничные условия
    #-----------------------------------------#
    for i in range(0, m + 1):
        V[i][0] = u(a, yj[i])
        V[i][n] = u(b, yj[i])
    for j in range(0, n + 1):
        V[0][j] = u(xi[j], c)
        V[m][j] = u(xi[j], d)

    #* Запускаем метод
    #-----------------------------------------#
    S, flag = 0, 0
    while (flag == 0):
        eps_max = 0 
        for i in range(1, m):
            for j in range(1, n):
                R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + ft(xi[j], yj[i])
        for i in range (1, m):
            for j in range (1, n):
                v_old = V[i][j]
                v_new = v_old + tau(S, cheb, lam1, lamn)*R[i][j]

                eps_cur = abs(v_old - v_new) 
                if(eps_cur > eps_max):
                    eps_max = eps_cur
                V[i][j] = v_new
        S = S + 1        
        if(eps_max < eps or S >= Nmax):
            flag = 1
            S = S - 1
    
    for i in range(0, m + 1):
        for j in range(0, n + 1):
            Z[i][j] = V[i][j] - U[i][j]
    err = error_test(m, n, U, V, R)
    graf(xi, yj, U)
    graf(xi, yj, V)
    graf(xi, yj, Z)
        
    data = pd.DataFrame(U)
    data.to_csv("data_u_test.csv", index=False)
    data = pd.DataFrame(V)
    data.to_csv("data_v_test.csv", index=False)
    data = pd.DataFrame(Z)
    data.to_csv("data_z_test.csv", index=False)
    
    print('Справка')
    print('Число итераций N: ', S)
    print('Точность на шаге N: ', eps_max)
    print('Невязка N: ', err[0])
    print('Невязка начального приближения: ', err[1])
    print('Максимальная погрешность: ', err[2][0], 'В точке: ', [round(xi[err[2][1]], 5), round(yj[err[2][2]], 5)])
    print('---------------------------------------------------------------------------------------------------')

#* Решение основной
def solve(n , m, eps, Nmax, cheb):
    V, R = [], []
    a, c = -1, -1
    b, d = 1, 1

    h = (b - a)/n
    k = (d - c)/m
    xi = [a + i*h for i in range(0, n + 1)]
    yj = [c + j*k for j in range(0, m + 1)]

    h2 = 1/(h**2)
    k2 = 1/(k**2)
    A = -2*(h2 + k2)

    lam1 = f_lam1(h, k, n, m)
    lamn = f_lamn(h, k, n, m)
    #* Заполняем массивы
    #-----------------------------------------#
    for i in range(0, m + 1):
        V.append([])
        R.append([])
        for j in range(0, n + 1):
            V[i].append(0)
            R[i].append(0)

    #* Заполняем граничные условия
    #-----------------------------------------#
    for i in range(0, m + 1):
        V[i][0] = mu1_2(a, yj[i])
        V[i][n] = mu1_2(b, yj[i])
    for j in range(0, n + 1):
        V[0][j] = mu3_4(xi[j], c)
        V[m][j] = mu3_4(xi[j], d)

    for i in range(0, m + 1):
        V[i][0] = u(a, yj[i])
        V[i][n] = u(b, yj[i])
    for j in range(0, n + 1):
        V[0][j] = u(xi[j], c)
        V[m][j] = u(xi[j], d)

    #* Запускаем метод
    #-----------------------------------------#
    S, flag = 0, 0
    while (flag == 0):
        eps_max = 0 
        for i in range(1, m):
            for j in range(1, n):
                R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + f(xi[j], yj[i])
        for i in range (1, m):
            for j in range (1, n):
                v_old = V[i][j]
                v_new = v_old + tau(S, cheb, lam1, lamn)*R[i][j]

                eps_cur = abs(v_old - v_new) 
                if(eps_cur > eps_max):
                    eps_max = eps_cur
                V[i][j] = v_new
        S = S + 1        
        if(eps_max < eps or S >= Nmax):
            flag = 1
            S = S - 1
    h = [xi, yj]

    return V, R, S, eps_max, h

def task_main(n , m, eps, Nmax, cheb):
    Z = []
    for i in range(0, m + 1):
        Z.append([])
        for j in range(0, n + 1):
            Z[i].append(0)

    V1, R1, S1, eps_max1, point1 = solve(n , m, eps, Nmax, cheb)
    V2, R2, S2, eps_max2, point2 = solve(int(2*n) , int(2*m), eps, Nmax, cheb)
    for i in range(0, m + 1):
        for j in range(0, n + 1):
            Z[i][j] = V1[i][j] - V2[2*i][2*j]

    data = pd.DataFrame(V1)
    data.to_csv("data_v1_main.csv", index=False)
    data = pd.DataFrame(V2)
    data.to_csv("data_v2_main.csv", index=False)
    data = pd.DataFrame(Z)
    data.to_csv("data_z_main.csv", index=False)

    err = error_main(m, n, V1, V2, R1, R2)
    x, y = point1[0], point1[1] 
    graf(x, y, V1)
    x, y = point2[0], point2[1] 
    graf(x, y, V2)
    x, y = point1[0], point1[1] 
    graf(x, y, Z)

    print('Справка')
    print('Число итераций основной сетки N: ', S1)
    print('Точность на шаге N: ', eps_max1)
    print('Невязка N обычной сетки: ', err[0][0])
    print('Невязка начального приближения: ', err[1][0])
    print('---------------------------------------------------------------------------------------------------')
    print('Число итераций контрольной сетки N: ', S2)
    print('Точность на шаге N: ', eps_max2)
    print('Невязка N контрольной сетки: ', err[0][1])
    print('Невязка начального приближения: ', err[1][1])
    print('')
    print('Максимальная разность решений: ', err[2][0], 'В точке: ',  [round(point1[0][err[2][1]], 5), round(point1[1][err[2][2]], 5)])
    print('---------------------------------------------------------------------------------------------------')
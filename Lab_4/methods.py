import functions as func
import numpy as np
import pandas as pd
a, b, c, d = -1, 1, -1, 1 

def Chebyshev(n, m, eps, Nmax, cheb, V, R, task):
    h = (b - a)/n
    k = (d - c)/m
    xi = [a + i*h for i in range(0, n + 1)]
    yj = [c + j*k for j in range(0, m + 1)]

    h2 = 1/(h**2)
    k2 = 1/(k**2)
    A = -2*(h2 + k2)

    lam1 = func.f_lam1(h, k, n, m)
    lamn = func.f_lamn(h, k, n, m)

    S = 0
    while(True):
        if (S == 1):
            nev0 = np.max(np.abs(R))

        eps_max = 0 
        for i in range(1, m):
            for j in range(1, n):
                if task == 0:
                    R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + func.ft(xi[j], yj[i])
                elif task == 1:
                    R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + func.f(xi[j], yj[i])
        for i in range (1, m):
            for j in range (1, n):
                v_old = V[i][j]
                v_new = v_old + func.tau(S, cheb, lam1, lamn)*R[i][j]
                    
                eps_cur = abs(v_old - v_new) 
                if(eps_cur > eps_max):
                    eps_max = eps_cur
                V[i][j] = v_new
        S += 1
        if(eps_max < eps or S >= Nmax - 1):
            if S/cheb == int(S/cheb):
                break
            else:
                continue
    return V, R, [eps_max, S, nev0]

def MVR(n, m, eps, Nmax, omega, V, R, task):
    h = (b - a)/n
    k = (d - c)/m
    xi = [a + i*h for i in range(0, n + 1)]
    yj = [c + j*k for j in range(0, m + 1)]

    h2 = 1/(h**2)
    k2 = 1/(k**2)
    A = -2*(h2 + k2)

    S = 0
    while(True):
        if (S == 1):
            nev0 = np.max(np.abs(R))
            
        eps_max = 0 
        for i in range(1, m):
            for j in range(1, n):
                if task == 0:
                    R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + func.ft(xi[j], yj[i])
                elif task == 1:
                    R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + func.f(xi[j], yj[i])
        for i in range (1, m):
            for j in range (1, n):
                v_old = V[i][j]
                v_new = -omega*(h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]))
                if task == 0:
                    v_new = v_new + (1 - omega)*A*V[i][j] - omega*func.ft(xi[j], yj[i]) 
                elif task == 1:
                    v_new = v_new + (1 - omega)*A*V[i][j] - omega*func.f(xi[j], yj[i]) 
                v_new = v_new/A
                    
                eps_cur = abs(v_old - v_new) 
                if(eps_cur > eps_max):
                    eps_max = eps_cur
                V[i][j] = v_new
        S += 1
        if(eps_max < eps or S >= Nmax - 1):
            break
    return V, R, [eps_max, S, nev0]

def MPI_standart(n, m, eps, Nmax, V, R, task):
    h = (b - a)/n
    k = (d - c)/m
    xi = [a + i*h for i in range(0, n + 1)]
    yj = [c + j*k for j in range(0, m + 1)]

    h2 = 1/(h**2)
    k2 = 1/(k**2)
    A = -2*(h2 + k2)

    lam1 = func.f_lam1(h, k, n, m)
    lamn = func.f_lamn(h, k, n, m)
    tau_MPI = 2/(lam1 + lamn)

    S = 0
    while(True):
        if (S == 1):
            nev0 = np.max(np.abs(R))

        eps_max = 0 
        for i in range(1, m):
            for j in range(1, n):
                if task == 0:
                    R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + func.ft(xi[j], yj[i])
                elif task == 1:
                    R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + func.f(xi[j], yj[i])
        for i in range (1, m):
            for j in range (1, n):
                v_old = V[i][j]
                v_new=v_old + tau_MPI*R[i][j]
                    
                eps_cur = abs(v_old - v_new) 
                if(eps_cur > eps_max):
                    eps_max = eps_cur
                V[i][j] = v_new
        S += 1
        if(eps_max < eps or S >= Nmax - 1):
            break
    return V, R, [eps_max, S, nev0, tau_MPI]

def MPI_non_standart(n, m, eps, Nmax):
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
    
    lam1 = func.f_lam1(h, k, n, m)
    lamn = func.f_lamn(h, k, n, m)
    tau_MPI = 2/(lam1 + lamn)

    #* Заполняем массивы
    #-----------------------------------------#
    for i in range(0, m + 1):
        U.append([])
        V.append([])
        R.append([])
        Z.append([])
        for j in range(0, n + 1):
            U[i].append(0)
            V[i].append(0)
            R[i].append(0)
            Z[i].append(0)

    for i in range(0, m + 1):
        for j in range(0, int(n/2) + 1):
            U[i][j] = func.u(xi[j], yj[i])

    for i in range(int(m/2), m + 1):
        for j in range(int(n/2) + 1, n + 1):
            U[i][j] = func.u(xi[j], yj[i])

    #* Заполняем граничные условия
    #-----------------------------------------#
    for i in range(0, m + 1):
        V[i][0] = func.u(a, yj[i])

    for i in range(0, int(m/2)):
        V[i][int(n/2)] = func.u(0, yj[i])

    for i in range(int(m/2), m + 1):
        V[i][n] = func.u(b, yj[i])

    for j in range(0, int(n/2)):
        V[0][j] = func.u(xi[j], c)

    for j in range(int(n/2), n + 1):
        V[int(m/2)][j] = func.u(xi[j], 0)

    for j in range(0, n + 1):
        V[m][j] = func.u(xi[j], d)

    #* Запускаем метод
    #-----------------------------------------#
    S = 0
    while (True):
        if (S == 1):
            nev0 = np.max(np.abs(R))

        eps_max = 0 
        for i in range(1, m):
            for j in range(1, int(n/2)):
                R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + func.ft(xi[j], yj[i])
        for i in range(int(m/2) + 1, m):
            for j in range(int(n/2), n):
                R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + func.ft(xi[j], yj[i])

        for i in range(1, m):
            for j in range(1, int(n/2)):
                v_old = V[i][j]
                v_new=v_old + tau_MPI*R[i][j]
        
                eps_cur = abs(v_old - v_new) 
                if(eps_cur > eps_max):
                    eps_max = eps_cur
                V[i][j] = v_new
        for i in range(int(m/2) + 1, m):
            for j in range(int(n/2), n):
                v_old = V[i][j]
                v_new=v_old + tau_MPI*R[i][j]
        
                eps_cur = abs(v_old - v_new) 
                if(eps_cur > eps_max):
                    eps_max = eps_cur
                V[i][j] = v_new
        S += 1
        if(eps_max < eps or S >= Nmax - 1):
            break
    
    #* Подсчет невязки и фрейма ошибок
    #-----------------------------------------#
    nevN, max_Z = 0, 0
    for i in range(0, m + 1):
        for j in range(0, int(n/2) + 1):
            Z[i][j] = abs(U[i][j] - V[i][j])
            if(Z[i][j] >= max_Z):
                max_Z = Z[i][j]
                err_i = i
                err_j = j
    
    for i in range(int(m/2), m + 1):
        for j in range(int(n/2) + 1, n + 1):
            Z[i][j] = abs(U[i][j] - V[i][j])
            if(Z[i][j] >= max_Z):
                max_Z = Z[i][j]
                err_i = i
                err_j = j
    nevN = np.max(np.abs(R))

    #* Отрисовка графиков и создание таблиц + вывод справки
    #-----------------------------------------#
    func.graf_nst(a, c, h, k, m, n, U, 'Истниное решение')
    func.graf_nst(a, c, h, k, m, n, V, 'Численное решение')
    func.graf_nst(a, c, h, k, m, n, Z, 'Погрешность')

    data = pd.DataFrame(U)
    data.to_csv("data_u_test.csv", index=False)
    data = pd.DataFrame(V)
    data.to_csv("data_v_test.csv", index=False)
    data = pd.DataFrame(Z)
    data.to_csv("data_z_test.csv", index=False)
    
    print('Справка')
    print('Параметр tau = ', round(tau_MPI, 7))
    print('Число итераций N: ', S)
    print('Точность на шаге N: ', eps_max)
    print('Невязка N: ', nevN)
    print('Невязка начального приближения: ', nev0)
    print('Максимальная погрешность: ', max_Z, 'В точке: ', [round(xi[err_i], 5), round(yj[err_j], 5)])
    print('---------------------------------------------------------------------------------------------------')



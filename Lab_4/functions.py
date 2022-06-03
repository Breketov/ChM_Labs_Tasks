import numpy as np
from numpy import exp, sin, cos, sqrt, pi, array, max, abs
import matplotlib.pyplot as plt
import methods as mth
import pandas as pd
a, b, c, d = -1, 1, -1, 1 

#================================================================================#
def u(x, y):
    return exp(1 - x**2 - y**2)

def f(x, y):
    return -abs((sin(pi*x*y))**3)

def ft(x, y):
    return -4*u(x, y)*(x**2 + y**2 - 1)

#================================================================================#
def mu1_2(x, y):
    return 1 - y**2

def mu3_4(x, y):
    return abs(sin(pi*x))

#================================================================================#
def f_lam1(h, k, n, m):
	return (4*sin(pi/(2*n))**2)/(h**2) + (4*sin(pi/(2*m))**2)/(k**2)

def f_lamn(h, k, n, m):
    return (4*cos((pi)/(2*n))**2)/(h**2) + (4*cos((pi)/(2*m))**2)/(k**2)

def tau(S, k, lam1, lamn):
    s = S % k
    return 2/((lam1 + lamn) + (lamn - lam1)*cos((pi*(2*s + 1))/(2*k)))

#================================================================================#
def graf_std(x, y, z, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(x, y)
    z = array(z)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='inferno')
    plt.title(name)
    plt.show()

def graf_nst(a, c, h, k, m, n, Z, name):
    x1 = [a + i*h for i in range(0, n + 1)]
    y1 = [c + j*k for j in range(0, m + 1)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1, y1 = np.meshgrid(x1, y1)
    z1 = array(Z)
    ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap='inferno')
    plt.title(name)
    plt.show()

#================================================================================#
def functional_test(n, m):
    U, V, R, Z = [], [], [], []

    h = (b - a)/n
    k = (d - c)/m
    xi = [a + i*h for i in range(0, n + 1)]
    yj = [c + j*k for j in range(0, m + 1)]

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
            U[i][j] = u(xi[j], yj[i])
    
    for i in range(0, m + 1):
        V[i][0] = u(a, yj[i])
        V[i][n] = u(b, yj[i])
    for j in range(0, n + 1):
        V[0][j] = u(xi[j], c)
        V[m][j] = u(xi[j], d)
    return U, V, R, Z, xi, yj

def functional_main(n, m):
    V, R, Z = [], [], []

    h = (b - a)/n
    k = (d - c)/m
    xi = [a + i*h for i in range(0, n + 1)]
    yj = [c + j*k for j in range(0, m + 1)]

    for i in range(0, m + 1):
        V.append([])
        R.append([])
        Z.append([])
        for j in range(0, n + 1):
            V[i].append(0)
            R[i].append(0)
            Z[i].append(0)

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

    return V, R, Z, [xi, yj]

#================================================================================#
def error_test(n, m, U, V, R, Z):
    max_Z = 0
    for i in range(0, m + 1):
        for j in range(0, n + 1):
            Z[i][j] = abs(U[i][j] - V[i][j])
            if(Z[i][j] >= max_Z):
                max_Z = Z[i][j]
                err_i = i
                err_j = j
    nevN = np.max(np.abs(R))

    return [max_Z, err_i, err_j], nevN

def error_main(n, m, V1, V2, Z):
    max_Z = 0
    for i in range(0, m + 1):
        for j in range(0, n + 1):
            Z[i][j] = abs(V1[i][j] - V2[2*i][2*j])
            if(Z[i][j] >= max_Z):
                max_Z = Z[i][j]
                err_i = i
                err_j = j
    return [max_Z, err_i, err_j]

#================================================================================#
def data_test(U, V, Z):
    data = pd.DataFrame(U)
    data.to_csv("data_u_test.csv", index=False)
    data = pd.DataFrame(V)
    data.to_csv("data_v_test.csv", index=False)
    data = pd.DataFrame(Z)
    data.to_csv("data_z_test.csv", index=False)

def data_main(V1, V2, Z):
    data = pd.DataFrame(V1)
    data.to_csv("data_v1_main.csv", index=False)
    data = pd.DataFrame(V2)
    data.to_csv("data_v2_main.csv", index=False)
    data = pd.DataFrame(Z)
    data.to_csv("data_z_main.csv", index=False)

#================================================================================#
def test_task(n, m, eps, Nmax, omg, cheb, part):
    if part == 1:
        U, V, R, Z, xi, yj = functional_test(n, m)
        V, R, [eps_max, S, nev0] = mth.MVR(n, m, eps, Nmax, omg, V, R, 0)
        [max_Z, err_i, err_j], nevN = error_test(n, m, U, V, R, Z)
        
        graf_std(xi, yj, U, 'Истинное решение')
        graf_std(xi, yj, V, 'Численное решение')
        graf_std(xi, yj, Z, 'Погрешность')

        data_test(U, V, Z)

        print('Справка')
        print('Параметр omega: ', omg)
        print('Число итераций N: ', S)
        print('Точность на шаге N: ', eps_max)
        print('Невязка N: ', nevN)
        print('Невязка начального приближения: ', nev0)
        print('Максимальная погрешность: ', max_Z, 'В точке: ', [round(xi[err_i], 5), round(yj[err_j], 5)])
        print('---------------------------------------------------------------------------------------------------')

    elif part == 2:
        U, V, R, Z, xi, yj = functional_test(n, m)
        V, R, [eps_max, S, nev0] = mth.Chebyshev(n, m, eps, Nmax, cheb, V, R, 0)
        [max_Z, err_i, err_j], nevN = error_test(n, m, U, V, R, Z)
        
        graf_std(xi, yj, U, 'Истинное решение')
        graf_std(xi, yj, V, 'Численное решение')
        graf_std(xi, yj, Z, 'Погрешность')

        data_test(U, V, Z)

        print('Справка')
        print('Чебышевский параметр: ', cheb)
        print('Число итераций N: ', S)
        print('Точность на шаге N: ', eps_max)
        print('Невязка N: ', nevN)
        print('Невязка начального приближения: ', nev0)
        print('Максимальная погрешность: ', max_Z, 'В точке: ', [round(xi[err_i], 5), round(yj[err_j], 5)])
        print('---------------------------------------------------------------------------------------------------')

    elif part == 3:
        U, V, R, Z, xi, yj = functional_test(n, m)
        V, R, [eps_max, S, nev0, tau_MPI] = mth.MPI_standart(n, m, eps, Nmax, V, R, 0)
        [max_Z, err_i, err_j], nevN = error_test(n, m, U, V, R, Z)

        graf_std(xi, yj, U, 'Истинное решение')
        graf_std(xi, yj, V, 'Численное решение')
        graf_std(xi, yj, Z, 'Погрешность')

        data_test(U, V, Z)

        print('Справка')
        print('Параметр tau: ', round(tau_MPI, 6))
        print('Число итераций N: ', S)
        print('Точность на шаге N: ', eps_max)
        print('Невязка N: ', nevN)
        print('Невязка начального приближения: ', nev0)
        print('Максимальная погрешность: ', max_Z, 'В точке: ', [round(xi[err_i], 5), round(yj[err_j], 5)])
        print('---------------------------------------------------------------------------------------------------')

    elif part == 4:
        mth.MPI_non_standart(n, m, eps, Nmax)

def main_task(n, m, eps, Nmax, omg, cheb, part):
    if part == 1:
        V, R, Z, [xi1, yj1] = functional_main(n, m)
        V1, R1, [eps_max1, S1, nev01] = mth.MVR(n, m, eps, Nmax, omg[0], V, R, 1)
        V, R, Z_trash, [xi2, yj2] = functional_main(2*n, 2*m)
        V2, R2, [eps_max2, S2, nev02] = mth.MVR(2*n, 2*m, eps/10, Nmax, omg[1], V, R, 1)
        nevN1 = np.max(np.abs(R1))
        nevN2 = np.max(np.abs(R2))
        [max_Z, err_i, err_j] = error_main(n, m, V1, V2, Z)

        data_main(V1, V2, Z)

        graf_std(xi1, yj1, V1, 'Решение на обычной сетке')
        graf_std(xi2, yj2, V2, 'Решение на контрольной сетке')
        graf_std(xi1, yj1, Z, 'Точность')

        print('Справка')
        print('Число итераций основной сетки N: ', S1)
        print('Точность на шаге N: ', eps_max1)
        print('Невязка N обычной сетки: ', nevN1)
        print('Невязка начального приближения: ', nev01)
        print('---------------------------------------------------------------------------------------------------')
        print('Число итераций контрольной сетки N: ', S2)
        print('Точность на шаге N: ', eps_max2)
        print('Невязка N контрольной сетки: ', nevN2)
        print('Невязка начального приближения: ', nev02)
        print('')
        print('Максимальная разность решений: ', max_Z, 'В точке: ',  [round(xi1[err_i], 5), round(yj1[err_j], 5)])
        print('---------------------------------------------------------------------------------------------------')

    elif part == 2:
        V, R, Z, [xi1, yj1] = functional_main(n, m)
        V1, R1, [eps_max1, S1, nev01] = mth.Chebyshev(n, m, eps, Nmax, cheb, V, R, 1)
        V, R, Z_trash, [xi2, yj2] = functional_main(2*n, 2*m)
        V2, R2, [eps_max2, S2, nev02] = mth.Chebyshev(2*n, 2*m, eps/10, Nmax, cheb, V, R, 1)
        nevN1 = np.max(np.abs(R1))
        nevN2 = np.max(np.abs(R2))
        [max_Z, err_i, err_j] = error_main(n, m, V1, V2, Z)

        data_main(V1, V2, Z)

        graf_std(xi1, yj1, V1, 'Решение на обычной сетке')
        graf_std(xi2, yj2, V2, 'Решение на контрольной сетке')
        graf_std(xi1, yj1, Z, 'Точность')

        print('Справка')
        print('Число итераций основной сетки N: ', S1)
        print('Точность на шаге N: ', eps_max1)
        print('Невязка N обычной сетки: ', nevN1)
        print('Невязка начального приближения: ', nev01)
        print('---------------------------------------------------------------------------------------------------')
        print('Число итераций контрольной сетки N: ', S2)
        print('Точность на шаге N: ', eps_max2)
        print('Невязка N контрольной сетки: ', nevN2)
        print('Невязка начального приближения: ', nev02)
        print('')
        print('Максимальная разность решений: ', max_Z, 'В точке: ',  [round(xi1[err_i], 5), round(yj1[err_j], 5)])
        print('---------------------------------------------------------------------------------------------------')

    elif part == 3:
        V, R, Z, [xi1, yj1] = functional_main(n, m)
        V1, R1, [eps_max1, S1, nev01, tau_MPI1] = mth.MPI_standart(n, m, eps, Nmax, V, R, 1)
        V, R, Z_trash, [xi2, yj2] = functional_main(2*n, 2*m)
        V2, R2, [eps_max2, S2, nev02, tau_MPI2] = mth.MPI_standart(2*n, 2*m, eps/10, Nmax, V, R, 1)
        nevN1 = np.max(np.abs(R1))
        nevN2 = np.max(np.abs(R2))
        [max_Z, err_i, err_j] = error_main(n, m, V1, V2, Z)

        data_main(V1, V2, Z)

        graf_std(xi1, yj1, V1, 'Решение на обычной сетке')
        graf_std(xi2, yj2, V2, 'Решение на контрольной сетке')
        graf_std(xi1, yj1, Z, 'Точность')

        print('Справка')
        print('Число tau на обычной сетке:', round(tau_MPI1, 6))
        print('Число итераций основной сетки N: ', S1)
        print('Точность на шаге N: ', eps_max1)
        print('Невязка N обычной сетки: ', nevN1)
        print('Невязка начального приближения: ', nev01)
        print('---------------------------------------------------------------------------------------------------')
        print('Число tau на контрольной сетке:', round(tau_MPI2, 6))
        print('Число итераций контрольной сетки N: ', S2)
        print('Точность на шаге N: ', eps_max2)
        print('Невязка N контрольной сетки: ', nevN2)
        print('Невязка начального приближения: ', nev02)
        print('')
        print('Максимальная разность решений: ', max_Z, 'В точке: ',  [round(xi1[err_i], 5), round(yj1[err_j], 5)])
        print('---------------------------------------------------------------------------------------------------')

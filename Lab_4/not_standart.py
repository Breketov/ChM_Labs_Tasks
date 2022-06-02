from functions import *

def solve_test_Nan(n, m, eps, Nmax):
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
    tau_MPI = 2/(lam1 + lamn)

    #* Заполняем массивы
    #-----------------------------------------#
    for i in range(0, m + 1):
        U.append([])
        V.append([])
        R.append([])
        Z.append([])
        for j in range(0, int(n/2) + 1):
            U[i].append(0)
            V[i].append(0)
            R[i].append(0)
            Z[i].append(0)
            U[i][j] = u(xi[j], yj[i])

    for i in range(int(m/2), m + 1):
        for j in range(int(n/2) + 1, n + 1):
            U[i].append(0)
            V[i].append(0)
            R[i].append(0)
            Z[i].append(0)
            U[i][j] = u(xi[j], yj[i])

    #* Заполняем граничные условия
    #-----------------------------------------#
    for i in range(0, m + 1):
        V[i][0] = u(a, yj[i])

    for i in range(0, int(m/2)):
        V[i][int(n/2)] = u(0, yj[i])

    for i in range(int(m/2), m + 1):
        V[i][n] = u(b, yj[i])

    for j in range(0, int(n/2)):
        V[0][j] = u(xi[j], c)

    for j in range(int(n/2), n + 1):
        V[int(m/2)][j] = u(xi[j], 0)

    for j in range(0, n + 1):
        V[m][j] = u(xi[j], d)

    #* Запускаем метод
    #-----------------------------------------#
    S = 0
    while (True):
        if (S == 1):
            nev0 = np.max(np.abs(R))

        eps_max = 0 
        #for i in range(1, m):
        #    for j in range(1, int(n/2)):
        #        R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + ft(xi[j], yj[i])
        #for i in range(int(m/2) + 1, m):
        #    for j in range(int(n/2), n):
        #        R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + ft(xi[j], yj[i])

        for i in range(1, m):
            j = 1
            if i < int(m/2) + 1:
                while j < int(n/2):
                    R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + ft(xi[j], yj[i])
                    j += 1
            else:
                while j < n:
                    R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + ft(xi[j], yj[i])
                    j += 1
        for i in range(1, m):
            j = 1
            if i < int(n/2) + 1:
                while j < int(n/2):
                    v_old = V[i][j]
                    v_new=v_old + tau_MPI*R[i][j]
                    eps_cur = abs(v_old - v_new) 
                    if(eps_cur > eps_max):
                        eps_max = eps_cur
                    V[i][j] = v_new
                    j += 1
            else:
                while j < n:
                    v_old = V[i][j]
                    v_new=v_old + tau_MPI*R[i][j]
                    eps_cur = abs(v_old - v_new) 
                    if(eps_cur > eps_max):
                        eps_max = eps_cur
                    V[i][j] = v_new
                    j += 1

        #for i in range(1, m):
        #    for j in range(1, int(n/2)):
        #        v_old = V[i][j]
        #        v_new=v_old + tau_MPI*R[i][j]
        #
        #        eps_cur = abs(v_old - v_new) 
        #        if(eps_cur > eps_max):
        #            eps_max = eps_cur
        #        V[i][j] = v_new
        #for i in range(int(m/2) + 1, m):
        #    for j in range(int(n/2), n):
        #        v_old = V[i][j]
        #        v_new=v_old + tau_MPI*R[i][j]
        #
        #        eps_cur = abs(v_old - v_new) 
        #        if(eps_cur > eps_max):
        #            eps_max = eps_cur
        #        V[i][j] = v_new
        S += 1
        if(eps_max < eps or S >= Nmax):
            S -= 1
            break
    
    #* Подсчет невязки и фрейма ошибок
    #-----------------------------------------#
    max_Z = 0
    for i in range(1, m):
        j = 1
        if i < int(m/2) + 1:
            while j < int(n/2):
                Z[i][j] = abs(U[i][j] - V[i][j])
                if(Z[i][j] >= max_Z):
                    max_Z = Z[i][j]
                    err_i = i
                    err_j = j
                j += 1
        else:
            while j < n:
                Z[i][j] = abs(U[i][j] - V[i][j])
                if(Z[i][j] >= max_Z):
                    max_Z = Z[i][j]
                    err_i = i
                    err_j = j
                j += 1
    nevN = np.max(np.abs(R))
    #for i in range(0, m + 1):
    #    for j in range(0, int(n/2) + 1):
    #        Z[i][j] = abs(U[i][j] - V[i][j])
    #        if(Z[i][j] >= max_Z):
    #            max_Z = Z[i][j]
    #            err_i = i
    #            err_j = j
    #
    #for i in range(int(m/2), m + 1):
    #    for j in range(int(n/2) + 1, n + 1):
    #        Z[i][j] = abs(U[i][j] - V[i][j])
    #        if(Z[i][j] >= max_Z):
    #            max_Z = Z[i][j]
    #            err_i = i
    #            err_j = j
    #nevN = np.max(np.abs(R))
    
    #* Отрисовка графиков и создание таблиц + вывод справки
    #-----------------------------------------#
    #graf_nst_Nan(a, c, h, k, m, n, U)
    #graf_nst_Nan(a, c, h, k, m, n, V)
    #graf_nst_Nan(a, c, h, k, m, n, Z)

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

def solve_test_0(n, m, eps, Nmax):
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
            U[i][j] = u(xi[j], yj[i])

    for i in range(int(m/2), m + 1):
        for j in range(int(n/2) + 1, n + 1):
            U[i][j] = u(xi[j], yj[i])

    #* Заполняем граничные условия
    #-----------------------------------------#
    for i in range(0, m + 1):
        V[i][0] = u(a, yj[i])

    for i in range(0, int(m/2)):
        V[i][int(n/2)] = u(0, yj[i])

    for i in range(int(m/2), m + 1):
        V[i][n] = u(b, yj[i])

    for j in range(0, int(n/2)):
        V[0][j] = u(xi[j], c)

    for j in range(int(n/2), n + 1):
        V[int(m/2)][j] = u(xi[j], 0)

    for j in range(0, n + 1):
        V[m][j] = u(xi[j], d)

    #* Запускаем метод
    #-----------------------------------------#
    S = 0
    while (True):
        if (S == 1):
            nev0 = np.max(np.abs(R))

        eps_max = 0 
        #for i in range(1, m):
        #    for j in range(1, int(n/2)):
        #        R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + ft(xi[j], yj[i])
        #for i in range(int(m/2) + 1, m):
        #    for j in range(int(n/2), n):
        #        R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + ft(xi[j], yj[i])

        for i in range(1, m):
            j = 1
            if i < int(m/2) + 1:
                while j < int(n/2):
                    R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + ft(xi[j], yj[i])
                    j += 1
            else:
                while j < n:
                    R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + ft(xi[j], yj[i])
                    j += 1
        for i in range(1, m):
            j = 1
            if i < int(n/2) + 1:
                while j < int(n/2):
                    v_old = V[i][j]
                    v_new=v_old + tau_MPI*R[i][j]
                    eps_cur = abs(v_old - v_new) 
                    if(eps_cur > eps_max):
                        eps_max = eps_cur
                    V[i][j] = v_new
                    j += 1
            else:
                while j < n:
                    v_old = V[i][j]
                    v_new=v_old + tau_MPI*R[i][j]
                    eps_cur = abs(v_old - v_new) 
                    if(eps_cur > eps_max):
                        eps_max = eps_cur
                    V[i][j] = v_new
                    j += 1

        #for i in range(1, m):
        #    for j in range(1, int(n/2)):
        #        v_old = V[i][j]
        #        v_new=v_old + tau_MPI*R[i][j]
        #
        #        eps_cur = abs(v_old - v_new) 
        #        if(eps_cur > eps_max):
        #            eps_max = eps_cur
        #        V[i][j] = v_new
        #for i in range(int(m/2) + 1, m):
        #    for j in range(int(n/2), n):
        #        v_old = V[i][j]
        #        v_new=v_old + tau_MPI*R[i][j]
        #
        #        eps_cur = abs(v_old - v_new) 
        #        if(eps_cur > eps_max):
        #            eps_max = eps_cur
        #        V[i][j] = v_new
        S += 1
        if(eps_max < eps or S >= Nmax):
            S -= 1
            break
    
    #* Подсчет невязки и фрейма ошибок
    #-----------------------------------------#
    nevN, max_Z = 0, 0
    for i in range(1, m):
        j = 1
        if i < int(m/2) + 1:
            while j < int(n/2):
                Z[i][j] = abs(U[i][j] - V[i][j])
                if(Z[i][j] >= max_Z):
                    max_Z = Z[i][j]
                    err_i = i
                    err_j = j
                j += 1
        else:
            while j < n:
                Z[i][j] = abs(U[i][j] - V[i][j])
                if(Z[i][j] >= max_Z):
                    max_Z = Z[i][j]
                    err_i = i
                    err_j = j
                j += 1
    nevN = np.max(np.abs(R))
    #for i in range(0, m + 1):
    #    for j in range(0, int(n/2) + 1):
    #        Z[i][j] = abs(U[i][j] - V[i][j])
    #        if(Z[i][j] >= max_Z):
    #            max_Z = Z[i][j]
    #            err_i = i
    #            err_j = j
    #
    #for i in range(int(m/2), m + 1):
    #    for j in range(int(n/2) + 1, n + 1):
    #        Z[i][j] = abs(U[i][j] - V[i][j])
    #        if(Z[i][j] >= max_Z):
    #            max_Z = Z[i][j]
    #            err_i = i
    #            err_j = j
    #nevN = np.max(np.abs(R))

    #* Отрисовка графиков и создание таблиц + вывод справки
    #-----------------------------------------#
    graf_nst_0(a, c, h, k, m, n, U)
    graf_nst_0(a, c, h, k, m, n, V)
    graf_nst_0(a, c, h, k, m, n, Z)

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


"""
Для множественного тестирования
"""
#!solve_test_0(n, m, eps, Nmax)
#solve_test_0(5, 5, 0.00001,        100000)
#solve_test_0(10, 10, 0.000001,     100000)
#solve_test_0(20, 20, 0.0000001,    100000)
#solve_test_0(40, 40, 0.00000001,   100000)
#solve_test_0(80, 80, 0.000000001,  100000)

#!solve_test_Nan(n, m, eps, Nmax)
#solve_test_Nan(5, 5, 0.00001,       100000)
#solve_test_Nan(10, 10, 0.000001,    100000)
#solve_test_Nan(20, 20, 0.0000001,   100000)
#solve_test_Nan(40, 40, 0.00000001,  100000)
#solve_test_Nan(80, 80, 0.000000001, 100000)
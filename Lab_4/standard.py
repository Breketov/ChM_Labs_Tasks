from functions import *

def solve_test(n , m, eps, Nmax, cheb, omega, part):
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
    S = 0
    while (True):
        if (S == 1):
            nev0 = 0
            for i in range(0, m + 1):
                for j in range(0, n + 1):
                    nev0 = nev0 + R[i][j]*R[i][j]
            nev0 = sqrt(nev0)

        eps_max = 0 
        for i in range(1, m):
            for j in range(1, n):
                R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + ft(xi[j], yj[i])
        for i in range (1, m):
            for j in range (1, n):
                v_old = V[i][j]
                if part == 1:
                    v_new = -omega*(h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]))
                    v_new = v_new + (1 - omega)*A*V[i][j] - omega*ft(xi[j], yj[i]) 
                    v_new = v_new/A
                elif part == 2:
                    v_new = v_old + tau(S, cheb, lam1, lamn)*R[i][j]
                else:
                    v_new=v_old + tau_MPI*R[i][j]

                eps_cur = abs(v_old - v_new) 
                if(eps_cur > eps_max):
                    eps_max = eps_cur
                V[i][j] = v_new
        S += 1
        if(eps_max < eps or S == Nmax):
            S -= 1
            break
    
    #* Подсчет невязки и фрейма ошибок
    #-----------------------------------------#
    nevN, max_Z = 0, 0
    for i in range(0, m + 1):
        for j in range(0, n + 1):
            Z[i][j] = abs(U[i][j] - V[i][j])
            if(Z[i][j] >= max_Z):
                max_Z = Z[i][j]
                err_i = i
                err_j = j
            nevN += R[i][j]**2
    
    #* Отрисовка графиков и создание таблиц + вывод справки
    #-----------------------------------------#
    graf_std(xi, yj, U)
    graf_std(xi, yj, V)
    graf_std(xi, yj, Z)

    data = pd.DataFrame(U)
    data.to_csv("data_u_test.csv", index=False)
    data = pd.DataFrame(V)
    data.to_csv("data_v_test.csv", index=False)
    data = pd.DataFrame(Z)
    data.to_csv("data_z_test.csv", index=False)
    
    print('Справка')
    print('Число итераций N: ', S)
    print('Точность на шаге N: ', eps_max)
    print('Невязка N: ', sqrt(nevN))
    print('Невязка начального приближения: ', nev0)
    print('Максимальная погрешность: ', max_Z, 'В точке: ', [round(xi[err_i], 5), round(yj[err_j], 5)])
    print('---------------------------------------------------------------------------------------------------')

def task_main(n , m, eps, Nmax, cheb, omg, part):
    def solve_main(n , m, eps, Nmax, cheb, omega, part):
        V, R = [], []
        a, c = -1, -1
        b, d = 1, 1

        h = (b - a)/n
        k = (d - c)/m
        xi = [a + i*h for i in range(0, n + 1)]
        yj = [c + j*k for j in range(0, m + 1)]
        grid = [xi, yj]

        h2 = 1/(h**2)
        k2 = 1/(k**2)
        A = -2*(h2 + k2)

        lam1 = f_lam1(h, k, n, m)
        lamn = f_lamn(h, k, n, m)
        tau_MPI = 2/(lam1 + lamn)

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
        S, nevN = 0, 0
        while (True):
            if (S == 1):
                nev0 = 0
                for i in range(0, m + 1):
                    for j in range(0, n + 1):
                        nev0 = nev0 + R[i][j]*R[i][j]
                nev0 = sqrt(nev0)

            eps_max = 0 
            for i in range(1, m):
                for j in range(1, n):
                    R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + f(xi[j], yj[i])
            for i in range (1, m):
                for j in range (1, n):
                    v_old = V[i][j]
                    if part == 1:
                        v_new = -omega*(h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j])) 
                        v_new = v_new + (1 - omega)*A*V[i][j] - omega*f(xi[j], yj[i]) 
                        v_new = v_new/A
                    elif part == 2:
                        v_new = v_old + tau(S, cheb, lam1, lamn)*R[i][j]
                    else:
                        v_new=v_old + tau_MPI*R[i][j]

                    eps_cur = abs(v_old - v_new) 
                    if(eps_cur > eps_max):
                        eps_max = eps_cur
                    V[i][j] = v_new
            S += 1
            if(eps_max < eps or S == Nmax):
                S -= 1
                break
            
        for i in range(0, m + 1):
            for j in range(0, n + 1):
                nevN += R[i][j]**2
        return V, nevN, S, eps_max, nev0, grid

    if part == 1:
        omg1 = omg[0]
        omg2 = omg[1]
    else:
        omg1, omg2 = 0, 0
    Z = []
    for i in range(0, m + 1):
        Z.append([])
        for j in range(0, n + 1):
            Z[i].append(0)
    
    #* Решаем сетки
    #-----------------------------------------#
    V1, nev1N, S1, eps_max1, nev0_1, point1 = solve_main(n , m, eps, Nmax, cheb, omg1, part)
    V2, nev2N, S2, eps_max2, nev0_2, point2 = solve_main(int(2*n) , int(2*m), eps, Nmax, cheb, omg2, part)

    #* Подсчет фрейма ошибок
    #-----------------------------------------#
    max_Z = 0
    for i in range(0, m + 1):
        for j in range(0, n + 1):
            Z[i][j] = abs(V1[i][j] - V2[2*i][2*j])
            if(Z[i][j] >= max_Z):
                max_Z = Z[i][j]
                err_i = i
                err_j = j

    #* Отрисовка графиков и создание таблиц + вывод справки
    #-----------------------------------------#
    data = pd.DataFrame(V1)
    data.to_csv("data_v1_main.csv", index=False)
    data = pd.DataFrame(V2)
    data.to_csv("data_v2_main.csv", index=False)
    data = pd.DataFrame(Z)
    data.to_csv("data_z_main.csv", index=False)

    x, y = point1[0], point1[1] 
    graf_std(x, y, V1)
    x, y = point2[0], point2[1] 
    graf_std(x, y, V2)
    x, y = point1[0], point1[1] 
    graf_std(x, y, Z)

    print('Справка')
    if part == 1:
        print('Параметр omega = ', omg[0])
    print('Число итераций основной сетки N: ', S1)
    print('Точность на шаге N: ', eps_max1)
    print('Невязка N обычной сетки: ', nev1N)
    print('Невязка начального приближения: ', nev0_1)
    print('---------------------------------------------------------------------------------------------------')
    if part == 1:
        print('Параметр omega = ', omg[1])
    print('Число итераций контрольной сетки N: ', S2)
    print('Точность на шаге N: ', eps_max2)
    print('Невязка N контрольной сетки: ', nev2N)
    print('Невязка начального приближения: ', nev0_2)
    print('')
    print('Максимальная разность решений: ', max_Z, 'В точке: ',  [round(point1[0][err_i], 5), round(point1[1][err_j], 5)])
    print('---------------------------------------------------------------------------------------------------')

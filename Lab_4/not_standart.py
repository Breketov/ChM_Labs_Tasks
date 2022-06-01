from functions import *
a, c = -1, -1
b, d = 1, 1

def graf_nst(h, k, m, n, Z):
    x1 = [a + i*h for i in range(0, int(n/2) + 1)]
    y1 = [c + j*k for j in range(0, int(m/2) + 1)]
    x2 = [a + i*h for i in range(0, n + 1)]
    y2 = [c + j*k for j in range(int(m/2), m + 1)]

    z1, z2 = [], []
    for i in range(0, int(m/2) + 1):
        if i == int(m/2):
            z1.append([])
            for j in range(0, int(n/2) + 1):
                z1[i].append(Z[i][j])
        else:
            z1.append(Z[i])

    for i in range(int(m/2), m + 1):
        z2.append(Z[i])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1, y1 = np.meshgrid(x1, y1)
    x2, y2 = np.meshgrid(x2, y2)
    z1 = array(z1)
    z2 = array(z2)
    ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap='inferno')
    ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap='inferno')
    plt.show()


def solve_test(n , m, eps, Nmax):
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
        for j in range(0, int(n/2) + 1):
            U[i].append(0)
            U[i][j] = u(xi[j], yj[i])

    for i in range(int(m/2), m + 1):
        for j in range(int(n/2) + 1, n + 1):
            U[i].append(0)
            U[i][j] = u(xi[j], yj[i])

    for i in range(0, m + 1):
        V.append([])
        R.append([])
        Z.append([])
        for j in range(0, int(n/2) + 1):
            V[i].append(0)
            R[i].append(0)
            Z[i].append(0)

    for i in range(int(m/2), m + 1):
        for j in range(int(n/2), n):
            V[i].append(0)
            R[i].append(0)
            Z[i].append(0)

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
            nev0 = 0
            for i in range(0, m + 1):
                for j in range(0, int(n/2) + 1):
                    nev0 = nev0 + R[i][j]*R[i][j]
            for i in range(int(m/2), m + 1):
                for j in range(int(n/2) + 1, n + 1):
                    nev0 = nev0 + R[i][j]*R[i][j]
            nev0 = sqrt(nev0)

        eps_max = 0 
        for i in range(1, m):
            for j in range(1, int(n/2)):
                R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + ft(xi[j], yj[i])
        for i in range(int(m/2) + 1, m):
            for j in range(int(n/2), n):
                R[i][j] = A*V[i][j] + h2*(V[i][j + 1] + V[i][j - 1]) + k2*(V[i + 1][j] + V[i - 1][j]) + ft(xi[j], yj[i])

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
        if(eps_max < eps or S >= Nmax):
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
            nevN += R[i][j]**2
    
    for i in range(int(m/2), m + 1):
        for j in range(int(n/2) + 1, n + 1):
            Z[i][j] = abs(U[i][j] - V[i][j])
            if(Z[i][j] >= max_Z):
                max_Z = Z[i][j]
                err_i = i
                err_j = j
            nevN += R[i][j]**2
    
    #* Отрисовка графиков и создание таблиц + вывод справки
    #-----------------------------------------#
    graf_nst(h, k, m, n, U)
    graf_nst(h, k, m, n, V)
    graf_nst(h, k, m, n, Z)

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

def solve_main():
    # пока нет
    return 21





solve_test(20, 20, 0.00001, 10000)
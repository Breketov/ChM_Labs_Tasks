from numpy import sqrt, sin, cos, pi, array, log
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import max_error

#?________________ОСНОВНЫЕ_ЗАДАЧИ_И_СПЛАЙН________________
def f(x):
    """Функция со всеми задачами\\
    Аргументы для funcflag:\\
    'TEST_1 - Тестовая задача\\
    'MAIN_11' - Основная задача 1\\
    'MAIN_12' - Основная задача 2\\
    'MAIN_13' - Основная задача 3
    
    'MAIN_21' - Основная осцилирующая 1\\
    'MAIN_22' - Основная осцилирующая 2\\
    'MAIN_23' - Основная осцилирующая 3"""
    # TODO Тестовая задача
    if funcflag == 'TEST_1':
        if -1 <= x <= 0:
            return x*x*x + 3*(x**2)
        else:
            return -x*x*x + 3*(x**2)

    # TODO Основная задача
    elif funcflag == 'MAIN_11':
        return sqrt(x)*sin(x)
    elif funcflag == 'MAIN_12':
        return log(x + 1)/(x + 1)
    elif funcflag == 'MAIN_13':
        return log(x + 1)/x

    # TODO Основная задача с осцилирующей функцией
    elif funcflag == 'MAIN_21':
        return sqrt(x)*sin(x) + cos(10*x)
    elif funcflag == 'MAIN_22':
        return log(x + 1)/(x + 1) + cos(10*x)
    elif funcflag == 'MAIN_23':
        return log(x + 1)/x + cos(10*x)

def df(x):
    """Функция со всеми первыми производными\\
    Аргументы для funcflag:\\
    'TEST_1 - Тестовая задача\\
    'MAIN_11' - Основная задача 1\\
    'MAIN_12' - Основная задача 2\\
    'MAIN_13' - Основная задача 3
    
    'MAIN_21' - Основная осцилирующая 1\\
    'MAIN_22' - Основная осцилирующая 2\\
    'MAIN_23' - Основная осцилирующая 3"""
    # TODO Тестовая задача
    if funcflag == 'TEST_1':
        if -1 <= x <= 0:
            return 3*(x**2) + 6*x
        else:
            return -3*(x**2) + 6*x

    # TODO Основная задача
    elif funcflag == 'MAIN_11':
        return sin(x)/(2*sqrt(x)) + sqrt(x)*cos(x)
    elif funcflag == 'MAIN_12':
        return (1 - log(x + 1))/((x + 1)**2)
    elif funcflag == 'MAIN_13':
        return (x/(x + 1))/(x**2) - log(x + 1)/(x**2)

    # TODO Основная задача с осцилирующей функцией
    elif funcflag == 'MAIN_21':
        return sin(x)/(2*sqrt(x)) + sqrt(x)*cos(x) - 10*sin(10*x)
    elif funcflag == 'MAIN_22':
        return (1 - log(x + 1))/((x + 1)**2) - 10*sin(10*x)
    elif funcflag == 'MAIN_23':
        return (x/(x + 1))/(x**2) - log(x + 1)/(x**2) - 10*sin(10*x)

def ddf(x):
    """Функция со всеми вторыми производными\\
    Аргументы для funcflag:\\
    'TEST_1 - Тестовая задача\\
    'MAIN_11' - Основная задача 1\\
    'MAIN_12' - Основная задача 2\\
    'MAIN_13' - Основная задача 3
    
    'MAIN_21' - Основная осцилирующая 1\\
    'MAIN_22' - Основная осцилирующая 2\\
    'MAIN_23' - Основная осцилирующая 3"""
    # TODO Тестовая задача
    if funcflag == 'TEST_1':
        if -1 <= x <= 0:
            return 6*x + 6
        else:
            return -6*x + 6

    # TODO Основная задача
    elif funcflag == 'MAIN_11':
        return cos(x)/sqrt(x) - sin(x)/(4*sqrt(x*x*x)) - sqrt(x)*sin(x)
    elif funcflag == 'MAIN_12':
        return (-3*x - 3 + 2*log(x + 1)*(x + 1))/((x + 1)**4)
    elif funcflag == 'MAIN_13':
        return ((1/((x + 1)**2) - 1/(x + 1))*(x**2) - 2*x*(x/(x + 1) - log(x + 1)))/(x**4)

    # TODO Основная задача с осцилирующей функцией
    elif funcflag == 'MAIN_21':
        return cos(x)/sqrt(x) - sin(x)/(4*sqrt(x*x*x)) - sqrt(x)*sin(x) - 100*cos(10*x)
    elif funcflag == 'MAIN_22':
        return (-3*x - 3 + 2*log(x + 1)*(x + 1))/((x + 1)**4) - 100*cos(10*x)
    elif funcflag == 'MAIN_23':
        return ((1/((x + 1)**2) - 1/(x + 1))*(x**2) - 2*x*(x/(x + 1) - log(x + 1)))/(x**4) - 100*cos(10*x)

def spline(a, b, c, d, X, x):
    """Каноническая запись сплайна"""
    for i in range(1, n + 1):
        if X[i - 1] <= x <= X[i]:
            return a[i] + b[i]*(x - X[i]) + 0.5*c[i]*(x - X[i])**2 + (d[i]/6)*(x - X[i])**3

def dspline(b, c, d, X, x):
    """Первая производная сплайна"""
    for i in range(1, n + 1):
        if X[i - 1] <= x <= X[i]:
            return b[i] + c[i]*(x - X[i]) + (d[i]/2)*(x - X[i])**2

def ddspline(c, d, X, x):
    """Вторая производная сплайна"""
    for i in range(1, n + 1):
        if X[i - 1] <= x <= X[i]:
            return c[i] + d[i]*(x - X[i])

#?________________РАСЧЕТЫ________________
def error(X, A, B, C, D):
    """Функция нахождения максимальной ошибки в точке и нахождения всех погрешностей"""
    N = 2*n
    h = (b - a)/N
    X_dop = []
    for i in range(N + 1):
        X_dop.append(a + i*h)
    max_error = 0
    max_df_error = 0
    max_ddf_error = 0
    x_max_error = 0
    x_max_df_error = 0
    x_max_ddf_error = 0

    err_spline = []
    err_dspline = []
    err_ddspline = []
    F, dF, ddF, S, dS, ddS = [], [], [], [], [], []
    for i in X_dop:
        tmp = f(i)
        spl = spline(A, B, C, D, X, i)
        err = abs(tmp - spl)
        F.append(tmp)
        S.append(spl)
        err_spline.append(err)
        if err > max_error:
            max_error = err
            x_max_error = i
        
        tmp = df(i)
        spl = dspline(B, C, D, X, i)
        err = abs(tmp - spl)
        dF.append(tmp)
        dS.append(spl)
        err_dspline.append(err)
        if err > max_df_error:
            max_df_error = err
            x_max_df_error = i

        tmp = ddf(i)
        spl = ddspline(C, D, X, i)
        err = abs(tmp - spl)
        ddF.append(tmp)
        ddS.append(spl)
        err_ddspline.append(err)
        if err > max_ddf_error:
            max_ddf_error = err
            x_max_ddf_error = i
        
    err = [max_error, x_max_error, max_df_error, x_max_df_error, max_ddf_error, x_max_ddf_error]
    err_list = [err_spline, err_dspline, err_ddspline]
    F_list = [F, dF, ddF]
    S_list = [S, dS, ddS]
    data_err(err_list, F_list, S_list, X_dop)
    return err

#?________________ГРАФИКИ________________
def fun_special_graf(flash, x):
    """Небольшой костыль для отрисовки графиков тестовой функции"""
    X = []
    if flash == 0:
        for i in range(len(x)):
            if -1 <= x[i] <= 0:
                X.append(3*(x[i]**2) + 6*x[i])
            else:
                X.append(-3*(x[i]**2) + 6*x[i])
    if flash == 1:
        for i in range(len(x)):
            if -1 <= x[i] <= 0:
                X.append(6*x[i] + 6)
            else:
                X.append(-6*x[i] + 6)
    return X

def graf(x, y):
    """Функция отрисовки и сохранения графиков для всех задач"""
    fig = plt.figure(figsize=(12,5))
    data1 = pd.read_csv('table_error1.csv')
    data2 = pd.read_csv('table_error2.csv')
    data3 = pd.read_csv('table_error3.csv')
    S_x = data1['x[j]']
    S_y = data1['S[xj]']
    dS_y = data2["S'[xj]"]
    ddS_y = data3["S''[xj]"]

    if funcflag == 'TEST_1':
        plt.subplot(1, 3, 1)
        plt.plot(x, y, marker='.')
        plt.plot(S_x, S_y)
        plt.grid()
        plt.title("Функция")

        y = fun_special_graf(0, x)
        plt.subplot(1, 3, 2)
        plt.plot(x, y, marker='.')
        plt.plot(S_x, dS_y)
        plt.grid()
        plt.title("Первая производная")

        y = fun_special_graf(1, x)
        plt.subplot(1, 3, 3)
        plt.plot(x, y, marker='.')
        plt.plot(S_x, ddS_y)
        plt.grid()
        plt.title("Вторая производная")

        plt.savefig("Test_graf.jpg")
        plt.show()
    else:
        plt.subplot(1, 3, 1)
        plt.plot(x, y, marker='.')
        plt.plot(S_x, S_y)
        plt.grid()
        plt.title("Функция")

        y = df(array(x))
        plt.subplot(1, 3, 2)
        plt.plot(x, y, marker='.')
        plt.plot(S_x, dS_y)
        plt.grid()
        plt.title("Первая производная")

        y = ddf(array(x))
        plt.subplot(1, 3, 3)
        plt.plot(x, y, marker='.')
        plt.plot(S_x, ddS_y)
        plt.grid()
        plt.title("Вторая производная")

        plt.savefig("Main_graf.jpg")
        plt.show()

#?________________ДАТАФРЕЙМЫ________________
def data(X, A, B, C, D):
    """Создание таблиц с данными по текущей задачи, принимает подсчитанные коэффициенты из функции main"""
    table = pd.DataFrame(data = {})
    for i in range(1, len(X)):
        row = {'i': i, 'X[i-1]': round(X[i - 1], 10), 'X[i]': round(X[i], 10), 'a[i]': round(A[i], 10), 
        'b[i]': round(B[i], 10), 'c[i]': round(C[i], 10), 'd[i]': round(D[i], 8)}
        table = table.append(row, ignore_index=True)
    table.to_csv('table_task.csv', index=False)

def data_err(error_list, F_list, S_list, X):
    """Создание таблиц с данными по производным, принимает подсчитанные коэффициенты из функции coef"""
    table_err1 = pd.DataFrame(data = {})
    for j in range(0, len(X) - 1):
        row = {'j': j, 'x[j]': round(X[j], 10), 'F[xj]': round(F_list[0][j], 10), 
        'S[xj]': round(S_list[0][j], 10), 'F[xj] - S[xj]': error_list[0][j]}
        table_err1 = table_err1.append(row, ignore_index=True)
    table_err1.to_csv('table_error1.csv', index=False)

    table_err2 = pd.DataFrame(data = {})
    for j in range(0, len(X) - 1):
        row = {'j': j, 'x[j]': round(X[j], 10), "F'[xj]": round(F_list[1][j], 10), 
        "S'[xj]": round(S_list[1][j], 10),  "F'[xj] - S'[xj]": error_list[1][j]}
        table_err2 = table_err2.append(row, ignore_index=True)
    table_err2.to_csv('table_error2.csv', index=False)

    table_err3 = pd.DataFrame(data = {})
    for j in range(0, len(X) - 1):
        row = {'j': j, 'x[j]': round(X[j], 10), "F''[xj]": round(F_list[2][j], 10), 
        "S''[xj]": round(S_list[2][j], 10),  "F''[xj] - S''[xj]": error_list[2][j]}
        table_err3 = table_err3.append(row, ignore_index=True)
    table_err3.to_csv('table_error3.csv', index=False)

#?___________________ЗАПУСК___________________
def main():
    """Принимает данные из протоинтерфейса и является основной функцией запуска программы на счет\\
    Ничего не принимает, так как интерфейс создает глобальные переменные\\
    \\
    Сама по себе функция считает ключевые коэффициенты A, B, C, D"""

    my1, my2 = 0, 0
    h = (b - a)/n
    X = [0]*(n + 1)
    A, B, C, D = [0]*(n + 1), [0]*(n + 1), [0]*(n + 1), [0]*(n + 1)
    alpha, beta = [0]*(n + 1), [0]*(n + 1)
    for i in range(0, n + 1):
        X[i] = a + i*h
        A[i] = f(X[i])
    
    alpha[1] = 0
    beta[1] = my1
    for i in range(1, n):
        alpha[i + 1] = -h/(alpha[i]*h + 4*h)
        beta[i + 1] = ((-6/h)*(A[i + 1] - 2*A[i] + A[i - 1]) + beta[i]*h)/(-4*h - alpha[i]*h)
    
    C[n] = my2
    for i in range(n, 0, -1):
        C[i - 1] = alpha[i]*C[i] + beta[i]
    
    for i in range(1, n + 1):
        B[i] = (A[i] - A[i - 1])/h + h*(2*C[i] + C[i - 1])/6
        D[i] = (C[i] - C[i - 1])/h
    
    data(X, A, B, C, D)

    err = error(X, A, B, C, D)
    graf(X, A)

    print('-----------------------------------------------------------------------------------------------')
    print('СПРАВКА')
    print('Максимальная ошибка:', err[0], '     в точке x =', err[1])
    print('Максимальная ошибка первой производной:', err[2], '     в точке x =', err[3])
    print('Максимальная ошибка второй производной:', err[4], '     в точке x =', err[5])


#?________________ТИПО_ИНТЕРФЕЙС________________
#Тут будет интерфейс типо
print('_____________________________________________________________________________________________________________________')
print('Команда Эльвина | Лабораторная работа №2 | Построение интерполирующего кубического сплайна')
print('_____________________________________________________________________________________________________________________')
print('Выберите тип задачи:')
print('Тестовая - 0')
print('----------------------------')
print('Основная №1 - 1')
print('Основная №2 - 2')
print('Основная №3 - 3')
print('----------------------------')
print('Осцилирующая основная №1 - 4')
print('Осцилирующая основная №2 - 5')
print('Осцилирующая основная №3 - 6')
task = int(input('Номер задачи: '))
print('Введите размерность сетки:')
n = int(input('n = '))
if (task == 0):
   funcflag = 'TEST_1'
   a = -1
   b = 1
elif (task == 1):
   funcflag = 'MAIN_11'
   a = 1
   b = pi
elif (task == 2):
   funcflag = 'MAIN_12'
   a = 0.2
   b = 2
elif (task == 3):
   funcflag = 'MAIN_13'
   a = 2
   b = 4
elif (task == 4):
   funcflag = 'MAIN_21'
   a = 1
   b = pi
elif (task == 5):
   funcflag = 'MAIN_22'
   a = 0.2
   b = 2
elif (task == 6):
   funcflag = 'MAIN_23'
   a = 2
   b = 4

main()
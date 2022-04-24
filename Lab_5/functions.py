from numpy import sqrt, sin, cos, pi, array
from sympy import ln
import matplotlib.pyplot as plt
import pandas as pd

#?________________ОСНОВНЫЕ ДАННЫЕ________________
def f(funcflag, x):
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
        return ln(x + 1)/(x + 1)
    elif funcflag == 'MAIN_13':
        return ln(x + 1)/x

    # TODO Основная задача с осцилирующей функцией
    elif funcflag == 'MAIN_21':
        return sqrt(x)*sin(x) + cos(10*x)
    elif funcflag == 'MAIN_22':
        return ln(x + 1)/(x + 1) + cos(10*x)
    elif funcflag == 'MAIN_23':
        return ln(x + 1)/x + cos(10*x)

def df(funcflag, x):
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
        return (1 - ln(x + 1))/((x + 1)**2)
    elif funcflag == 'MAIN_13':
        return 1/(x*x*x + x**2) - ln(x + 1)/(x**2)

    # TODO Основная задача с осцилирующей функцией
    elif funcflag == 'MAIN_21':
        return sin(x)/(2*sqrt(x)) + sqrt(x)*cos(x) - 10*sin(10*x)
    elif funcflag == 'MAIN_22':
        return (1 - ln(x + 1))/((x + 1)**2) - 10*sin(10*x)
    elif funcflag == 'MAIN_23':
        return 1/(x*x*x + x**2) - ln(x + 1)/(x**2) - 10*sin(10*x)

def ddf(funcflag, x):
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
        return (-x - 3 + 2*ln(x + 1))/((x + 1)**4)
    elif funcflag == 'MAIN_13':
        return -(3*(x**2) + 2*x)/((x*x*x + x**2)**2) - ((x**2)/(x + 1) -2*x*ln(x + 1))/(x**4)

    # TODO Основная задача с осцилирующей функцией
    elif funcflag == 'MAIN_21':
        return cos(x)/sqrt(x) - sin(x)/(4*sqrt(x*x*x)) - sqrt(x)*sin(x) - 100*cos(10*x)
    elif funcflag == 'MAIN_22':
        return (-x - 3 + 2*ln(x + 1))/((x + 1)**4) - 100*cos(10*x)
    elif funcflag == 'MAIN_23':
        return -(3*(x**2) + 2*x)/((x*x*x + x**2)**2) - ((x**2)/(x + 1) -2*x*ln(x + 1))/(x**4) - 100*cos(10*x)

def spline(a,b,c,d, X, x, n):
    """Каноническая запись сплайна"""
    for i in range(1, n + 1):
        if X[i - 1] <= x <= X[i]:
            return a[i] + b[i]*(x - X[i]) + 0.5*c[i]*(x - X[i])**2 + (d[i]/6)*(x - X[i])**3

def dspline(a,b,c,d, X, x, n):
    """Первая производная сплайна"""
    for i in range(1, n + 1):
        if X[i - 1] <= x <= X[i]:
            return b[i] + c[i]*(x - X[i]) + (d[i]/2)*(x - X[i])**2

def ddspline(a,b,c,d, X, x, n):
    """Вторая производная сплайна"""
    for i in range(1, n + 1):
        if X[i - 1] <= x <= X[i]:
            return c[i] + d[i]*(x - X[i])

#?________________РАСЧЕТЫ________________
def coef(a, b, n, my1, my2, flag):
    """Функция для расчетов\\
    a - левая граница\\
    b - правая граница\\
    n - размерность сетки\\
    my1 - левое ГУ\\
    my2 - правое ГУ"""
    h = (b - a)/n
    X = [0]*(n + 1)
    A, B, C, D = [0]*(n + 1), [0]*(n + 1), [0]*(n + 1), [0]*(n + 1)
    alpha, beta = [0]*(n + 1), [0]*(n + 1)
    for i in range(0, n + 1):
        X[i] = a + i*h
        A[i] = f(flag, X[i])
    
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

    return X, A, B, C, D


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

def graf(x, y, funcflag):
    """Функция отрисовки и сохранения графиков для всех задач"""
    fig = plt.figure(figsize=(12,5))
    if funcflag == 'TEST_1':
        plt.subplot (1, 3, 1)
        plt.plot(x, y, marker='.')
        plt.grid()
        plt.title ("Функция")

        y = fun_special_graf(0, x)
        plt.subplot (1, 3, 2)
        plt.plot(x, y, marker='.')
        plt.grid()
        plt.title ("Первая производная")

        y = fun_special_graf(1, x)
        plt.subplot (1, 3, 3)
        plt.plot (x, y, marker='.')
        plt.grid()
        plt.title ("Вторая производная")

        plt.savefig("Test_graf.jpg")
        plt.show()
    else:
        plt.subplot (1, 3, 1)
        plt.plot(x, y, marker='.')
        plt.grid()
        plt.title ("Функция")

        y = df(funcflag, array(x))
        plt.subplot (1, 3, 2)
        plt.plot(x, y, marker='.')
        plt.grid()
        plt.title ("Первая производная")

        y = ddf(funcflag, array(x))
        plt.subplot (1, 3, 3)
        plt.plot (x, y, marker='.')
        plt.grid()
        plt.title ("Вторая производная")

        plt.savefig("Main_graf.jpg")
        plt.show()

#?________________ДАТАФРЕЙМ________________
def data(X, A, B, C, D):
    """Создание таблиц с данными по текущей задачи, принимает подсчитанные коэффициенты из функции coef"""
    table = pd.DataFrame(data = {})
    for i in range(1, len(X)):
        row = {'i': i, 'X[i-1]': X[i-1], 'X[i]': X[i], 'a[i]': A[i], 'b[i]': B[i], 'c[i]': C[i], 'd[i]': D[i]}
        table = table.append(row, ignore_index=True)
    table.to_csv('table_task.csv', index=False)    

def main(a, b, n, my1, my2, flag):
    """Принимает данные из протоинтерфейса и является основной функцией запуска программы на счет"""
    x, y, b, c, d = coef(a, b, n, my1, my2, flag)
    graf(x, y, flag)
    data(x, y, b, c, d)
    



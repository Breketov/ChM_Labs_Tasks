from functions import *
cheb, omg = 0, 0
if __name__ == "__main__":
    print('_____________________________________________________________________________________________________________________')
    print('Команда Эльвина | Лабораторная работа №1 | Решение задачи Дирихле для уравнения Пуассона')
    print('Выберите ступень:')
    print('Ступень №1 МВР - 1')
    print('Ступень №2 Чебышев - 2')
    print('Ступень №2 МПИ - 3')
    print('Ступень №3 - 4')
    print('----------------')
    part = int(input('Ступень: '))
    if part == 1:
        print('Эльвин Джафарзаде | Ступенька №1 | Метод верхней релаксации')
    elif part == 2:
        print('Бекетов Евгений | Ступенька №2 | Метод с Чебышевским набором параметров')
    elif part == 3:
        print('Вивас Каролина  | Ступенька №2 | Метод простой итерации на стандартной области')
    elif part == 4:
        print('Катранов Кирилл | Ступенька №3 | Метод простой итерации на нестандартной области')
    print('_____________________________________________________________________________________________________________________')
    if part == 1 or part == 2 or part == 3:
        print('Введите все необходимые данные')
        print('Выберите задачу:')
        print('Тестовая - 0')
        print('Основная - 1')
        print('----------------------------')
        task = int(input('Задача: '))
    elif part == 4:
        task = 0
        print('Решается тестовая задача')
    print('Введите размерность сетки:')
    n = int(input('Разбиение по x: n = '))
    m = int(input('Разбиение по x: m = '))
    if part == 1:
        if task == 0:
            print('Введите параметр релаксации (0, 2):')
            omg = float(input('omega = '))
        elif task == 1:
            print('Введите параметр релаксации (0, 2) для обычной сетки:')
            omg1 = float(input('omega1 = '))
            print('Введите параметр релаксации (0, 2) для контрольной сетки:')
            omg2 = float(input('omega2 = '))
            omg = [omg1, omg2]
    elif part == 2:
        print('Введите параметр метода k (натуральное число):')
        cheb = int(input('k = '))
    print('Введите критерий выхода по точности epsilon:')
    eps = float(input('epsilon = '))
    print('Введите число шаго Nmax:')
    Nmax = int(input('Nmax = '))
    print('---------------------------------------------------------------------------------------------------')
    if task == 0:
        test_task(n, m, eps, Nmax, omg, cheb, part)
    elif task == 1:
        main_task(n, m, eps, Nmax, omg, cheb, part)
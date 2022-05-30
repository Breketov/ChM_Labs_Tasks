import functions as func

def main(n , m, eps, Nmax, cheb, omg, task, part):
    if task == 0:
        func.solve_test(n , m, eps, Nmax, cheb, omg, part)
    elif task == 1:
        func.task_main(n , m, eps, Nmax, cheb, omg, part)

if __name__ == "__main__":
    cheb = 0
    omg = 0
    print('_____________________________________________________________________________________________________________________')
    print('Команда Эльвина | Лабораторная работа №1 | Решение задачи Дирихле для уравнения Пуассона')
    print('Выберите ступень:')
    print('Ступень №1 - 1')
    print('Ступень №2 - 2')
    print('Ступень №3 - 3')
    print('----------------')
    part = int(input('Ступень: '))
    if part == 1:
        print('Эльвин Джафарзаде | Ступенька №1 | Метод верхней релаксации')
    elif part == 2:
        print('Бекетов Евгений | Ступенька №2 | Метод с Чебышевским набором параметров')
    else:
        print('Вивас Каролина  | Ступенька №3 | Метод простой итерации на стандартной области')
        print('Катранов Кирилл | Ступенька №3 | Метод простой итерации на нестандартной области')
    print('_____________________________________________________________________________________________________________________')
    print('Введите все необходимые данные')
    print('Выберите задачу:')
    print('Тестовая - 0')
    print('Основная - 1')
    print('----------------------------')
    task = int(input('Задача: '))
    print('Введите размерность сетки:')
    n = int(input('Разбиение по x: n = '))
    m = int(input('Разбиение по x: m = '))
    if part == 1:
        print('Введите параметр релаксации (0, 2):')
        omg = float(input('omega = '))
    elif part == 2:
        print('Введите параметр метода k (натуральное число):')
        cheb = int(input('k = '))
    print('Введите критерий выхода по точности epsilon:')
    eps = float(input('epsilon = '))
    print('Введите число шаго Nmax:')
    Nmax = int(input('Nmax = '))
    print('---------------------------------------------------------------------------------------------------')
    main(n , m, eps, Nmax, cheb, omg, task, part)
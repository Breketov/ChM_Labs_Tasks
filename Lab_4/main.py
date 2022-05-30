import functions as func

def main(n , m, eps, Nmax, cheb, task):
    if task == 0:
        func.solve_test(n , m, eps, Nmax, cheb)
    elif task == 1:
        func.task_main(n , m, eps, Nmax, cheb)

if __name__ == "__main__":
    print('_____________________________________________________________________________________________________________________')
    print('Команда Эльвина | Лабораторная работа №1 | Решение задачи Дирихле для уравнения Пуассона')
    print('Бекетов Евгений | Ступенька №2 | Метод с Чебышевским набором параметров')
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
    print('Введите параметр метода k (натуральное число):')
    cheb = int(input('k = '))
    print('Введите критерий выхода по точности epsilon:')
    eps = float(input('epsilon = '))
    print('Введите число шаго Nmax:')
    Nmax = int(input('Nmax = '))
    print('---------------------------------------------------------------------------------------------------')
    main(n , m, eps, Nmax, cheb, task)

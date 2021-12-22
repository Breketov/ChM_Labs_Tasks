import pandas as pd
import numpy as np
class test:
    def __init__(self, num_part):
        """
        Определение параметров и компонент стационарного уравнения теплоемкости
        """

        self.n = num_part
        self.node = self.n + 1

        self.start, self.end = 0, 1
        # шаг интегрирования
        self.h = (self.end - self.start) / self.n

        # целый шаг
        self.x = [self.start + i * self.h for i in range(self.node)]
        # половинный шаг
        self.x2 = [self.start + (i + 0.5) * self.h for i in range(self.node - 1)]

        self.mu = np.array([1, 0])
        self.kappa = np.array([0, 0])
        # точка разрыва
        self.ksi = 0.3

        # для a
        self.k1 = lambda x: 2.09
        self.k2 = lambda x: 0.09

        # для d
        self.q1 = lambda x: 0.3
        self.q2 = lambda x: 0.09

        # для phi
        self.f1 = lambda x: 1
        self.f2 = lambda x: np.sin(0.3*np.pi)

    def analytic(self):
        """
        Построение точного решения
        """

        c1 = -0.9603081818828776
        c2 = -1.373025151450456
        c3 = -2.459846416926839
        c4 = -6.258903608512381

        def first(x):
            # c3 = ((a * np.exp(a * (y - x)) * (self.mu[0] - f1) - b * np.exp(b * (t - x)) * \
            # (self.mu[1] - f2)) * (np.exp(a * x) - np.exp(a * (2 * y - x))) + (np.exp(a * \
            # 	(y - x)) * (self.mu[0] - f1) - np.exp(b * (t - x)) * (self.mu[1] - f2) + f1 - f2) * \
            # (a * np.exp(a * x) + a * np.exp(a * (2*y-x))))/((np.exp(b * x) - np.exp(b * (2 * t - x))) * \
            # (a * np.exp(a * x) + a * np.exp(a * (2 * y - x))) - (np.exp(a * x) - np.exp(a * (2 * y -x)))*\
            # (b * np.exp(b * x) + b * np.exp(b * (2 * t - x))))
            # c1 = (c3 * (np.exp(b * x) - np.exp(b * (2 * t - x))) + np.exp(b * (t - x)) * (self.mu[1] - f2) - \
            # np.exp(a * (y - x)) * (self.mu[0] - f1) + f2 - f1) / (np.exp(a * x) - np.exp(a * (2 * y - x)))
            # c2 = np.exp(a * y) * (self.mu[0] - f1 - c1 * np.exp(a * y))
            # c4 = np.exp(b * t) * (self.mu[1] - f2 - c3 * np.exp(b * t))

            # # лучше пока что нет
            # x1 = (-np.sqrt(2) * np.exp(1) + np.exp(1) + np.sqrt(2) * np.exp(1 / 2) + 1) / (np.exp(1) + np.exp(3/2) + np.exp(1/2) + 1)
            # x2 = (np.sqrt(2) * np.exp(1) + np.exp(3/2) - np.sqrt(2) * np.exp(1 / 2) + np.exp(1/2)) / (np.exp(1) + np.exp(3/2) + np.exp(1/2) + 1)
            # return x1 * np.exp(x * np.sqrt(2)) + x2 * np.exp(-x*np.sqrt(2))

            # return x1 * np.exp(np.sqrt(2) * x) + x2 * np.exp(-np.sqrt(2) * x)
            return c1 * np.exp(np.sqrt(30/209) * x) + c2 * np.exp(-np.sqrt(30/209) * x) + 10/3

        def second(x):
            # np.sqrt(2) * np.exp(np.sqrt(2) / 2) + np.sqrt(2) * np.exp(-np.sqrt(2) / 2) + 2 * np.sqrt(2)) / ((1 - np.sqrt(2)) * \
            # (np.exp((np.sqrt(2) - 3) / 2) + np.exp((np.sqrt(2) - 1) / 2)) + (-1-np.sqrt(2)) * (np.exp((-np.sqrt(2) - 3) / 2) + \
            # np.exp((-np.sqrt(2) - 1) / 2)))

            # return (2 * np.exp(-1) - c2 * np.exp(-2)) * np.exp(x) + c2 * np.exp(-x) - 1

            # # лучше пока что нет
            # x3 = (2 * np.exp(1) + np.sqrt(2) * np.exp(1/2) - np.sqrt(2) + 2) / (np.exp(2) + np.exp(1) + np.exp(3/2) + np.exp(1/2))
            # x4 = (-np.sqrt(2) * np.exp(2) + 2 * np.exp(2) + 2 * np.exp(1) + np.sqrt(2) * np.exp(3/2)) / (np.exp(1) + np.exp(3/2) + np.exp(1/2) + 1)
            # return x3 * np.exp(x) + x4 * np.exp(-x) - 1

            # return x3 * np.exp(x) + x4 * np.exp(-x) + 1
            return c3 * np.exp(x) + c4 * np.exp(-x) + 100*np.sin(0.3*np.pi)/9

        # return np.concatenate(([self.mu[0]], \
        #     np.array([first(self.x[i]) if self.x[i] < self.ksi else second(self.x[i]) for i in range(1, self.node - 1)]), \
        #     [self.mu[1]]
        # ))

        return np.array([first(self.x[i]) if self.x[i] < self.ksi else second(self.x[i]) for i in range(self.node)])

    def numerical(self):
        """
        Построение численного решения
        """

        # введем новые переменные
        self.phi = np.empty(self.n - 1, np.double)
        self.d = np.empty(self.n - 1, np.double)
        self.a = np.empty(self.n, np.double)

        # подсчет phi, d
        j = 1
        for i in range(self.n - 1):
            if self.ksi >= self.x2[i + 1]:
                self.phi[i] = self.f1(self.x[j])
                self.d[i] = self.q1(self.x[j])

            elif self.ksi <= self.x2[i]:
                self.phi[i] = self.f2(self.x[j])
                self.d[i] = self.q2(self.x[j])

            elif self.x2[i] < self.ksi < self.x2[i + 1]:
                self.phi[i] = (self.f1(self.x[j]) + self.f2(self.x[j])) / 2
                self.d[i] = (self.q1(self.x[j]) + self.q2(self.x[j])) / 2

            j += 1

        # подсчет a
        j = 1
        for i in range(self.n):
            if self.ksi >= self.x[j]:
                self.a[i] = self.k1(self.x2[i])

            elif self.ksi <= self.x[j - 1]:
                self.a[i] = self.k2(self.x2[i])

            elif self.x[j - 1] < self.ksi < self.x[j]:
                self.a[i] = 1 / (1 / (2 * self.k1(self.x[i])) + 1 / (2 * self.k2(self.x[i])))

            j += 1

        # print()
        # print("x: ", self.x)
        # print("x2: ", self.x2)
        # print("phi: ", self.phi)
        # print("d: ", self.d)
        # print("a: ", self.a)
        # print()

        # # нижняя, главная, верхняя диагонали
        # bottom = np.diag(self.calcdiag(-1), k=-1)
        # middle = np.diag(self.calcdiag(0), k=0)
        # high = np.diag(self.calcdiag(1), k=1)

        # # Построим разностную схему в матричном виде: (AAv = b)
        # self.AA = bottom + middle + high

        # self.A = np.diag(self.AA, k=-1)[:-1]
        # self.B = np.diag(self.AA, k=1)[1:]
        # self.C = (-1) * np.diag(self.AA)[1:-1]
        # self.b = np.concatenate(([self.mu[0]], -self.phi, [self.mu[1]]))

        self.A = np.array([self.a[i] / (self.h ** 2) for i in range(self.n - 1)])
        self.B = np.array([self.a[i] / (self.h ** 2) for i in range(1, self.n)])
        self.C = np.array(
            [self.a[i] / (self.h ** 2) + self.a[i + 1] / (self.h ** 2) + self.d[i] for i in range(self.n - 1)])

        # print("C: ", self.C)
        # print("B: ", self.B)
        # print("A: ", self.A)
        # print("AA: \n", self.AA)
        # print("b: \n", self.b)

        # найдем v
        return self.run_through()

    def calcdiag(self, k):
        if k == 0:
            # главная диаг = 1 a1 ... an-1 1
            return np.concatenate(
                ([1], (-1) * np.array(
                    [self.a[i] / (self.h ** 2) + self.a[i + 1] / (self.h ** 2) + self.d[i] for i in range(self.n - 1)]),
                 [1])
            )
        elif k == -1:
            # нижняя диаг = a1 ... an-1 -kappa2
            return np.concatenate(
                (np.array([self.a[i] / (self.h ** 2) for i in range(self.n - 1)]), [-self.kappa[1]])
            )
        else:
            # верхняя диаг = -kappa1 a2 ... an
            return np.concatenate(
                ([-self.kappa[0]], np.array([self.a[i] / (self.h ** 2) for i in range(1, self.n)]))
            )

    def direct(self):
        """
        Прямой ход прогонки
        """

        # находим параметры
        alpha = np.empty(self.n, np.double)
        beta = np.empty(self.n, np.double)

        # нач условие (3)
        alpha[0] = self.kappa[0]
        beta[0] = self.mu[0]

        # i = 1, n - 1
        for i in range(self.n - 1):
            alpha[i + 1] = self.B[i] / (self.C[i] - self.A[i] * alpha[i])
            beta[i + 1] = (self.phi[i] + self.A[i] * beta[i]) / (self.C[i] - self.A[i] * alpha[i])

        return alpha, beta

    def reverse(self):
        """
        Обратный ход прогонки
        """

        # находим численное решение
        y = np.empty(self.n, np.double)

        # нач условие (6)
        y[-1] = (-self.kappa[1] * self.beta[-1] - self.mu[1]) / (self.kappa[1] * self.alpha[-1] - 1)

        # i = 0, n - 1
        for i in range(self.n - 2, -1, -1):
            y[i] = self.alpha[i + 1] * y[i + 1] + self.beta[i + 1]

        return y

    def run_through(self):
        """
        Метод прогонки
        """

        self.alpha, self.beta = self.direct()
        # print("alp = ", self.alpha, "\nbeta = ", self.beta)
        return self.reverse()

    def Solution(self):
        """
        Составление таблицы
        """

        # print('xi = ', self.x)
        # print('x2i = ', self.x2)
        # print()

        self.u = self.analytic()
        self.v = np.concatenate(([self.mu[0]], self.numerical()))

        # Разница численного и точного решений
        self.dif = np.absolute(self.u - self.v)

        # таблица
        ndata = pd.DataFrame({
            '№ узла': np.arange(self.node), 'x': np.round(self.x, len(str(self.h))), 'u(x)': np.round(self.u, 14),
            'v(x)': np.round(self.v, 14), \
            '|u(x) - v(x)|': np.round(self.dif, 14)
        })

        # записи
        note = np.array(
            [self.n, self.dif[self.node // 2 if self.node % 2 == 1 else int(self.node / 2 - 1)], np.argmax(self.dif)])

        return note, ndata
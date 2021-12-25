#                       Лабораторная работа №2
#                       Версия 0.59

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#! Функции
k1 = lambda x: x*x + 2
k2 = lambda x: x*x
q1 = lambda x: x
q2 = lambda x: x*x
f1 = lambda x: 1
f2 = lambda x: np.sin(np.pi*x)

#! Матричные коэффициенты
def coef_matrix(n, a, d, fi):
   A = np.array([a[i]/((1/n)**2) for i in range(n - 1)])
   B = np.array([a[i]/((1/n)**2) for i in range(1, n)])
   C = np.array([a[i]/((1/n)**2) + a[i + 1]/((1/n)**2) + d[i] for i in range(n - 1)])
   return A, B, C

#! Прогонка
def progonka(n, A, B, C, a, d ,fi):
   #* Прямая прогонка
   alfa = np.empty(n)
   beta = np.empty(n)
   alfa[0] = kappa[0]
   beta[0] = my[0]
   for i in range(n - 1):
      alfa[i + 1] = B[i]/(C[i] - A[i]*alfa[i])
      beta[i + 1] = (fi[i] + A[i]*beta[i])/(C[i] - A[i]*alfa[i])
   
   #* Обратная прогонка
   y = np.empty(n)
   y[-1] = (-kappa[1]*beta[-1] - my[1])/(kappa[1]*alfa[-1] - 1)
   for i in range(n - 2, -1, -1):
      y[i] = alfa[i + 1]*y[i + 1] + beta[i + 1]
   return y 

#TODO------------------------------------ТЕСТОВАЯ ЗАДАЧА------------------------------------
#! Точное решение задачи
def true_plot(x_main):
   C1 = -0.9603081818828
   C2 = -1.3730251514504
   C3 = -2.4598464169268
   C4 = -6.2589036085123
   u1, u2 = [], []
   x1, x2 = [], []
   for x in x_main:
      if x <= ksi:
         u1.append(C1*np.exp(np.sqrt(ksi/(ksi**2 + 2))*x) + C2*np.exp(-np.sqrt(ksi/(ksi**2 + 2))*x) + 1/ksi)
         x1.append(x)
      elif x >= ksi:
         u2.append(C3*np.exp(x) + C4*np.exp(-x) + np.sin(np.pi*ksi)/0.09)
         x2.append(x)
   plt.plot(x1, u1,'o-', linewidth = 2.0, label='траектория до точки разрыва')
   plt.plot(x2, u2,'o-', linewidth = 2.0, label='траетория после точки разрыва')
   plt.legend()
   plt.xlabel('x')
   plt.ylabel('u(x)')
   plt.grid()
   plt.savefig('График истинной траектории.png')
   plt.show()
   return u1 + u2

#! Коэффициенты тестовой задачи
def test_coef(x_main, x_aux, n, a, d, fi):
   for i in range(n):
      if (x_main[i + 1] <= 0.3):
         a[i] = 2.09
      elif (x_main[i] >= 0.3):
         a[i] = 0.09
      elif x_main[i] < 0.3 < x_main[i + 1]:
         a[i] = 1/(1/(2*k1(0.3)) + 1/(2*k2(0.3)))

   for i in range(n - 1):
      if (x_aux[i + 1] <= 0.3):
         d[i] = 0.3
      elif (x_aux[i] >= 0.3):
         d[i] = 0.09
      elif x_aux[i] < 0.3 < x_aux[i + 1]:
         d[i] = (q1(0.3) + q2(0.3))/2
   
   for i in range(n - 1):
      if (x_aux[i + 1] <= 0.3):
         fi[i] = 1
      elif (x_aux[i] >= 0.3):
         fi[i] = np.sin(0.3*np.pi)
      elif x_aux[i] < 0.3 < x_aux[i + 1]:
         fi[i] = (f1(0.3) + f2(0.3))/2
   return a, d, fi

#! Тестовая задача
def test_task(n, x_main, x_aux):
   a = np.empty(n)
   d = np.empty(n - 1)
   fi = np.empty(n - 1)
   a, d ,fi = test_coef(x_main, x_aux, n, a, d, fi)
   A, B, C = coef_matrix(n, a, d, fi)
   v = list(progonka(n, A, B, C, a, d ,fi))
   return v

#TODO------------------------------------ОСНОВНАЯ ЗАДАЧА------------------------------------
#! Коэффициенты основной задачи
def main_coef(x_main, x_aux, n, a, d, fi):
   for i in range(n):
      if (x_main[i + 1] <= 0.3):
         a[i] = k1(x_aux[i])
      elif (x_main[i] >= 0.3):
         a[i] = k2(x_aux[i])
      elif x_main[i] < 0.3 < x_main[i + 1]:
         a[i] = 1/(n*(((0.3 - x_main[i + 1])/(k1((x_main[i] + 0.3)/2))) + ((x_main[i + 1] - 0.3)/(k2((0.3 + x_main[i + 1])/2)))))

   for i in range(n - 1):
      if (x_aux[i + 1] <= 0.3):
         d[i] = q1(x_main[i + 1])
      elif (x_aux[i] >= 0.3):
         d[i] = q2(x_main[i + 1])
      elif x_aux[i] < 0.3 < x_aux[i + 1]:
         d[i] = n*(q1((x_aux[i] + 0.3)/2)*(0.3 - x_aux[i]) + q2((0.3 + x_aux[i + 1])/2)*(x_aux[i + 1] - 0.3))
   
   for i in range(n - 1):
      if (x_aux[i + 1] <= 0.3):
         fi[i] = f1(x_main[i + 1])
      elif (x_aux[i] >= 0.3):
         fi[i] = f2(x_main[i + 1])
      elif x_aux[i] < 0.3 < x_aux[i + 1]:
         fi[i] = n*(f1((x_aux[i] + 0.3)/2)*(0.3 - x_aux[i]) + f2((0.3 + x_aux[i + 1])/2)*(x_aux[i + 1] - 0.3))
   return a, d, fi

#! Основная задача
def main_task(n, x_main, x_aux):
   a = np.empty(n)
   d = np.empty(n - 1)
   fi = np.empty(n - 1)
   a, d ,fi = main_coef(x_main, x_aux, n, a, d, fi)
   A, B, C = coef_matrix(n, a, d, fi)
   v = list(progonka(n, A, B, C, a, d ,fi))
   return v

#? Запуск просчетов
def solution(h, n):
   #TODO----------ТЕСТОВАЯ ЧАСТЬ----------
   x_main = [x_start + i*h for i in range(n + 1)]
   x_aux = [x_start + (i + 0.5)*h for i in range(n)]
   v1 = test_task(n, x_main, x_aux)
   u = true_plot(x_main)
   u[0] = 1
   u[-1] = 0
   v1.insert(0, 1)
   E = list(np.abs(np.array(u) - np.array(v1)))

   for i in range(len(x_main)):
      x_main[i] = round(x_main[i], 14)
   Node = [i for i in range(len(x_main))]
   data_test = {'№ Узла': Node, 'x': x_main, 'v': v1, 'u': u, '|u - v|': E}
   data = pd.DataFrame(data = data_test)
   data.to_csv("data_test.csv", index=False)

   max_error_E = 0
   for i in range(len(v1)):   
      if E[i] > max_error_E:
         max_error_E = E[i]
   
   max_E = E.index(max_error_E)
   x1_max = x_main[max_E]
   
   #* График разность численнного и аналитического решения
   plt.plot(x_main, E)
   plt.xlabel('x')
   plt.ylabel('|u - v|')
   plt.legend()
   plt.grid()
   plt.savefig('Разность численнного и аналитического решения.png')
   plt.show()
   
   #TODO----------ОСНОВНАЯ ЧАСТЬ----------
   x_main = [x_start + i*h for i in range(n + 1)]
   x_aux = [x_start + (i + 0.5)*h for i in range(n)]
   v1 = main_task(n, x_main, x_aux)
   n = int(2*n)
   h = h/2
   x_main = [x_start + i*h for i in range(n + 2)]
   x_aux = [x_start + (i + 0.5)*h for i in range(n + 1)]
   v2 = list(main_task(n, x_main, x_aux))
   v1.insert(0, 1)
   v2.insert(0, 1)
   v2 = v2[::2]
   e = np.abs(np.array(v1) - np.array(v2))
   
   for i in range(len(x_main)):
      x_main[i] = round(x_main[i], 14)
   Node = [i for i in range(len(x_main))][::2]
   data_test = {'№ Узла': Node, 'x': x_main[::2], 'v1': v1, 'v2': v2, '|v2 - v1|': e}
   data = pd.DataFrame(data = data_test)
   data.to_csv("data_main.csv", index=False)

   e = list(e)
   max_error_e = 0
   for i in range(len(v1)):
      if e[i] > max_error_e:
         max_error_e = e[i]
   
   max_e = e.index(max_error_e)
   x2_max = x_main[max_e]
   
   #* График численнного и афро-численного решения
   plt.plot(x_main[::2], v1)
   plt.plot(x_main[::2], v2, 'x-')
   plt.xlabel('x')
   plt.ylabel('v1(x)   v2(x)')
   plt.legend()
   plt.grid()
   plt.savefig('Численный траетории.png')
   plt.show()

   #* График разности численнного и афро-численного решения
   plt.plot(x_main[::2], e)
   plt.xlabel('x')
   plt.ylabel('|v1 - v2|')
   plt.legend()
   plt.grid()
   plt.savefig('Разность численнного решения и численного с половинным шагом.png')
   plt.show()

   print("Справка")
   print('Максимальная погрешность тестовой задачи: ', max_error_E, '  ', 'при x = ', x1_max)
   print('Максимальная погрешность основной задачи: ', max_error_e, '  ', 'при x = ', x2_max)


#* Терминал
print('_____________________________________________________________________________________________________________________')
print('Команда Эльвина | Лабораторная работа №2 | Решение краевых задач для ОДУ')
print('_____________________________________________________________________________________________________________________')
print('В рамках данной лабораторной работы решается тествая и основная задачи с определенной сеткой.')
print('Задайте размерность сетки')
n = int(input('n = '))

#* Результат работы
x_start = 0
x_end = 1
h = 1/n
kappa = [0, 0]
my = [1, 0]
ksi = 0.3

solution(h, n)
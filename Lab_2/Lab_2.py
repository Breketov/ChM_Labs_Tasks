#                                Лабораторная работа №2
#                                Версия 0.6.12
import decimal
from math import sin
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import *

#TODO Основные параметры
x_start = 0
x_end = 1
kappa = [0, 0]
my = [1, 0]
ksi = 0.3

#!-------------------------------ТЕСТОВАЯ ЧАСТЬ-------------------------------#
#* Нахождение всех коэффициентов и параметров тестовой задачи
#* Построение решения
def test_task(n):
   h = 1/n
   x_main = [x_start + i*h for i in range(n + 1)]
   x_aux = [x_start + (i + 0.5)*h for i in range(n)]
   a, d ,fi = test_coef(x_main, x_aux, n)
   A, B, C = coef_matrix(n, a, d, fi)
         #* Запуск прогонки
   v = progonka(n, A, B, C, a, d ,fi)
         #* Истинная траектория
   u = true_solution(x_main)

      #* Погрешность и запись в дата 
   E = list(np.abs(np.array(u) - np.array(v)))
   for i in range(len(x_main)):
      x_main[i] = round(x_main[i], 13)
   Node = [i for i in range(len(x_main))]

      #* Данные для справки
   max_error_E = 0
   for i in range(len(E)):   
      if E[i] > max_error_E:
         max_error_E = E[i]
   max_E = E.index(max_error_E)
   x_max = x_main[max_E]

   data_test = {'№ Узла': Node, 'x': x_main, 'v': v, 'u': u, '|u - v|': E}
   data = pd.DataFrame(data = data_test)
   data.to_csv("data_test.csv", index=False)
   a = list(a)
   a.insert(0, 0)
   data_dop = {'x': x_main, 'a': a}
   data = pd.DataFrame(data = data_dop)
   data.to_csv("data_dop.csv", index=False)
   
   data_dop2 = {'d': d, 'fi': fi, 'A': A, 'B': B, 'C': C}
   data = pd.DataFrame(data = data_dop2)
   data.to_csv("data_dop2.csv", index=False)
   #plot_test(x_main, u, v, E)
   print('Максимальная погрешность тестовой задачи: ', max_error_E, '  ', 'при x = ', x_max)

#* Коэффициенты тестовой задачи
def test_coef(x_main, x_aux, n):
   a = np.empty(n)
   for i in range(1, n + 1):
      if (node(i - 1, n) <= 0.3 and node(i, n) <= 0.3):
         a[i - 1] = 0.3**2 + 2
      elif (node(i - 1, n) >= 0.3 and node(i, n) >= 0.3):
         a[i - 1] = 0.3**2
      elif (node(i - 1, n) < 0.3 < node(i, n)):
         a[i - 1] = 1/(n*(0.3/2.09 - node(i - 1, n)/2.09 + node(i, n)/0.09 - 0.3/0.09))

   d = np.empty(n - 1)
   for i in range(1, n):
      if (node(i - 0.5, n) <= 0.3 and node(i + 0.5, n) <= 0.3):
         d[i - 1] = 0.3
      elif (node(i - 0.5, n) >= 0.3 and node(i + 0.5, n) >= 0.3):
         d[i - 1] = 0.3**2
      elif (node(i - 0.5, n) < 0.3 < node(i + 0.5, n)):
         d[i - 1] = n*(0.3*0.3 - 0.3*node(i - 0.5, n) + 0.09*node(i + 0.5, n) - 0.09*0.3)

   fi = np.empty(n - 1)
   for i in range(1, n):
      if (node(i - 0.5, n) <= 0.3 and node(i + 0.5, n) <= 0.3):
         fi[i - 1] = 1
      elif (node(i - 0.5, n) >= 0.3 and node(i + 0.5, n) >= 0.3):
         fi[i - 1] = np.sin(0.3*np.pi)
      elif node(i - 0.5, n) < 0.3 < node(i + 0.5, n):
         fi[i - 1] = n*(0.3 - node(i - 0.5, n) + np.sin(0.3*np.pi)*node(i + 0.5, n) - np.sin(0.3*np.pi)*0.3)
   return a, d, fi

#* Точное решение задачи
def true_solution(x_list):
   C1 = -0.9603081818828
   C2 = -1.3730251514504
   C3 = -2.4598464169268
   C4 = -6.2589036085123
   u1, u2, x1, x2 = [], [], [], []
   for x in x_list:
      if x <= ksi:
         u1.append(C1*np.exp(np.sqrt(0.3/(2.09))*x) + C2*np.exp(-np.sqrt(0.3/(2.09))*x) + 1/0.3)
         x1.append(x)
      elif x >= ksi:
         u2.append(C3*np.exp(x) + C4*np.exp(-x) + np.sin(np.pi*0.3)/0.09)
         x2.append(x)
   u = u1 + u2
   u[0] = my[0]
   u[-1] = my[1]
   return u


#!-------------------------------ОСНОВНАЯ ЧАСТЬ-------------------------------#
#* Нахождение всех коэффициентов и параметров основной задачи
#* Построение решения
def main_task(n):
         #* Обычная траектория
   h = 1/n
   x_main = [x_start + i*h for i in range(n + 1)]
   x_aux = [x_start + (i + 0.5)*h for i in range(n)]
   a, d ,fi = main_coef(x_main, x_aux, n)
   A, B, C = coef_matrix(n, a, d, fi)
         #* Запуск прогонки
   v1 = progonka(n, A, B, C, a, d ,fi)
         #* Траектория половинного шага
   h = h/2
   n = int(2*n)
   x_main = [x_start + i*h for i in range(n + 1)]
   x_aux = [x_start + (i + 0.5)*h for i in range(n)]
   a, d ,fi = main_coef(x_main, x_aux, n)
   A, B, C = coef_matrix(n, a, d, fi)
            #* Запуск прогонки
   v2 = progonka(n, A, B, C, a, d ,fi)[::2]
   
         #* Погрешность и запись в дата 
   e = list(np.abs(np.array(v1) - np.array(v2)))
   x_main = x_main[::2]
   for i in range(len(x_main)):
      x_main[i] = round(x_main[i], 13)
   Node = [i for i in range(len(x_main))]

         #* Данные для справки
   max_error_e = 0
   for i in range(len(e)):
      if e[i] > max_error_e:
         max_error_e = e[i] 
   max_e = e.index(max_error_e)
   x_max = x_main[max_e]

   data_main = {'№ Узла': Node, 'x': x_main, 'v1': v1, 'v2': v2, '|v2 - v1|': e}
   data = pd.DataFrame(data = data_main)
   data.to_csv("data_main.csv", index=False)

   #plot_main(x_main, v1, v2, e)
   print('Максимальная погрешность основной задачи: ', max_error_e, '  ', 'при x = ', x_max)

#* Коэффициенты основной задачи
def main_coef(x_main, x_aux, n):
   a = np.empty(n)
   for i in range(1, n + 1):
      if (node(i - 1, n) <= 0.3 and node(i, n) <= 0.3):
         a[i - 1] = k1(node(i - 0.5, n))
      elif (node(i - 1, n ) >= 0.3 and node(i, n) >= 0.3):
         a[i - 1] = k2(node(i - 0.5, n))
      elif (node(i - 1, n) < 0.3 < node(i, n)):
         a[i - 1] =(n*(((0.3 - node(i - 1, n))/(k1((node(i - 1, n) + 0.3)/2))) + ((node(i, n) - 0.3)/(k2((0.3 + node(i, n))/2)))))**(-1)

   d = np.empty(n - 1)
   for i in range(1, n):
      if (node(i - 0.5, n) <= 0.3 and node(i + 0.5, n) <= 0.3):
         d[i - 1] = q1(node(i, n))
      elif (node(i - 0.5, n) >= 0.3 and node(i + 0.5, n) >= 0.3):
         d[i - 1] = q2(node(i, n))
      elif (node(i - 0.5, n) < 0.3 < node(i + 0.5, n)):
         d[i - 1] = n*(q1((node(i - 0.5, n) + 0.3)/2)*(0.3 - node(i - 0.5, n)) + q2((0.3 + node(i + 0.5, n))/2)*(node(i + 0.5, n) - 0.3))
   
   fi = np.empty(n - 1)
   for i in range(1, n):
      if (node(i - 0.5, n) <= 0.3 and node(i + 0.5, n) <= 0.3):
         fi[i - 1] = f1(node(i, n))
      elif (node(i - 0.5, n) >= 0.3 and node(i + 0.5, n) >= 0.3):
         fi[i - 1] = f2(node(i, n))
      elif (node(i - 0.5, n) < 0.3 < node(i + 0.5, n)):
         fi[i - 1] = n*(f1((node(i - 0.5, n) + 0.3)/2)*(0.3 - node(i - 0.5, n)) + f2((0.3 + node(i + 0.5, n))/2)*(node(i + 0.5, n) - 0.3))
   return a, d, fi


#!-------------------------------Функциональая ЧАСТЬ-------------------------------#
#* Графики Тестовой задачи
def plot_test(x, u, v, E):
   #* Численная и истинная траектории
   plt.plot(x, u, linewidth = 4.0, label='Истинная траектория')
   plt.plot(x, v, 'x-', linewidth = 1.0, label='Численная траектория')
   plt.legend()
   plt.xlabel('x')
   plt.ylabel('u(x)   v(x)')
   plt.grid()
   plt.savefig('График численной и истинной траекторий.png')
   plt.show()

   #* График разности численнного и аналитического решения
   plt.plot(x, E, label='Разность численного и аналитичесского решений')
   plt.legend()
   plt.xlabel('x')
   plt.ylabel('|u(x) - v(x)|')
   plt.grid()
   plt.savefig('Разность численнного и аналитического решения.png')
   plt.show()

#* Графики Основной задачи
def plot_main(x, v1, v2, e):
   #* Численная и афрро-численная траектории
   plt.plot(x, v1, linewidth = 4.0, label='Траектория с обычным шагом')
   plt.plot(x, v2, 'x-', linewidth = 2.0, label='Траектория с половинным шагом')
   plt.legend()
   plt.xlabel('x')
   plt.ylabel('v1(x)   v2(x)')
   plt.grid()
   plt.savefig('График численных траекторий.png')
   plt.show()

   #* График разности численнного и афро-численного решения
   plt.plot(x, e, label='Разность численных решений')
   plt.legend()
   plt.xlabel('x')
   plt.ylabel('|v1(x) - v2(x)|')
   plt.grid()
   plt.savefig('Разность численнных решений решения.png')
   plt.show()

#* Функции
k1 = lambda x: x*x + 2
k2 = lambda x: x*x
q1 = lambda x: x
q2 = lambda x: x*x
f1 = lambda x: 1
f2 = lambda x: np.sin(np.pi*x)

#* Узел
def node(i, n):
   return i/n

#* Матричные коэффициенты
def coef_matrix(n, a, d, fi):
   A = np.array([a[i]*(n**2) for i in range(n - 1)])
   B = np.array([a[i + 1]*(n**2) for i in range(n - 1)])
   C = np.array([a[i]*(n**2) + a[i + 1]*(n**2) + d[i] for i in range(n - 1)])
   return A, B, C

#* Прогонка
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
   y[-1] = 0
   for i in range(n - 2, -1, -1):
      y[i] = alfa[i + 1]*y[i + 1] + beta[i + 1]
   y = list(y)
   y.insert(0, 1)
   return y 


#* Терминал
print('________________________________________________________________________________________________')
print('Команда Эльвина | Лабораторная работа №2 | Решение краевых задач для ОДУ')
print('________________________________________________________________________________________________')
print('В рамках данной лабораторной работы решается тествая и основная задачи с определенной сеткой.')
print('Задайте размерность сетки')
n = int(input('n = '))
print("Справка")
test_task(n)
main_task(n)
print('Конец работы программы')
print('________________________________________________________________________________________________')
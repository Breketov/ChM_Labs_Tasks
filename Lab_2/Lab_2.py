#                       Лабораторная работа №2
#                       Версия 0.46

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" 

# Подсчет коэффициентов
def TEST_coef(a, d, fi):
   for i in range(0, n):
      if ((node(i, n) < 0.3) and (node(i + 1, n) < 0.3)):
         a[i] = 2.09
      elif ((node(i, n) > 0.3) and (node(i + 1, n) > 0.3)):
         a[i] = 0.09
      else:
         a[i] = 1/(n*(0.3 - node(i - 1, n))/2.09 + n*(node(i, n) - 0.3)/0.09)

   for i in range(1, n):
      if ((node(i + 0.5, n) < 0.3) and (node(i - 0.5, n) < 0.3)):
         d[i - 1] = 0.3
      elif ((node(i + 0.5, n) > 0.3) and (node(i - 0.5, n) > 0.3)):
         d[i - 1] = 0.09
      else:
         d[i - 1] = n*0.3*(0.3 - node(i - 0.5, n)) + n*0.09*(node(i + 0.5, n) - 0.3)
   
   for i in range(1, n):
      if ((node(i + 0.5, n) < 0.3) and (node(i - 0.5, n) < 0.3)):
         fi[i - 1] = 1
      elif ((node(i + 0.5, n) > 0.3) and (node(i - 0.5, n) > 0.3)):
         fi[i - 1] = np.sin(np.pi*0.3)
      else:
         fi[i - 1] = n*(0.3 - node(i - 0.5, n)) + n*np.sin(np.pi*0.3)*(node(i + 0.5, n) - 0.3)
   return a, d, fi
def MAIN_coef(a, d, fi):
   for i in range(0, n):
      if ((node(i, n) < 0.3) and (node(i + 1, n) < 0.3)):
         a[i] = k1(node(i - 0.5, n))
      elif ((node(i, n) > 0.3) and (node(i + 1, n) > 0.3)):
         a[i] = k2(node(i - 0.5, n))
      else:
         a[i] = 1/((n*(0.3 - node(i - 1, n))/k1(0.5 * (node(i - 1, n) + 0.3))) + (node(i - 1, n) - 0.3)/k2(0.5 * (0.3 + node(i, n))))

   for i in range(1, n):
      if ((node(i + 0.5, n) < 0.3) and (node(i - 0.5, n) < 0.3)):
         d[i - 1] = q1(node(i, n))
      elif ((node(i + 0.5, n) > 0.3) and (node(i - 0.5, n) > 0.3)):
         d[i - 1] = q2(node(i, n))
      else:
         d[i - 1] = n*q1(0.5 * (node(i - 0.5, n) + 0.3)*(0.3 - node(i - 0.5, n))) + n*q2(0.5 * (0.3 + node(i + 0.5, n)))*(node(i + 0.5, n) - 0.3)
   
   for i in range(1, n):
      if ((node(i + 0.5, n) < 0.3) and (node(i - 0.5, n) < 0.3)):
         fi[i - 1] = f1(node(i, n))
      elif ((node(i + 0.5, n) > 0.3) and (node(i - 0.5, n) > 0.3)):
         fi[i - 1] = f2(node(i, n))
      else:
         fi[i - 1] = n*f1(0.5 * (node(i - 0.5, n) + 0.3))*(0.3 - node(i - 0.5, n)) + n*f2(0.5 * (0.3 + node(i + 0.5, n)))*(node(i + 0.5, n) - 0.3)
   return a, d, fi

# Матрицы
def TEST_TASK(n, ny1, ny2, next, cur, prev, f):
   a = np.empty(n, dtype=np.float16)
   d = np.empty(n - 1, dtype=np.float16)
   fi = np.empty(n - 1, dtype=np.float16)
   a, d, fi = TEST_coef(a, d, fi)
   cur[0] = d[0] + a[0]*((n + 1)**2) + a[1]*((n + 1)**2)
   next[0] = -a[1]*((n + 1)**2)
   f[0] = fi[0]
   cur[-1] = d[-1] + a[-2]*((n + 1)**2) + a[-1]*((n + 1)**2)
   prev[-1] = -a[-2]*((n + 1)**2)
   f[-1] = fi[-1]

   for i in range(0, n - 2):
      next[i] = -a[i + 1]*((n + 1)**2)
      cur[i] = d[i] + a[i]*((n + 1)**2) + a[i + 1]*((n + 1)**2)
      prev[i - 1] = -a[i]*((n + 1)**2)
      f[i] = fi[i]
   return a, d, fi, cur, next, f, cur, prev, f
def MAIN_TASK(n, ny1, ny2, next, cur, prev, f):
   a = np.empty(n, dtype=np.float16)
   d = np.empty(n - 1, dtype=np.float16)
   fi = np.empty(n - 1, dtype=np.float16)
   a, d, fi = MAIN_coef(a, d, fi)
   C[0] = d[0] + a[0]*((n + 1)**2) + a[1]*((n + 1)**2)
   B[0] = -a[1]*((n + 1)**2)
   f[0] = fi[0]
   cur[n - 1] = (d[n - 1] + a[n - 1]*((n + 1)**2) + a[n]*((n + 1)**2))
   prev[n - 2] = -a[n - 1]*((n + 1)**2)
   f[n - 1] = fi[n - 1]

   for i in range(1, n - 1):
      next[i] = -a[i + 1]*((n + 1)**2)
      cur[i] = d[i] + a[i]*((n + 1)**2) + a[i + 1]*((n + 1)**2)
      prev[i - 1] = -a[i]*((n + 1)**2)
      f[i] = fi[i]
   return a, d, fi, cur, next, f, cur, prev, f

# Решение СЛАУ
def SLAY_TEST(n, A, C, B, f, x):
   for i in range(1, n - 1):
      m = A[i - 1]/C[i - 1]
      C[i] = C[i] - m * B[i - 1]
      f[i] = f[i] - m * f[i - 1]
   x[-1] = f[-1]/C[-1]
   for i in range(n - 2, -1, -1):
      x[i - 1] = (f[i] - B[i - 1] * x[i])/C[i]
   return x
def SLAY_MAIN(n, A, C, B, f, x):
   n = n - 1
   alfa = np.empty(n, dtype=np.float16)
   beta= np.empty(n, dtype=np.float16)
   alfa[0] = -B[0]/C[0]
   beta[0] = f[0]/C[0]
   for i in range(1, n):
      alfa[i] = -B[i]/(C[i] + alfa[i - 1]*A[i - 1])
      beta[i] = (f[i] - A[i - 1]*beta[i - 1])/(C[i] + alfa[i - 1]*A[i - 1])
   x[n] = (f[n] - A[n - 1]*beta[n - 1])/(C[n] + A[n - 1]*alfa[n - 1])
   for i in range(n - 1, -1):
      x[i] = alfa[i]*x[i + 1] + beta[i]
   return x

def plot(x):
   x = np.linspace(0, 0.3, 11) 
   plt.plot(x, u1(x),'o-', linewidth = 2.0, label='траектория до точки разрыва')
   x = np.linspace(0.3, 1, 11)
   plt.plot(x, u2(x),'o-', linewidth = 2.0, label='траетория после точки разрыва')
   plt.legend()
   plt.xlabel('x')
   plt.ylabel('u(x)')
   plt.grid()
   plt.savefig('График истинной траектории.png')
   plt.show()

 """

""" 
a, d, fi, cur, next, f, cur, prev, f = TEST_TASK(n, 0, 0, next, cur, prev, f)
x1 = SLAY_TEST(n, prev, cur, next, f, x)
for i in range(1, n):
   if (node(i, n - 1) < 0.3):
      if (max < np.abs(u1(node(i, n - 1)) - x[i - 1])):
         max = np.abs(u1(node(i, n - 1)) - x[i - 1])
         maxxn = node(i, n - 1)
   else:
      if (max < np.abs(u2(node(i, n - 1)) - x[i - 1])):
         max = np.abs(u2(node(i, n - 1)) - x[i - 1])
         maxxn = node(i, n - 1)


plot(x1)
print(x1)
 """



#_________________________________________________________________________________________

#! Функции
def k1(x):
   return x**2 + 2
def k2(x):
   return x**2

def q1(x):
   return x
def q2(x):
   return x**2

def f1(x):
   return 1
def f2(x):
   return np.sin(np.pi*x)

#! Матричные коэффициенты
def coef_matrix(h, n, a, d, fi):
   A = np.empty(n - 1)
   B = np.empty(n)
   C = np.empty(n - 1)
   for i in range(len(A)):
      A[i] = a[i]/(h**2)
   for i in range(len(d)):
      B[i] = a[i + 1]/(h**2)
      C[i] = (a[i] + a[i + 1])/(h**2) + d[i]
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
   y[-1] = my[1]
   for i in range(n - 2, -1, -1):
      y[i] = alfa[i + 1]*y[i + 1] + beta[i + 1]
   return y 

#TODO------------------------------------ТЕСТОВАЯ ЗАДАЧА------------------------------------
#! Точное решение задачи
def true_solution(x_main):
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

#! Узел
def node(i, n):
   return i/n

#! Коэффициенты тестовой задачи
def test_coef(n, a, d, fi):
   for i in range(n):
      if ((node(i, n) <= 0.3) and (node(i + 1, n) <= 0.3)):
         a[i] = 2.09
      elif ((node(i, n) >= 0.3) and (node(i + 1, n) >= 0.3)):
         a[i] = 0.09
      else:
         a[i] = 1/(n*(0.3 - node(i - 1, n))/2.09 + n*(node(i, n) - 0.3)/0.09)

   for i in range(n - 1):
      if ((node(i + 0.5, n) <= 0.3) and (node(i - 0.5, n) <= 0.3)):
         d[i] = 0.3
      elif ((node(i + 0.5, n) >= 0.3) and (node(i - 0.5, n) >= 0.3)):
         d[i] = 0.09
      else:
         d[i] = n*0.3*(0.3 - node(i - 0.5, n)) + n*0.09*(node(i + 0.5, n) - 0.3)
   
   for i in range(n - 1):
      if ((node(i + 0.5, n) <= 0.3) and (node(i - 0.5, n) <= 0.3)):
         fi[i] = 1
      elif ((node(i + 0.5, n) >= 0.3) and (node(i - 0.5, n) >= 0.3)):
         fi[i] = np.sin(np.pi*0.3)
      else:
         fi[i] = n*(0.3 - node(i - 0.5, n)) + n*np.sin(np.pi*0.3)*(node(i + 0.5, n) - 0.3)
   return a, d, fi

#! Тестовая задача
def test_task(h, n):
   a = np.empty(n)
   d = np.empty(n - 1)
   fi = np.empty(n - 1)
   a, d ,fi = test_coef(n, a, d, fi)
   A, B, C = coef_matrix(h, n, a, d, fi)
   return progonka(n, A, B, C, a, d ,fi)

#TODO------------------------------------ОСНОВНАЯ ЗАДАЧА------------------------------------
#! Коэффициенты тестовой задачи
def main_coef(n, a, d, fi):
   for i in range(n):
      if ((node(i, n) <= 0.3) and (node(i + 1, n) <= 0.3)):
         a[i] = k1(node(i - 0.5, n))
      elif ((node(i, n) >= 0.3) and (node(i + 1, n) >= 0.3)):
         a[i] = k2(node(i - 0.5, n))
      else:
         a[i] = 1/((n*(0.3 - node(i - 1, n))/k1(0.5 * (node(i - 1, n) + 0.3))) + (node(i - 1, n) - 0.3)/k2(0.5 * (0.3 + node(i, n))))

   for i in range(n - 1):
      if ((node(i + 0.5, n) <= 0.3) and (node(i - 0.5, n) <= 0.3)):
         d[i] = q1(node(i, n))
      elif ((node(i + 0.5, n) >= 0.3) and (node(i - 0.5, n) >= 0.3)):
         d[i] = q2(node(i, n))
      else:
         d[i] = n*q1(0.5 * (node(i - 0.5, n) + 0.3)*(0.3 - node(i - 0.5, n))) + n*q2(0.5 * (0.3 + node(i + 0.5, n)))*(node(i + 0.5, n) - 0.3)
   
   for i in range(n - 1):
      if ((node(i + 0.5, n) <= 0.3) and (node(i - 0.5, n) <= 0.3)):
         fi[i] = f1(node(i, n))
      elif ((node(i + 0.5, n) >= 0.3) and (node(i - 0.5, n) >= 0.3)):
         fi[i] = f2(node(i, n))
      else:
         fi[i] = n*f1(0.5 * (node(i - 0.5, n) + 0.3))*(0.3 - node(i - 0.5, n)) + n*f2(0.5 * (0.3 + node(i + 0.5, n)))*(node(i + 0.5, n) - 0.3)
   return a, d, fi

#! Основная задача
def main_task(h, n):
   a = np.empty(n)
   d = np.empty(n - 1)
   fi = np.empty(n - 1)
   a, d ,fi = main_coef(n, a, d, fi)
   A, B, C = coef_matrix(h, n, a, d, fi)
   return progonka(n, A, B, C, a, d ,fi)

#! Запуск просчетов
def solution(n):
   h = 1/n
   x_main = [x_start + i*h for i in range(1, n + 1)]
   v1 = test_task(h, n)
   v1[0] = my[0]
   u = true_solution(x_main)
   u[0] = my[0]
   u[-1] = my[1]
   E = np.abs(u - v1)

   for i in range(len(x_main)):
      x_main[i] = round(x_main[i], 14)
   Node = [i for i in range(len(x_main))]
   data_test = {'№ Узла': Node, 'x': x_main, 'v': v1, 'u': u, '|u - v|': E}
   data = pd.DataFrame(data = data_test)
   data.to_csv("data_test.csv", index=False)
   
   h = h/2
   v2 = main_task(h, n)
   v2[0] = my[0]
   v2[-1] = my[1]

   e = np.abs(v1 - v2)
   Node = [i for i in range(len(x_main))]
   data_test = {'№ Узла': Node, 'x': x_main, 'v1': v1, 'v2': v2, '|v2 - v1|': e}
   data = pd.DataFrame(data = data_test)
   data.to_csv("data_main.csv", index=False)
   





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
kappa = [0, 0]
my = [1, 0]
ksi = 0.3

solution(n)

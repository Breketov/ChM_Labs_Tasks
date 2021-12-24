#                       Лабораторная работа №2
#                       Версия 0.45

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

""" 
ksi = 0.3
ny = [1, 0]
n = 100

C1 = -2.441089498977808
C2 = 0.1077561656444751
C3 = 0.8031153039014358
C4 = -0.1086898371251591

x = np.empty(n - 1, dtype=np.float16)
f = np.empty(n - 1, dtype=np.float16)
cur = np.empty(n, dtype=np.float16)
next = np.empty(n - 2, dtype=np.float16)
prev = np.empty(n - 2, dtype=np.float16)
max = 0
maxxn = 0

#* Аналитическое решение
def u1(x):
   return C1*np.exp(-np.sqrt(ksi/(ksi**2 + 2))*x) + C2*np.exp(np.sqrt(ksi/(ksi**2 + 2))*x) + 1/ksi
def u2(x):
   return C3*np.exp(-x) + C4*np.exp(x) + np.sin(np.pi*x)/((ksi**2)*(1 + np.pi**2))

# Функции
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

# Узлы
def node(i, n):
   return i/(n + 1)

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


#! Точное решение задачи
def true_solution(x_main):
   C1 = -2.441089498977808
   C2 = 0.1077561656444751
   C3 = 0.8031153039014358
   C4 = -0.1086898371251591
   u1, u2 = [], []
   x1, x2 = [], []
   for x in x_main:
      if x <= ksi:
         u1.append(C1*np.exp(-np.sqrt(ksi/(ksi**2 + 2))*x) + C2*np.exp(np.sqrt(ksi/(ksi**2 + 2))*x) + 1/ksi)
         x1.append(x)
      elif x >= ksi:
         u2.append(C3*np.exp(-x) + C4*np.exp(x) + np.sin(np.pi*x)/((ksi**2)*(1 + np.pi**2)))
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

def node(i, n):
   return i/n

def coef_matrix(a, d, fi):
   A = np.empty(n)
   B = np.empty(n - 1)
   C = np.empty(n - 1)
   for i in range(len(a)):
      A[i] = a[i]*(n**2)
   for i in range(len(d)):
      B[i] = a[i + 1]*(n**2)
      C[i] = (a[i] + a[i + 1])*(n**2) + d[i]
   return A, B, C

def test_coef(a, d, fi):
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

def test_progonka(A, B, C, a, d ,fi):
   #! Прямая прогонка
   alfa = np.empty(n - 1)
   beta = np.empty(n - 1)
   alfa[0] = kappa[0]
   beta[0] = my[0]
   for i in range(n - 2):
      alfa[i + 1] = B[i]/(C[i] - A[i]*alfa[i])
      beta[i + 1] = (fi[i] + A[i]*beta[i])/(C[i] - A[i]*alfa[i])
   
   #! Обратная прогонка
   y = np.empty(n)
   y[-1] = my[1]
   for i in range(n - 2, -1, -1):
      y[i] = alfa[i]*y[i + 1] + beta[i]
   return y 

def test_task():
   a = np.empty(n)
   d = np.empty(n - 1)
   fi = np.empty(n - 1)
   a, d ,fi = test_coef(a, d, fi)
   A, B, C = coef_matrix(a, d, fi)
   return test_progonka(A, B, C, a, d ,fi)

def solution():
   h = 1/n
   x_main = [x_start + i*h for i in range(n + 1)]
   x_aux = [x_start + (i + 0.5)*h for i in range(n - 1)]
   v = test_task()
   v[0] = my[0]
   u = true_solution(x_main)
   E = np.abs(u - v)

   Node = [i for i in range(len(x_main))]
   data_test = {'№ Узла': Node, 'x': x_main, 'v': v, 'u': u, '|u - v|': E}
   data = pd.DataFrame(data = data_test)
   data.to_csv("data_test.csv", index=False)
   


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

solution()

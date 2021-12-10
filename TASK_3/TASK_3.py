import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def task(x, u):
   du1dx = -500.005*u[0] + 499.995*u[1]
   du2dx = 499.995*u[0] - 500.005*u[1]
   return np.array([du1dx, du2dx])

def true_task(x, v):
   u = []
   for i in range(0, len(x)):
      u.append(list(3*np.exp(-1000*x[i])*np.array([-1, 1]) + 10*np.exp(-0.01*x[i])*np.array([1, 1])))
   u = np.array(u)
   E = []
   for i in range(0, len(u)):
      e = ((u[i] - v[i]))
      E.append(max(abs(e[0]), abs(e[1])))
   return u, E

def record(x1, v1, v2, h, h0, n, olp_, S_, olp, S, x_1, v_1, v_2, i):
   olp.append(olp_)
   S.append(S_)
   h.append(h0)
   x_1.append(x1)
   v_1.append(v1)
   v_2.append(v2)
   n.append(i)
   return h, n, olp, S, x_1, v_1, v_2

def values(x1, v1, v2, x_1, v_1, v_2, h, h0, n, olp, S, C1, C2, i):
   S_ = (v2 - v1)
   S_ = max(abs(S_[0]), abs(S_[1]))
   olp_ = S_*2
   if (contr_loc_ == 1):
      if((epsilon_min <= abs(S_)) and (abs(S_)<= epsilon)):
         h, n, olp, S, x_1, v_1, v_2 = record(x1, v1, v2, h, h0, n, olp_, S_, olp, S, x_1, v_1, v_2, i)
         h0 = h0
         i = i + 1
      elif(abs(S_) < (epsilon_min)):
         h, n, olp, S, x_1, v_1, v_2 = record(x1, v1, v2, h, h0, n, olp_, S_, olp, S, x_1, v_1, v_2, i)
         C1 = C1 + 1
         h0 = 2*h0
         i = i + 1
      else:
         x1 = x_1[-1]
         v1 = v_1[-1]
         v2 = v_2[-1] 
         C2 = C2 + 1
         h0 = h0/2
   else:
      h, n, olp, S, x_1, v_1, v_2 = record(x1, v1, v2, h, h0, n, olp_, S_, olp, S, x_1, v_1, v_2, i)
      C1 = 0
      C2 = 0
      i = i + 1
   return x1, v1, x_1, v_1, v_2, h, h0, n, olp, S, C1, C2, i

def func_border(x1):
   if ((border - epsilon_gr <= x1) and (x1 <= border)):
      return 0
   elif (x1 >= border):
      return 1

def plot():
   x = np.linspace(0, 1000, 51)
   u = true_task(x)
   plt.plot(x, u[:, 0], 'o-', label='u1(t) - Истинная траектория')
   plt.plot(x, u[:, 1], 'x-', label='u2(t) - Истинная траектория')
   plt.legend()
   plt.xlabel('t')
   plt.ylabel('u1(t)   u2(t)')
   plt.grid()
   plt.show()

   x = np.linspace(0, 0.01, 51)
   u = true_task(x)
   plt.plot(x, u[:, 0], 'o-', label='u1(t) - Истинная траектория')
   plt.plot(x, u[:, 1], 'x-', label='u2(t) - Истинная траектория')
   plt.axis([0, 0.01, 6, 14])
   plt.legend()
   plt.xlabel('t')
   plt.ylabel('u1(t)   u2(t)')
   plt.grid()
   plt.show()

   x = np.linspace(0, 0.5, 51)
   u = true_task(x)
   plt.plot(x, u[:, 0], 'o-', label='u1(t) - Истинная траектория')
   plt.plot(x, u[:, 1], 'x-', label='u2(t) - Истинная траектория')
   plt.axis([0, 0.5, 6, 14])
   plt.legend()
   plt.xlabel('t')
   plt.ylabel('u1(t)   u2(t)')
   plt.grid()
   plt.show()

def Euler(x0, v0, h0, Nmax):
   i = 1
   x1 = x0
   x_1 = [x1]
   v1 = np.array(v0)
   v2 = np.array(v0)
   n, h, S, olp = [0], [0], [0], [0]
   v_1, v_2 = [v1], [v2]
   C1, C2 = 0, 0
   E = np.array([[1, 0], [0, 1]])
   A = np.array([[-500.005, 499.995], [499.995, -500.005]])
   while i < Nmax:
      B = np.linalg.inv(E - h0*A)
      v1 = B @ v1
      x1 = x1 + h0
      i = i + 1
      h0 = h0/2
      v2 = v_1[-1]
      for j in range(0, 2):
         B = np.linalg.inv(E - h0*A)
         v2 = B @ v2
      h0 = h0*2
      x1, v1, x_1, v_1, v_2, h, h0, n, olp, S, C1, C2, i = values(x1, v1, v2, x_1, v_1, v_2, h, h0, n, olp, S, C1, C2, i)
      z = func_border(x1)
      if (z == 0):
         break
      elif (z == 1):
         x_1.pop(-1)
         v_1.pop(-1)
         v_2.pop(-1)
         S.pop(-1)
         olp.pop(-1)
         h.pop(-1)
         n.pop(-1)
         x1 = x_1[-1]
         v1 = v_1[-1]
         v2 = v_2[-1]
         C2 = C2 + 1
         i = i - 1
         h0 = h0/2
   return n, h, x_1, np.array(v_1), np.array(v_2), S, olp, C1, C2

#* Терминал
print('_______________________________________________________________________________________________')
print('Бекетов Евгений | Команда Эльвина | Программа №3 "Жесткая задача"')
print('Реализуется неявный метод Эйлера')
print('_______________________________________________________________________________________________')
print('Исходные начальные данные: x0 = 0, u0 = (7, 13)')
print('Введите необходимые данные')
print('Задайте границу по времени:')
border = float(input('X = '))
print('Задайте точность выхода на границу:')
epsilon_gr = float(input('epsilon = '))
print ('С контролем локальной погрешности?  Введите Да или Нет')
contr_loc = input()
if ((contr_loc == 'да') or (contr_loc == 'Да')):
   print('Задайте контроль локальной погрешности:')
   epsilon = float(input('epsilon = '))
   epsilon_min = epsilon/4
   contr_loc_ = 1
elif ((contr_loc == 'нет') or (contr_loc == 'Нет')):
   contr_loc_ = 0
print('Задайте максимальное число шагов:')
Nmax = float(input('Nmax = '))
print('Задайте начальный шаг:')
h0 = float(input('h = '))

u0 = [7, 13]
x0 = 0

#* Результат работы РК4 и функционала
#n, h, x1, v1, v2, S, olp, C1, C2
n, h, x1, v1, v2, S, olp, C1, C2 = Euler(x0, u0, h0, Nmax)

for i in range(0, len(x1)):
   x1[i] = round(x1[i], 11)
abs_olp_ = []
for i in range(0, len(olp)):
   abs_olp = abs(olp[i])
   abs_olp_.append(abs_olp)
max_S = max(abs_olp_[1:])
k_max_S = abs_olp_.index(max_S)
x_max_S = x1[k_max_S]
min_S = min(abs_olp_[1:])
k_min_S = abs_olp_.index(min_S)
x_min_S = x1[k_min_S]

max_h = max(h[1:])
k_max_h = h.index(max_h)
x_max_h = x1[k_max_h]
min_h = min(h[1:])
k_min_h = h.index(min_h)
x_min_h = x1[k_min_h]

#* Справка
print('_______________________________________Справка_________________________________________________')
print('Тип метода: Неявный метод Эйлера')
print('Неявный метод Эйлера:    способ счета одношаговый')
print('Время начала счета = ', x0, '  ', 'Начальный вектор = ', '(',u0,')')                               
print('Точка выхода (условие остановки счета) = ', border)
print('epsilon граничный  = ', epsilon_gr)                               
print('Начальный шаг h0 = ', h0,'  ', 'Nmax = ', Nmax)
if (contr_loc_ == 1):
   print('Контроль модуля локальной погрешности включен')
   print('Контроль модуля локальной погрешности сверху = ', epsilon)    
   print('Контроль модуля локальной погрешности снизу = ', epsilon_min)
elif (contr_loc_ == 0):
   print('Контроль модуля локальной погрешности отключен')
print(' ')
print('Результаты расчета:')
print('Число шагов  = ', len(n) - 1)
print('Выход к границе заданной точности  = ', border - x1[-1])
print('Текущее время = ', x1[-1], '  ', 'Текущая вектор = ', v1[-1])
print('Максимальная контрольная величина: ', max_S/2, '  ', 'при x = ', x_max_S)
print('Минимальная контрольная величина: ', min_S/2, '  ', 'при x = ', x_min_S)
if (contr_loc_ == 1):
   print('Число увеличений шага: ', C1)
   print('Число уменьшений шага: ', C2)
   print('Максимальный шаг: ', max_h, '  ', 'при x = ', x_max_h)
   print('Минимальный шаг: ', min_h, '  ', 'при x = ', x_min_h)
elif (contr_loc_ == 0):
   print('Увеличения или уменьшения шага не происходило')
   print('Шаг не менялся h = ', max_h)
print('_______________________________________________________________________________________________')

u, E = true_task(x1, v1)

#* Вывод
table = {'n': n, 'h': h, 'x': x1, 'v1[1]': v1[:, 0], 'v1[2]': v1[:, 1], 'v2[1]': v2[:, 0], 'v2[2]': v2[:, 1], 'S': S, 'ОЛП': abs_olp_, 'u[1]': u[:, 0], 'u[2]': u[:, 1], 'E': E}
data = pd.DataFrame(data = table)
data.to_csv("Таблица_Жесткая_Зачада.csv", index=False)

""" plot() """
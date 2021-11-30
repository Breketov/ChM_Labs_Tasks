import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#* Всякие задачи
def test_true(x, u0):
   x = np.array(x)
   u = u0 * np.exp(-3/2 * x)
   return x, u

def tasks(x, u):
   if (zadacha == 0):
      return -1* 3/2 * u
   elif (zadacha == 1):
      return (u**2)/((1+(x**2))**(1/3)) + u - (u**3)*math.sin(10*x)
   elif (zadacha == 2):
      du = u[1]
      dx = -a* (u[1]**2) - b * u[0]
      return np.array([du, dx])

#* Тут рисуются граифики 
def plot():
   if (zadacha == 0):
      plt.plot(x1, v1, 'x-', label = 'Численная траектория')
      plt.plot(x1, u, 'o--', label = 'Истинная траектория')
      plt.xlabel('x')
      plt.ylabel('u(x)    v(x)')
      plt.legend()
      plt.grid()
      plt.savefig('График_Тестовая.png', bbox_inches='tight')
      plt.show()
   elif (zadacha == 1):
      plt.plot(x1, v1, 'o-', label = 'Численная траектория')
      plt.xlabel('x')
      plt.ylabel('v(x)')
      plt.legend()
      plt.grid()
      plt.savefig('График_Основная_1.png', bbox_inches='tight')
      plt.show()
   elif (zadacha == 2):
      plt.plot(x1, v1[:, 0], 'o-', label = 'Численная траектория по v[1]')
      plt.plot(x1, v1[:, 1], 'x-', label = 'Численная траектория по v[2]')
      plt.xlabel('x')
      plt.ylabel('v(x)')
      plt.legend()
      plt.grid()
      plt.savefig('График_Основная_2.png', bbox_inches='tight')
      plt.show()

      plt.plot(v1[:, 0], v1[:, 1], 'o-', label = 'Фазовый портрет')
      plt.xlabel('v[1]')
      plt.ylabel('v[2]')
      plt.legend()
      plt.grid()
      plt.savefig('График_Фазовый_Основная_2.png', bbox_inches='tight')
      plt.show()
 
#* Всякий функционал
def record(x1, v1, h, h0, n, olp_, S_, olp, S, x_1, v_1, i):
   olp.append(olp_)
   S.append(S_)
   h.append(h0)
   x_1.append(x1)
   v_1.append(v1)
   n.append(i)
   return h, n, olp, S, x_1, v_1

def znach(x1, v1, x2, v2, h, h0, n, olp, S, C1, C2, x_1, v_1, x_2, v_2, i):
   S_ = (v2 - v1)/15
   if ((zadacha == 0) or (zadacha == 1)):
      olp_ = S_*16
   elif (zadacha == 2):
      S_ = math.sqrt((S_[0]**2) + (S_[1]**2))
      olp_ = S_*16
   if (contr_loc_ == 1):
      if((epsilon/32) <= abs(S_) <= epsilon):
         h, n, olp, S, x_1, v_1 = record(x1, v1, h, h0, n, olp_, S_, olp, S, x_1, v_1, i)
         h0 = h0
         i = i + 1
      elif(abs(S_) <= (epsilon/32)):
         h, n, olp, S, x_1, v_1 = record(x1, v1, h, h0, n, olp_, S_, olp, S, x_1, v_1, i)
         C1 = C1 + 1
         h0 = 2*h0
         i = i + 1
      else:
         x1 = x_1[-1]
         v1 = v_1[-1]
         x_2.pop(-1)
         v_2.pop(-1)
         x2 = x_2[-1]
         v2 = v_2[-1]
         C2 = C2 + 1
         h0 = h0/2
   else:
      h, n, olp, S, x_1, v_1 = record(x1, v1, h, h0, n, olp_, S_norm, olp, S, x_1, v_1, i)
      C1 = 0
      C2 = 0
      i = i + 1
   return x1, v1, x2, v2, h, h0, n, olp, S, C1, C2, x_1, v_1, x_2, v_2, i

def func_border(x1):
   if ((border - epsilon_gr <= x1) and (x1 <= border)):
      return 0
   elif (x1 >= border):
      return 1

#* Рунге-Кутта
def RK4(x0, v0, h0, Nmax):
   i = 1
   x1, x2 = x0, x0
   x_1, x_2 = [x1], [x2]
   C1, C2 = 0, 0
   h, n, olp, S = [0], [0], [0], [0]
   if (zadacha == 0 or zadacha == 1):
      v1, v2 = v0, v0
   elif (zadacha == 2):
      v1, v2 = np.array(v0), np.array(v0)
   v_1, v_2 = [v1], [v2]
   while i < Nmax + 1:
      k1_1 = tasks(x1, v1)
      k2_1 = tasks(x1 + h0/2, v1 + 0.5 * h0 * k1_1)
      k3_1 = tasks(x1 + h0/2, v1 + 0.5 * h0 * k2_1)
      k4_1 = tasks(x1 + h0, v1 + h0 * k3_1)
      x1 = x1 + h0
      v1 = v1 + h0/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
      j = i - 1
      while j < i + 1:
         k1_2 = tasks(x2, v2)
         k2_2 = tasks(x2 + h0/4, v2 + 0.25 * h0 * k1_2)
         k3_2 = tasks(x2 + h0/4, v2 + 0.25 * h0 * k2_2)
         k4_2 = tasks(x2 + h0/2, v2 + 0.5 * h0 * k3_2)
         x2 = x2 + h0/2
         v2 = v2 + h0/12 * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
         j = j + 1
      x_2.append(x2)
      v_2.append(v2)
      x1, v1, x2, v2, h, h0, n, olp, S, C1, C2, x_1, v_1, x_2, v_2, i = znach(x1, v1, x2, v2, h, h0, n, olp, S, C1, C2, x_1, v_1, x_2, v_2, i)
      z = func_border(x1)
      if (z == 0):
         break
      elif (z == 1):
         olp.pop(-1)
         h.pop(-1)
         x_1.pop(-1)
         x_2.pop(-1)
         x1 = x_1[-1]
         x2 = x_2[-1]
         v_1.pop(-1)
         v_2.pop(-1)
         v1 = v_1[-1]
         v2 = v_2[-1]
         n.pop(-1)
         S.pop(-1)
         C2 = C2 + 1
         h0 = h0/4
   return n, h, x_1, v_1, v_2, S, olp, C1, C2

#* Терминал
print('_____________________________________________________________________________________________________________________')
print('Команда Эльвина | Лабораторная работа №1 | Численное решение задачи Коши для ОДУ')
print('_____________________________________________________________________________________________________________________')
print('Выберите тип задачи:')
print('Тестовая - 0')
print('Основная №1 - 1')
print('Основная №2 - 2')
zadacha = int(input('Задача: '))
if (zadacha == 0):
   print('Введите начальные условия для Тестовой задачи:')
   x0 = float(input('x0 = '))
   u0 = float(input('u0 = '))
elif (zadacha == 1):
   print('Введите начальные условия для Основной №1 задачи:')
   x0 = float(input('x0 = '))
   u0 = float(input('u0 = '))
elif (zadacha == 2):
   print('Введите начальные условия для Основной №2 задачи:')
   x0 = float(input('x0 = '))
   u0_1 = float(input('u0 = '))
   u0_2 = float(input("u'0 = "))
   u0 = [u0_1, u0_2]
   print('Параметры a и b;')
   a = float(input('a = '))
   b = float(input('b = '))
print('Задайте правую границу по x:')
border = float(input('X = '))
print('Задайте точность выхода на границу:')
epsilon_gr = float(input('epsilon = '))
print('Задайте максимальное число шагов:')
Nmax = int(input('Nmax = '))
print('Задайте начальный шаг:')
h0 = float(input('h = '))
print ('С контролем локальной погрешности?  Введите Да или Нет')
contr_loc_str = input()
if ((contr_loc_str == 'да') or (contr_loc_str == 'Да')):
   print('Задайте контроль локальной погрешности:')
   epsilon = float(input('epsilon = '))
   contr_loc_ = 1
elif ((contr_loc_str == 'нет') or (contr_loc_str == 'Нет')):
   contr_loc_ = 0

#* Результат работы РК4 и функционала
#n, h, x_1, v_1, v_2, S, olp, C1, C2
n, h, x1, v1, v2, S, olp, C1, C2 = RK4(x0, u0, h0, Nmax)

#* Тут всякое для справки
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

v1 = np.array(v1)
v2 = np.array(v2)
S = np.array(S)

max_h = max(h[1:])
k_max_h = h.index(max_h)
x_max_h = x1[k_max_h]
min_h = min(h[1:])
k_min_h = h.index(min_h)
x_min_h = x1[k_min_h]


print('_______________________________________________________Справка_______________________________________________________')
if (zadacha == 0):
   print('Тип задачи: Тестовая')
elif (zadacha == 1):
   print('Тип задачи: Основная №1')
elif (zadacha == 2):
   print('Тип задачи: Основная №2')
print('Метод Рунге Кутта порядка p = 4')
print('Начало счета = ', x0, '  ', 'Начальное значение = ', u0)
print('Точка выхода (условие остановки счета) = ', border)
print('epsilon граничный  = ', epsilon_gr)
print('Начальный шаг h0 = ', h[0],'  ', 'Nmax = ', Nmax)
if (contr_loc_ == 1):
   print('Контроль локальной погрешности включен')
   print('Контроль локальной погрешности = ', epsilon)    
elif (contr_loc_ == 0):
   print('Контроль локальной погрешности отключен')
print(' ')
print('Результаты расчета:')
print('Число шагов  = ', len(n) - 1)
print('Выход к границе заданной точности  = ', border - x1[-1])
print('Текущий x = ', x1[-1], '  ', 'Текущий v = ', v1[-1])
print('Максимальная контрольная величина: ', max_S/16, '  ', 'при x = ', x_max_S)
print('Минимальная контрольная величина: ', min_S/16, '  ', 'при x = ', x_min_S)
print('Число увеличений шага: ', C1)
print('Число уменьшений шага: ', C2)
print('Максимальный шаг: ', max_h, '  ', 'при x = ', x_max_h)
print('Минимальный шаг: ', min_h, '  ', 'при x = ', x_min_h)
print('_____________________________________________________________________________________________________________________')

#* Вывод данных
if (zadacha == 0):
   x, u = test_true(x1, u0)
   v1 = np.array(v1)
   e = u - v1
   e = np.abs(e)
   table = {'n': n, 'h': h, 'x': x1, 'v_1': v1, 'v_2': v2, 'S': S, 'ОЛП': olp, 'u': u, '|u-v|': e}
   data = pd.DataFrame(data = table)
   data.to_csv("table_lab_1.csv", index=False)
elif (zadacha == 1):
   table = {'n': n, 'h': h, 'x': x1, 'v_1': v1, 'v_2': v2, 'S': S, 'ОЛП': olp}
   data = pd.DataFrame(data = table)
   data.to_csv("table_lab_1.csv", index=False)
elif (zadacha == 2):
   table = {'n': n, 'h': h, 'x': x1, 'v[1]_1': v1[:, 0], 'v[2]_1': v1[:, 1], 'v[1]_2': v2[:, 0], 'v[2]_2': v2[:, 1], 'S': S, 'ОЛП': abs_olp_}
   data = pd.DataFrame(data = table)
   data.to_csv("table_lab_1.csv", index=False)
plot()
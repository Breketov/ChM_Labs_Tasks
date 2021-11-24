from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def task(x, u):
   dudx = u[1]
   duxy = (-c*u[1] - k* u[0])/m
   return np.array([dudx, duxy])

def plot():
   plt.plot(x1, v1[:, 0], 'o-', label = 'Численная траектория по v[1]')
   plt.plot(x1, v1[:, 1], 'x-', label = 'Численная траектория по v[2]')
   plt.xlabel('x')
   plt.ylabel('v[1](x)    v[2](x)')
   plt.legend()
   plt.grid()
   plt.savefig('График_Задача_11.png', bbox_inches='tight')
   plt.show()

   plt.plot(v1[:, 0], v1[:, 1], 'o-', linewidth = 2.0, label = 'Фазовый портрет')
   plt.xlabel('Отклонение')
   plt.ylabel('Скорость')
   plt.legend()
   plt.grid()
   plt.show()
   plt.savefig('График_Фазовый_Задача_11.png', bbox_inches='tight')

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
   S_norm = sqrt((S_[0]**2) + (S_[1]**2))
   olp_ = S_norm*16
   if (contr_loc_ == 1):
      if((epsilon/32) <= abs(S_norm) <= epsilon):
         h, n, olp, S, x_1, v_1 = record(x1, v1, h, h0, n, olp_, S_, olp, S, x_1, v_1, i)
         h0 = h0
         i = i + 1
      elif(abs(S_norm) <= (epsilon/32)):
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
      h, n, olp, S, x_1, v_1 = record(x1, v1, h, h0, n, olp_, S_, olp, S, x_1, v_1, i)
      C1 = 0
      C2 = 0
      i = i + 1
   return x1, v1, x2, v2, h, h0, n, olp, S, C1, C2, x_1, v_1, x_2, v_2, i

def func_border(x1):
   if ((border - epsilon_gr <= x1) and (x1 <= border)):
      return 0
   elif (x1 >= border):
      return 1
   else: 
      return 2

#* Рунге-Кутта
def RK4(x0, v0, h0, Nmax):
   i = 1
   x1, x2 = x0, x0
   x_1, x_2 = [x0], [x0]
   v1, v2 = np.array(v0), np.array(v0)
   v_1, v_2 = [v1], [v2]
   olp, S = [0], [0]
   h, n = [0], [0]
   C1, C2 = 0, 0
   while i < Nmax + 1:
      k1_1 = task(x1, v1)
      k2_1 = task(x1 + h0/2, v1 + 0.5 * h0 * k1_1)
      k3_1 = task(x1 + h0/2, v1 + 0.5 * h0 * k2_1)
      k4_1 = task(x1 + h0, v1 + h0 * k3_1)
      x1 = x1 + h0
      v1 = v1 + h0/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
      j = i - 1
      while j < i + 1:
         k1_2 = task(x2, v2)
         k2_2 = task(x2 + h0/4, v2 + 0.25 * h0 * k1_2)
         k3_2 = task(x2 + h0/4, v2 + 0.25 * h0 * k2_2)
         k4_2 = task(x2 + h0/2, v2 + 0.5 * h0 * k3_2)
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
         v_1.pop(-1)
         v_2.pop(-1)
         n.pop(-1)
         break
   return n, h, x_1, v_1, v_2, olp, C1, C2

#* Терминал
print('_______________________________________________________________________________________________')
print('Джафарзаде Эльвин | Команда Эльвина | Задача №11')
print('Вариант 4')
print('Груз массы m может совершать прямолинейные перемещения вдоль оси абсцисс по горизонтальной \nплоскости без трения. Для стабилизации положения груза используется аналогичная система \nно пружина с нелинейной характеристикой отсутствует.')
print('Положение груза в системе описывает линейное дифференциальное уравнение:')
print("        m * u'' + c * u' + k * u = 0 \n        u(x0) = u0; \n        u'(x0) = u'0")
print('Здесь к - жесткость пружины, с - коэффициент демпфирования, m - масса груза, \nu(x) - отклонение груза от состояния равновесия в момент времени x, u0 - отклонение тела \nв начальный момент времени.')
print('Задача находит численную зависимость отклонения груза от времени.')
print('_______________________________________________________________________________________________')
print('Введите начальные условия задачи:')
x0 = float(input('Начальное время x0 = '))
u0 = float(input('Начальное отклонение груза u0 = '))
dudx = float(input("Начальная скорость груза u'0 = "))
u0_2 = [u0, dudx]
print('Введите параметры задачи:')
m = float(input('Масса груза m = '))
c = float(input('Коэффициент демпфирования c = '))
k = float(input('Постоянная жесткость пружины k = '))
print('Задайте условие остановки счета: ')
border = float(input('Последний x = '))
print('Задайте точность выхода по времени:')
epsilon_gr = float(input('epsilon = '))
print ('С контролем локальной погрешности?  Введите Да или Нет')
contr_loc = input()
if ((contr_loc == 'да') or (contr_loc == 'Да')):
   print('Задайте контроль локальной погрешности:')
   epsilon = float(input('epsilon = '))
   epsilon_min = epsilon/32
   contr_loc_ = 1
elif ((contr_loc == 'нет') or (contr_loc == 'Нет')):
   contr_loc_ = 0
print('Задайте максимальное число шагов:')
Nmax = float(input('Nmax = '))
print('Задайте начальный шаг:')
h0 = float(input('h = '))

# n, h, x_1, v_1, v_2, olp, C1, C2
n, h, x1, v1, v2, olp, C1, C2 = RK4(x0, u0_2, h0, Nmax)

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

max_h = max(h[1:])
k_max_h = h.index(max_h)
x_max_h = x1[k_max_h]
min_h = min(h[1:])
k_min_h = h.index(min_h)
x_min_h = x1[k_min_h]

#* Справка
print('_______________________________________Справка_________________________________________________')
print('№ Варианта задания: 4')
print('Тип задачи: Основная')
print('Метод Рунге Кутта порядка p = 4   способ счета одношаговый')
print('Время начала счета = ', x0, '  ', 'Начальная отклонение груза = ', u0, 'см')
print('Точка выхода (условие остановки счета) = ', border)
print('epsilon граничный  = ', epsilon_gr)
print('Начальный шаг h0 = ', h[1],'  ', 'Nmax = ', Nmax)
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
print('Текущее время = ', x1[-1], '  ', 'Текущее отклонение = ', v1[-1],)
print('Максимальная контрольная величина по длинне: ', max_S/16, '  ', 'при x = ', x_max_S)
print('Минимальная контрольная величина по длинне: ', min_S/16, '  ', 'при x = ', x_min_S)
if (contr_loc_ == 1):
   print('Число увеличений шага: ', C1)
   print('Число уменьшений шага: ', C2)
   print('Максимальный шаг: ', max_h, '  ', 'при x = ', x_max_h)
   print('Минимальный шаг: ', min_h, '  ', 'при x = ', x_min_h)
elif (contr_loc_ == 0):
   print('Увеличения или уменьшения шага не происходило')
   print('Шаг не менялся h = ', max_h)
print('_______________________________________________________________________________________________')

#* Вывод
table = {'n': n, 'h': h, 'x': x1, 'v[1]_1': v1[:, 0], 'v[2]_1': v1[:, 1], 'v[1]_2': v2[:, 0], 'v[2]_2': v2[:, 1], 'ОЛП': abs_olp_}
data = pd.DataFrame(data = table)
data.to_csv("Таблица_Зачада_11.csv", index=False)
plot()
import math
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

#____________________________________________________________________________________________________________________________
#* Здесь решение тестовой задачи
def test_true(x0, u0, x_last):
   x = x0
   u = u0
   i  = 0
   u_ = []
   while i < len(x_last):
      u = u0 * math.exp(-3/2 * x_last[i])
      u_.append(u)
      i = i + 1
   return x_last, u_ 

#____________________________________________________________________________________________________________________________
#* Здесь описаны задачи
def test(x, u):
   dudx = -1* 3/2 * u
   return dudx

def O1(x, u):
   dudx = 1/((1+(x**2))**(1/3))*(u**2) + u - (u**3)*math.sin(10*x)
   return dudx

def O2(x, u):
   du_ = u[1]
   dx_= -a* (u[1]**2) - b * u[0]
   return du_, dx_ 

#____________________________________________________________________________________________________________________________
#* Здесь описаны РК4 для каждой задачи
def RK4_test_O1(x0, v0, h, Nmax):
   i = 1
   x1 = x0
   v1 = v0
   x2 = x0
   v2 = v0

   x_1 = [x0]
   v_1 = [v0]
   x_2 = [x0]
   v_2 = [v0]

   h_ = [0]
   n_ = [0]
   olp_ = [0]
   S_ = [0]
   C1 = 0
   C2 = 0
   
   if (zadacha == 0):
      while i < Nmax + 1:
         k1_1 = test(x1, v1)
         k2_1 = test(x1 + h/2, v1 + 0.5 * h * k1_1)
         k3_1 = test(x1 + h/2, v1 + 0.5 * h * k2_1)
         k4_1 = test(x1 + h, v1 + h * k3_1)
         x1 = x1 + h
         v1 = v1 + h/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
         j = i - 1
         while j < i + 1:
            k1_2 = test(x2, v2)
            k2_2 = test(x2 + h/4, v2 + 0.25 * h * k1_2)
            k3_2 = test(x2 + h/4, v2 + 0.25 * h * k2_2)
            k4_2 = test(x2 + h/2, v2 + 0.5 * h * k3_2)
            x2 = x2 + h/2
            v2 = v2 + h/12 * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
            j = j + 1
         x_2.append(x2)
         v_2.append(v2)
         S = (v2 - v1)/15
         olp = S*16
         if (contr_loc_ == 1):
            if((epsilon/32) <= abs(S) <= epsilon):
               olp_.append(olp)
               S_.append(S)
               h_.append(h)
               x_1.append(x1)
               v_1.append(v1)
               n_.append(i)
               h = h
               i = i + 1
            elif(abs(S) <= (epsilon/32)):
               olp_.append(olp)
               S_.append(S)
               h_.append(h)
               x_1.append(x1)
               v_1.append(v1)
               n_.append(i)
               C1 = C1 + 1
               h = 2*h
               i = i + 1
            else:
               x_2.pop(-1)
               v_2.pop(-1)
               x2 = x_2[-1]
               v2 = v_2[-1]
               C2 = C2 + 1
               h = h/2
               i = i + 1
         else:
            olp_.append(olp)
            S_.append(S)
            x_1.append(x1)
            v_1.append(v1)
            h_.append(h)
            n_.append(i)
            C1 = 0
            C2 = 0
            i = i + 1
         if((granitsa - epsilon_gr <= v1) and (v1 <= granitsa)):
            break
   elif (zadacha == 1):
      while i < Nmax + 1:
         k1_1 = O1(x1, v1)
         k2_1 = O1(x1 + h/2, v1 + 0.5 * h * k1_1)
         k3_1 = O1(x1 + h/2, v1 + 0.5 * h * k2_1)
         k4_1 = O1(x1 + h, v1 + h * k3_1)
         x1 = x1 + h
         v1 = v1 + h/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
         j = i - 1
         while j < i + 1:
            k1_2 = O1(x2, v2)
            k2_2 = O1(x2 + h/4, v2 + 0.25 * h * k1_2)
            k3_2 = O1(x2 + h/4, v2 + 0.25 * h * k2_2)
            k4_2 = O1(x2 + h/2, v2 + 0.5 * h * k3_2)
            x2 = x2 + h/2
            v2 = v2 + h/12 * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
            j = j + 1
         x_2.append(x2)
         v_2.append(v2)
         S = (v2 - v1)/15
         olp = S*16
         if (contr_loc_ == 1):
            if((epsilon/32) <= abs(S) <= epsilon):
               olp_.append(olp)
               S_.append(S)
               h_.append(h)
               x_1.append(x1)
               v_1.append(v1)
               n_.append(i)
               h = h
               i = i + 1
            elif(abs(S) <= (epsilon/32)):
               olp_.append(olp)
               S_.append(S)
               h_.append(h)
               x_1.append(x1)
               v_1.append(v1)
               n_.append(i)
               C1 = C1 + 1
               h = 2*h
               i = i + 1
            else:
               x_2.pop(-1)
               v_2.pop(-1)
               x2 = x_2[-1]
               v2 = v_2[-1]
               C2 = C2 + 1
               h = h/2
               i = i + 1
         else:
            olp_.append(olp)
            S_.append(S)
            x_1.append(x1)
            v_1.append(v1)
            h_.append(h)
            n_.append(i)
            C1 = 0
            C2 = 0
            i = i + 1
         if((granitsa - epsilon_gr <= v1) and (v1 <= granitsa)):
            break
   return n_, h_, x_1, v_1, v_2, olp_, C1, C2, S_

def RK4_O2(x0, v0, h, Nmax):
   i = 1
   x1 = x0
   v1 = v0
   x2 = x0
   v2 = v0

   h_ = [h]
   n_ = [0]
   S1 = [0]
   S2 = [0]
   olp1 = [0]
   olp2 = [0]
   C1 = 0
   C2 = 0

   x_1 = [x1]
   v1_ = np.array(v1)
   v1_1 = [v1[0]]
   v2_1 = [v1[1]]

   x_2 = [x2]
   v2_ = np.array(v2)
   v1_2 = [v2[0]]
   v2_2 = [v2[1]]

   while i < Nmax + 1:
      k1_1 = O2(x1, v1_)
      k1_1 = np.array(k1_1)
      k2_1 = O2(x1 + h/2, v1_ + 0.5 * h * k1_1)
      k2_1 = np.array(k2_1)
      k3_1 = O2(x1 + h/2, v1_ + 0.5 * h * k2_1)
      k3_1 = np.array(k3_1)
      k4_1 = O2(x1 + h, v1_ + h * k3_1)
      k4_1 = np.array(k4_1)
      x1 = x1 + h
      v1 = v1_ + h/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
      j = i - 1
      while j < i + 1:
         k1_2 = O2(x2, v2_)
         k1_2 = np.array(k1_2)
         k2_2 = O2(x2 + h/4, v2_ + 0.25 * h * k1_2)
         k2_2 = np.array(k2_2)
         k3_2 = O2(x2 + h/4, v2_ + 0.25 * h * k2_2)
         k3_2 = np.array(k3_2)
         k4_2 = O2(x2 + h/2, v2_ + 0.5 * h * k3_2)
         k4_2 = np.array(k4_2)
         x2 = x2 + h/2
         v2 = v2_ + h/12 * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
         j = j + 1
      x_2.append(x2)
      v1_2.append(v2[0])
      v2_2.append(v2[1])

      S_1 = (v2[0] - v1[0])/15
      S_2 = (v2[1] - v1[1])/15
      olp_1 = S_1*16
      olp_2 = S_2*16
      if (contr_loc_ == 1):
         if(((epsilon/16) <= abs(S_1) and abs(S_1) <= epsilon) and ((epsilon/16) <= abs(S_2) and abs(S_2) <= epsilon)):
            olp1.append(olp_1)
            olp2.append(olp_2)
            S1.append(S_1)
            S2.append(S_2)
            x_1.append(x1)
            v1_1.append(v1[0])
            v2_1.append(v1[1])
            h_.append(h)
            n_.append(i)
            h = h
            i = i + 1
         elif((abs(S_1) <= (epsilon/16)) and (abs(S_2) <= (epsilon/16))):
            olp1.append(olp_1)
            olp2.append(olp_2)
            S1.append(S_1)
            S2.append(S_2)
            x_1.append(x1)
            v1_1.append(v1[0])
            v2_1.append(v1[1])
            h_.append(h)
            n_.append(i)
            C1 = C1 + 1
            h = 2*h
            i = i + 1
         else:
            x_2.pop(-1)
            v1_2.pop(-1)
            v2_2.pop(-1)
            x2 = x_2[-1]
            v2 = [v1_2[-1], v2_2[-1]]
            C2 = C2 + 1
            h = h/2
            i = i + 1
      else:
         olp1.append(olp_1)
         olp2.append(olp_2)
         S1.append(S_1)
         S2.append(S_2)
         x_1.append(x1)
         v1_1.append(v1[0])
         v2_1.append(v1[1])
         h_.append(h)
         n_.append(i)
         C1 = 0
         C2 = 0
         i = i + 1
      if((granitsa - epsilon_gr <= v1) and (v1 <= granitsa)):
            break
   return x_1, v1_1, v2_1, v1_2, v2_2, S1, S2, olp1, olp2, h_, C1, C2, n_

#____________________________________________________________________________________________________________________________
#Терминал
print('_______________________________________________________________________________________________')
print('Команда Эльвина | Лабораторная работа №1 | Численное решение задачи Коши для ОДУ')
print('_______________________________________________________________________________________________')
print('Выберите тип задачи:')
print('Тестовая - 0')
print('Основная №1 - 1')
print('Основная №2 - 2')
zadacha = int(input('Задача: '))
if (zadacha == 0):
   print('Введите начальные Тестовой задачи:')
   x0 = float(input('x0 = '))
   u0 = float(input('u0 = '))
elif (zadacha == 1):
   print('Введите начальные Основной №1 задачи:')
   x0 = float(input('x0 = '))
   u0 = float(input('u0 = '))
elif (zadacha == 2):
   print('Введите начальные Основной №2 задачи:')
   x0 = float(input('x0 = '))
   u0 = float(input('u0 = '))
   u0_ = float(input("u'0 = "))
   u0_02 = [u0, u0_]
   print('Параметры a и b;')
   a = float(input('a = '))
   b = float(input('b = '))
print('Задайте правую границу:')
granitsa = float(input('V = '))
print('Задайте точность выхода на правую границу:')
epsilon_gr = float(input('epsilon = '))
print('Задайте максимальное число шагов:')
Nmax = int(input('Nmax = '))
print('Задайте начальный шаг:')
h = float(input('h = '))
print ('С контролем локальной погрешности?  Введите Да или Нет')
contr_loc_str = input()
if ((contr_loc_str == 'да') or (contr_loc_str == 'Да')):
   print('Задайте контроль локальной погрешности:')
   epsilon = float(input('epsilon = '))
   contr_loc_ = 1
elif ((contr_loc_str == 'нет') or (contr_loc_str == 'Нет')):
   contr_loc_ = 0

#____________________________________________________________________________________________________________________________
#* Здесь считаются задачи
if (zadacha == 0):
   test = RK4_test_O1(x0, u0, h, Nmax)
   n = test[0]
   h = test[1]
   x_1 = test[2]
   v_1 = test[3]
   v_2 = test[4]
   olp = test[5]
   C1 = test[6]
   C2 = test[7]
   S = test[8]

   tt = test_true(x0, u0, x_1)
   tt_x = tt[0]
   tt_u = tt[1]

   plt.plot(tt_x, tt_u, 'o-', linewidth = 2.0, label = 'u(x)')
   plt.plot(x_1, v_1, 'o-', linewidth = 2.0, label = 'v(x)')
   plt.xlabel('x')
   plt.ylabel('u(x)   v(x)')
   plt.legend()
   plt.grid()
   plt.show()
elif (zadacha == 1):
   Osnov1 = RK4_test_O1(x0, u0, h, Nmax)
   n = Osnov1[0]
   h = Osnov1[1]
   x_1 = Osnov1[2]
   v_1 = Osnov1[3]
   v_2 = Osnov1[4]
   olp = Osnov1[5]
   C1 = Osnov1[6]
   C2 = Osnov1[7]
   S = Osnov1[8]

   plt.plot(x_1, v_1, 'o-', linewidth = 2.0, label = 'v(x)')
   plt.xlabel('x')
   plt.ylabel('v(x)')
   plt.legend()
   plt.grid()
   plt.show()
elif (zadacha == 2):
   Osnov2 = RK4_O2(x0, u0_02, h, Nmax)
   x_1 = Osnov2[0]
   v1_1 = Osnov2[1]
   v2_1 = Osnov2[2]
   v1_2 = Osnov2[3]
   v2_2 = Osnov2[4]
   S1 = Osnov2[5]
   S2 = Osnov2[6]
   olp1 = Osnov2[7]
   olp2 = Osnov2[8]
   h = Osnov2[9]
   C1 = Osnov2[10]
   C2 = Osnov2[11]
   n = Osnov2[12]
   plt.plot(x_1, v1_1, 'o-', linewidth = 2.0, label = 'v1(x)')
   plt.plot(x_1, v2_1, 'o-', linewidth = 2.0, label = 'v2(x)')
   plt.xlabel('x')
   plt.ylabel('v1(x)    v2(x)')
   plt.legend()
   plt.grid()
   plt.show()

   plt.plot(v1_1, v2_1, 'o-', linewidth = 2.0, label = 'v(x)')
   plt.xlabel('x')
   plt.ylabel('v(x)')
   plt.legend()
   plt.grid()
   plt.show()

#____________________________________________________________________________________________________________________________
#* Здесь находятся значения для справки
if ((zadacha == 0) or (zadacha == 1)):
   abs_olp_ = []
   for i in range(1, len(olp)):
      abs_olp = abs(olp[i])
      abs_olp_.append(abs_olp)
   max_S = max(abs_olp_)
   k_max_S = abs_olp_.index(max_S) + 1
   x_max_S = x_1[k_max_S]
   min_S = min(abs_olp_)
   k_min_S = abs_olp_.index(min_S) + 1
   x_min_S = x_1[k_min_S]
   min_S = min(abs_olp_)

   max_h = max(h)
   k_max_h = h.index(max_h)
   x_max_h = x_1[k_max_h]
   min_h = min(h)
   k_min_h = h.index(min_h)
   x_min_h = x_1[k_min_h]

   for i in range(0, len(x_1)):
      x_1[i] = round(x_1[i], 13)
elif (zadacha == 2):
   abs_olp_1 = []
   abs_olp_2 = []
   for i in range(1, len(olp1)):
      abs_olp1 = abs(olp1[i])
      abs_olp_1.append(abs_olp1)
      abs_olp2 = abs(olp2[i])
      abs_olp_2.append(abs_olp2)
   max_S_1 = max(abs_olp_1)
   k_max_S_1 = abs_olp_1.index(max_S_1) + 1
   x_max_S_1 = x_1[k_max_S_1]
   min_S_1 = min(abs_olp_1)
   k_min_S_1 = abs_olp_1.index(min_S_1) + 1
   x_min_S_1 = x_1[k_min_S_1]

   max_S_2 = max(abs_olp_2)
   k_max_S_2 = abs_olp_2.index(max_S_2) + 1
   x_max_S_2 = x_1[k_max_S_2]
   min_S_2 = min(abs_olp_2)
   k_min_S_2 = abs_olp_2.index(min_S_2) + 1
   x_min_S_2 = x_1[k_min_S_2]

   max_h = max(h)
   k_max_h = h.index(max_h)
   x_max_h = x_1[k_max_h]
   min_h = min(h)
   k_min_h = h.index(min_h)
   x_min_h = x_1[k_min_h]

   for i in range(0, len(x_1)):
      x_1[i] = round(x_1[i], 13)


#____________________________________________________________________________________________________________________________
#Справка
print('_______________________________________________________Справка_______________________________________________________')
if (zadacha == 0):
   print('Тип задачи: Тестова')
elif (zadacha == 1):
   print('Тип задачи: Основная №1')
elif (zadacha == 2):
   print('Тип задачи: Основная №2')
print('Метод Рунге Кутта порядка p = 4')
print('Начало счета = ', x0, '  ', 'Начальное значение = ', u0)
print('Точка выхода (условие остановки счета) = ', granitsa)
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
if ((zadacha == 0) or (zadacha == 1)):
   print('Выход к границе заданной точности  = ', granitsa - x_1[-1])
   print('Текущий x = ', x_1[-1], '  ', 'Текущий v = ', v_1[-1])
   print('Максимальная контрольная величина: ', max_S/16, '  ', 'при x = ', x_max_S)
   print('Минимальная контрольная величина: ', min_S/16, '  ', 'при x = ', x_min_S)
   print('Число увеличений шага: ', C1)
   print('Число уменьшений шага: ', C2)
   print('Максимальный шаг: ', max_h, '  ', 'при x = ', x_max_h)
   print('Минимальный шаг: ', min_h, '  ', 'при x = ', x_min_h)
elif(zadacha == 2):
   print('Выход к границе заданной точности по п  = ', granitsa - x_1[-1])
   print('Текущий x = ', x_1[-1], '  ', 'Текущий v = ({v1_1[-1]} , {v2_2[-1]}) ')
   print('Максимальная контрольная величина по первой компоненте v: ', max_S_1/16, '  ', 'при x = ', x_max_S_1)
   print('Минимальная контрольная величина по первой компоненте v: ', min_S_1/16, '  ', 'при x = ', x_min_S_1)
   print('Максимальная контрольная величина по второй компоненте v: ', max_S_2/16, '  ', 'при x = ', x_max_S_2)
   print('Минимальная контрольная величина по второй компоненте v: ', min_S_2/16, '  ', 'при x = ', x_min_S_2)
   print('Число увеличений шага: ', C1)
   print('Число уменьшений шага: ', C2)
   print('Максимальный шаг: ', max_h, '  ', 'при x = ', x_max_h)
   print('Минимальный шаг: ', min_h, '  ', 'при x = ', x_min_h)
print('_____________________________________________________________________________________________________________________')

#____________________________________________________________________________________________________________________________
#* Таблица
if (zadacha == 0):
   S_ = np.array(S)
   S_ = S_*15
   u = np.array(tt_u)
   v_array = np.array(v_1)
   E = []
   for i in range(0, len(v_1)):
      E_ = u[i] - v_array[i]
      E_ = abs(E_)
      E.append(E_)
   table = PrettyTable()
   table.add_column("n", n)
   table.add_column("x", x_1)
   table.add_column("v1", v_1)
   table.add_column("v2", v_2)
   table.add_column("v2-v1", S_)
   table.add_column("OLP", olp)
   table.add_column("h", h)
   table.add_column("u", u)
   table.add_column("|u - v|", E)

   print(table)
elif (zadacha == 1):
   S_ = np.array(S)
   S_ = S_*15
   table = PrettyTable()
   table.add_column("n", n)
   table.add_column("x", x_1)
   table.add_column("v1", v_1)
   table.add_column("v2", v_2)
   table.add_column("v2-v1", S_)
   table.add_column("OLP", olp)
   table.add_column("h", h)
   print(table)
elif (zadacha == 2):
   S_1 = np.array(S1)
   S_1 = S_1*15
   S_2 = np.array(S2)
   S_2 = S_2*15
   table = PrettyTable()
   table.add_column("n", n)
   table.add_column("x", x_1)
   table.add_column("v1_1", v1_1)
   table.add_column("v2_1", v2_1)
   table.add_column("v1_2", v1_2)
   table.add_column("v2_2", v2_2)
   table.add_column("v2-v1", S_1)
   table.add_column("v2-v1", S_2)
   table.add_column("OLP1", olp1)
   table.add_column("OLP2", olp2)
   table.add_column("h", h)
   print(table)


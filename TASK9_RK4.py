import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# _________________________________________________________________
# Обозначения
# a - вводится с клавиатуры 
# theta(0) - вводится с клавиатуры 
# x0, u0 - начальные условия вводится с клавиатуры 
# _________________________________________________________________
# h0 = 0.0001 - начальный шаг
# Nmax > 100000 - максимальное число шагов
# epsilon_gr = 0.5 * 10**(-6)
# epsilon - вводится с клавиатуры - для контроля ЛП сверху
# epsilon_min - для контроля ЛП снизу, вычисляется из epsilon
# _________________________________________________________________

def task(x, u):
   dudx = -1 * a * (u - theta)  
   return dudx

def RK4(x0, v0, h, Nmax):
   i = 1
   x1 = x0                #с обычным шагом
   v1 = v0                #с обычным шагом
   x2 = x0                #с половинным шагом
   v2 = v0                #с половинным шагом
                   
   olp_ = [0]             #мн-во, содержащее все значения ОЛП на каждом шаге
   h_ = [0]               #мн-во, содержащее все значения шагов
   x_1 = [x0]               
   v_1 = [v0]
   n_ = [0]
   x_2 = [x0]
   v_2 = [v0]
   
   C1 = 0                 #счетчик удвоения шага
   C2 = 0                 #счетчик деления шага
   while i < Nmax + 1:
      k1_1 = task(x1, v1)
      k2_1 = task(x1 + h / 2, v1 + h / 2 * k1_1)
      k3_1 = task(x1 + h / 2, v1 + h / 2 * k2_1)
      k4_1 = task(x1 + h, v1 + h * k3_1)
      x1 = x1 + h
      v1 = v1 + h / 6 * (k1_1 + 2*k2_1+ 2*k3_1 + k4_1)

      j = i - 1
      while j < i + 1:
         k1_2 = task(x2, v2)
         k2_2 = task(x2 + h/4, v2 + 0.25 * h * k1_2)
         k3_2 = task(x2 + h/4, v2 + 0.25 * h * k2_2)
         k4_2 = task(x2 + h/2, v2 + 0.5 * h * k3_2)

         x2 = x2 + h/2
         v2 = v2 + h/12 * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
         j = j + 1
      x_2.append(x2)
      v_2.append(v2)

      S = (v2 - v1)/15             #контрольная величина
      olp = S*16                   #оценка ЛП

      if (contr_loc_ == 1):
         if((epsilon/32) <= abs(S) <= epsilon):
            olp_.append(olp)
            h_.append(h)
            x_1.append(x1)
            v_1.append(v1)
            n_.append(i)

            h = h                     #шаг не изменяется
            i = i + 1
         elif(abs(S) <= (epsilon/32)):
            olp_.append(olp)
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
            olp_.append(olp)
            h_.append(h)
            x_1.append(x1)
            v_1.append(v1)
            x_2.append(x2)
            v_2.append(v2)
            n_.append(i)
            
            C2 = C2 + 1
            h = h/2
            i = i + 1
      
      else:
         olp_.append(olp)
         x_1.append(x1)
         v_1.append(v1)
         h_.append(h)
         n_.append(i)

         C1 = 0
         C2 = 0
         i = i + 1

      if((v0 >= theta)):
         if((v1 <= theta + epsilon_gr) and (theta <= v1)):
            break
      else:    
         if((theta - epsilon_gr <= v1) and (v1 <= theta)):              #If we when out of the border
            break
   return n_, h_, x_1, v_1, v_2, olp_, C1, C2

#Here we start the terminal
print('_______________________________________________________________________________________________')
print('Вивас Каролина | Команда Эльвина | Задача №9')
print('Вариант 7')
print('Остывание разогретого тела, помещенного с целью охлаждения в поток жидкости или газа \nимеющего постоянную температуру thetha, описывается дифференциальным уравнением:')
print('        du/dx = -a(u - theta) \n            u(0) = u0.')
print('Здесь a - постоянный, положительный коэффициент пропорциональности, \nu(x) - температура тела в момент времени x, u0 - температура тела \nв начальный момент времеи. Задача находит численную зависимость \nтемпературы от времени')
print('_______________________________________________________________________________________________')
print('Введите начальные условия задачи:')                      
x0 = float(input('Начальное время x0 = '))
u0 = float(input('Начальная температура u0 = '))
print('Введите параметры задачи:')                  
a = float(input('Положительный коэффициент пропорциональности a = '))
theta = float(input('Температура жидкости/газа theta = '))
print('Задайте точность выхода по температуре сверху:')
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
h = float(input('h = '))

# n_, h_, x_1, v_1, v_2, olp_, C1, C2 
test = RK4(x0, u0, h, Nmax)
n = test[0]
h = test[1]
x_1 = test[2]
v_1 = test[3]
v_2 = test[4]
olp = test[5]
C1 = test[6]
C2 = test[7]

for i in range(0, len(x_1)):
   x_1[i] = round(x_1[i], 13)
abs_olp_ = []
for i in range(0, len(olp)):
   abs_olp = abs(olp[i])
   abs_olp_.append(abs_olp)
max_S = max(abs_olp_[1:])
k_max_S = abs_olp_.index(max_S)
x_max_S = x_1[k_max_S]
min_S = min(abs_olp_[1:])
k_min_S = abs_olp_.index(min_S)
x_min_S = x_1[k_min_S]

max_h = max(h[1:])
k_max_h = h.index(max_h)
x_max_h = x_1[k_max_h]
min_h = min(h[1:])
k_min_h = h.index(min_h)
x_min_h = x_1[k_min_h]

#Справка
print('_______________________________________Справка_________________________________________________')
print('№ Варианта задания: 7')
print('Тип задачи: Основная')
print('Метод Рунге Кутта порядка p = 4   способ счета одношаговый')
print('Время начала счета = ', x0, '  ', 'Начальная температура = ', u0, '°C')                               
print('Точка выхода (условие остановки счета) = ', theta, '°C')
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
print('Выход к границе заданной точности  = ', theta - v_1[-1])                           #v temperature
print('Текущее время = ', x_1[-1], '  ', 'Текущая температура = ', v_1[-1], '°C')
print('Максимальная контрольная величина: ', max_S/16, '  ', 'при x = ', x_max_S)
print('Минимальная контрольная величина: ', min_S/16, '  ', 'при x = ', x_min_S)
if (contr_loc_ == 1):
   print('Число увеличений шага: ', C1)
   print('Число уменьшений шага: ', C2)
   print('Максимальный шаг: ', max_h, '  ', 'при x = ', x_max_h)
   print('Минимальный шаг: ', min_h, '  ', 'при x = ', x_min_h)
elif (contr_loc_ == 0):
   print('Увеличения или уменьшения шага не происходило')
   print('Шаг не менялся h = ', max_h)
print('_______________________________________________________________________________________________')

# Table
table = PrettyTable()
table.add_column("n", n)
table.add_column("h", h)
table.add_column("x", x_1)
table.add_column("v1", v_1)
table.add_column("v2", v_2)
table.add_column("OLP", olp)
print(table)

#График
plt.plot(x_1, v_1,'o-', linewidth = 2.0, label='График изменения температуры от времени')
plt.legend()
plt.xlabel('time')
plt.ylabel("temperature")
plt.grid()
plt.show()
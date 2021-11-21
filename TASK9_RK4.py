import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def task(x, u):
   dudx = -1 * a * (u - theta)
   return dudx

#* Рунге-Кутта
def RK4(x0, v0, h, Nmax):
   i = 1
   x1 = x0
   v1 = v0
   x2 = x0
   v2 = v0
   olp_ = [0]
   h_ = [0]
   x_1 = [x0]
   v_1 = [v0]
   n_ = [0]
   x_2 = [x0]
   v_2 = [v0]
   
   C1 = 0
   C2 = 0
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
      S = (v2 - v1)/15
      olp = S*16
      if (contr_loc_ == 1):
         if((epsilon/32) <= abs(S) <= epsilon):
            olp_.append(olp)
            h_.append(h)
            x_1.append(x1)
            v_1.append(v1)
            n_.append(i)
            h = h
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
         if((theta - epsilon_gr <= v1) and (v1 <= theta)):
            break
   return n_, h_, x_1, v_1, v_2, olp_, C1, C2

#* Терминал
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

#* Тут считается РК
# n_, h_, x_1, v_1, v_2, olp_, C1, C2 
n, h, x_1, v_1, v_2, olp, C1, C2  = RK4(x0, u0, h, Nmax)

for i in range(0, len(x_1)):
   x_1[i] = round(x_1[i], 11)
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

#* Справка
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

#* Вывод
table = {'n': n, 'h': h, 'x': x_1, 'v_1': v_1, 'v_2': v_2, 'ОЛП': olp}
data = pd.DataFrame(data = table)
data.to_csv("Таблица_Зачада_9.csv", index=False)

plt.plot(x_1, v_1,'o-', linewidth = 2.0, label='График изменения температуры от времени')
plt.legend()
plt.xlabel('t')
plt.ylabel("°C")
plt.grid()
plt.savefig('График_Задача_9.png', bbox_inches='tight')
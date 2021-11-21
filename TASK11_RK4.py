import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# _________________________________________________________________
# Обозначения
# m - вводится с клавиатуры 
# c - вводится с клавиатуры 
# k - вводится с клавиатруы
# x0, u0, u_0 - начальные условия вводится с клавиатуры 
# _________________________________________________________________
# h0 - вводится с клавиатуры
# Nmax - вводится с клавиатуры
# epsilon_gr - вводится с клавиатуры
# epsilon - вводится с клавиатуры
# _________________________________________________________________


def task(x, u0_2):
   dudx = u0_2[1]
   duxy = (-c*u0_2[1] - k* u0_2[0])/m
   return dudx, duxy

def RK4(x0, v0, h, Nmax):
   i = 1
   x1 = x0                          #с обычным шагом
   x2 = x0                          #с половинным шагом
   
   x_1 = [x0]               
   v1_1 = [v0[0]]
   v2_1 = [v0[1]]
    
   x_2 = [x0]
   v1_2 = [v0[0]]
   v2_2 = [v0[1]]
   
   S1 = [0]
   S2 = [0]
   olp1 = [0]             #мн-во, содержащее все значения ОЛП на каждом шаге
   olp2 = [0]
   n_ = [0]
   h_ = [0]      
   C1 = 0                 #счетчик удвоения шага
   C2 = 0                 #счетчик деления шага
   v1 = np.array(v0)
   v2 = np.array(v0)
   while i < Nmax + 1:
      k1_1 = task(x1, v1)
      k1_1 = np.array(k1_1)
      k2_1 = task(x1 + h / 2, v1 + h / 2 * k1_1)
      k2_1 = np.array(k2_1)
      k3_1 = task(x1 + h / 2, v1 + h / 2 * k2_1)
      k3_1 = np.array(k3_1)
      k4_1 = task(x1 + h, v1 + h * k3_1)
      k4_1 = np.array(k4_1)
      x1 = x1 + h
      v1 = v1 + h / 6 * (k1_1 + 2*k2_1+ 2*k3_1 + k4_1)
      j = i - 1
      while j < i + 1:
         k1_2 = task(x2, v2)
         k1_2 = np.array(k1_2)
         k2_2 = task(x2 + h/4, v2 + 0.25 * h * k1_2)
         k2_2 = np.array(k2_2)
         k3_2 = task(x2 + h/4, v2 + 0.25 * h * k2_2)
         k3_2 = np.array(k3_2)
         k4_2 = task(x2 + h/2, v2 + 0.5 * h * k3_2)
         k4_2 = np.array(k4_2)
         x2 = x2 + h/2
         v2 = v2 + h/12 * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
         j = j + 1
      x_2.append(x2)
      v1_2.append(v2[0])
      v2_2.append(v2[1])

      S_1 = (v2[0] - v1[0])/15             #контрольная величина
      S_2 = (v2[1] - v1[1])/15             #контрольная величина
      olp_1 = S1*16                        #оценка ЛП
      olp_2 = S2*16                        #оценка ЛП

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

            olp1.append(olp_1)
            olp2.append(olp_2)
            S1.append(S_1)
            S2.append(S_2)
            x_1.append(x1)
            v1_1.append(v1[0])
            v2_1.append(v1[1])
            x_2.append(x2)
            v1_2.append(v2[0])
            v2_2.append(v2[1])
            h_.append(h)
            n_.append(i)

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

      """ if((v1 <= granitsa + epsilon_gr) and (granitsa <= v1)):
         break """
   return n_, h_, x_1, v1_1, v2_1, v1_2, v2_2, S1, S2, olp1, olp2, C1, C2

#Here we start the terminal
print('_______________________________________________________________________________________________')
print('Джафарзаде Эльвин | Команда Эльвина | Задача №11')
print('Вариант 4')
print('Груз массы m может совершать прямолинейные перемещения вдоль оси абсцисс по горизонтальной')
print('плоскости без трения. Для стабилизации положения груза используется аналогичная система,')
print('но пружина с нелинейной характеристикой отсутствует.')
print('Положение груза в системе описывает линейное дифференциальное уравнение:')
print('        m * (d^2/du^2) + c * (du/dx) + k * u = 0 \n            u(x0) = u0;      du/dx(x0) = (du/dx)0.')
print('Здесь к - жесткость пружины, с - коэффициент демпфирования, m - масса груза, ')
print('u(x) - отклонение груза от состояния равновесия в момент времени x, u0 - отклонение тела')
print('в начальный момент времени. Задача находит численную зависимость')
print('отклонения груза от времени.')
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
print('Задайте точность выхода по скорости:')
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

# n_, h_, x_1, v1_1, v2_1, v1_2, v2_2, S1, S2, olp1, olp2, C1, C2
test = RK4(x0, u0_2, h, Nmax)
n = test[0]
h = test[1]
x_1 = test[2]
v1_1 = test[3]
v2_1 = test[4]
v1_2 = test[5]
v2_2 = test[6]
S1 = test[7]
S2 = test[8]
olp1 = test[9]
olp2 = test[10]
C1 = test[11]
C2 = test[12]

for i in range(0, len(x_1)):
   x_1[i] = round(x_1[i], 13)
""" abs_olp_ = []
for i in range(0, len(olp)):
   abs_olp = abs(olp[i])
   abs_olp_.append(abs_olp)
max_S = max(abs_olp_[1:])
k_max_S = abs_olp_.index(max_S)
x_max_S = x_1[k_max_S]
min_S = min(abs_olp_[1:])
k_min_S = abs_olp_.index(min_S)
x_min_S = x_1[k_min_S] """

max_h = max(h[1:])
k_max_h = h.index(max_h)
x_max_h = x_1[k_max_h]
min_h = min(h[1:])
k_min_h = h.index(min_h)
x_min_h = x_1[k_min_h]

#Справка
print('_______________________________________Справка_________________________________________________')
print('№ Варианта задания: 4')
print('Тип задачи: Основная')
print('Метод Рунге Кутта порядка p = 4   способ счета одношаговый')
print('Время начала счета = ', x0, '  ', 'Начальная отклонение груза = ', u0, 'см')                               
#print('Точка выхода (условие остановки счета) = ', u, 'см')
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
print('Выход к границе заданной точности  = ',)                           #v temperature
#print('Текущее время = ', x_1[-1], '  ', 'Текущее отклонение = ', u[-1], 'см')
#print('Максимальная контрольная величина: ', max_S/16, '  ', 'при x = ', x_max_S)
#print('Минимальная контрольная величина: ', min_S/16, '  ', 'при x = ', x_min_S)
if (contr_loc_ == 1):
   print('Число увеличений шага: ', C1)
   print('Число уменьшений шага: ', C2)
   print('Максимальный шаг: ', max_h, '  ', 'при x = ', x_max_h)
   #print('Минимальный шаг: ', min_h, '  ', 'при x = ', x_min_h)
elif (contr_loc_ == 0):
   print('Увеличения или уменьшения шага не происходило')
   print('Шаг не менялся h = ', max_h)
print('_______________________________________________________________________________________________')

# Table

table = PrettyTable()
table.add_column("n", n)
table.add_column("h", h)
table.add_column("x", x_1)
table.add_column("v1_1", v1_1)
table.add_column("v2_1", v2_1)
table.add_column("v1_2", v1_2)
table.add_column("v2_2", v2_2)
print(table)


plt.plot(v1_1, v2_1, 'o-', linewidth = 2.0, label = 'vector V')
plt.xlabel('v[1]')
plt.ylabel('v[2]')
plt.legend()
plt.grid()
plt.show()
import math
import matplotlib.pyplot as plt
import numpy as np

#* Здесь всякие параметры, которые будут потом браться из интерфейса
B = 10
epsilon = 0.001
epsilon_gr = 0.001
a = 3
b = 2

#* Здесь описан критерий выхода на правую границу
def control_gran(B, epsilon_gr, x):
   x_ = x
   B_ = B
   epsilon_gr_ = epsilon_gr
   if((B_ - epsilon_gr_ <= x_) and (x_ <= B_)):
      return True
   else:
      return False

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

#* Здесь описаны РК4 для каждой задачи
def RK4_test(x, v, step, n):
   h = step
   i = 1

   x1 = x
   v1 = v
   x2 = x
   v2 = v

   x_1 = [x]
   v_1 = [v]
   x_2 = [x]
   v_2 = [v]

   S = [0]
   e = [0]
   h_ = [h]

   while i < n + 1:
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

      S_ = v2 - v1
      e_ = (S_/31) * 32

      if((epsilon/32) <= abs(S_) and abs(S_) <= epsilon):
         e.append(e_)
         S.append(S_)
         h_.append(h)
         x_1.append(x1)
         v_1.append(v1)

         h = h
         i = i + 1
      elif(abs(S_) <= (epsilon/32)):
         e.append(e_)
         S.append(S_)
         h_.append(h)
         x_1.append(x1)
         v_1.append(v1)

         h = 2*h
         i = i + 1
      else:
         x_1.pop(-1)
         v_1.pop(-1)
         x1 = x_1[-1]
         v1 = v_2[-1]

         x_2.pop(-1)
         v_2.pop(-1)
         x2 = x_2[-1]
         v2 = v_2[-1]

         h = h/2
         i = i - 1
         j = j - 1

      if (control_gran(B, epsilon_gr, x1) == True):
         break
   return x_1, v_1, v_2, S, e, h_

def RK4_O1(x, v, step, n):
   h = step
   i = 1
   
   x1 = x
   v1 = v
   x2 = x
   v2 = v

   x_1 = [x]
   v_1 = [v]
   x_2 = [x]
   v_2 = [v]

   S = [0]
   e = [0]
   h_ = [h]

   while i < n + 1:
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

      S_ = v2 - v1
      e_ = (S_/31) * 32
      if((epsilon/32) <= abs(S_) and abs(S_) <= epsilon):
         e.append(e_)
         S.append(S_)
         h_.append(h)
         x_1.append(x1)
         v_1.append(v1)

         h = h
         i = i + 1
      elif(abs(S_) <= (epsilon/32)):
         e.append(e_)
         S.append(S_)
         h_.append(h)
         x_1.append(x1)
         v_1.append(v1)

         h = 2*h
         i = i + 1
      else:
         x_1.pop(-1)
         v_1.pop(-1)
         x1 = x_1[-1]
         v1 = v_2[-1]

         x_2.pop(-1)
         v_2.pop(-1)
         x2 = x_2[-1]
         v2 = v_2[-1]

         h = h/2
         i = i - 1
         j = j - 1

      if (control_gran(B, epsilon_gr, x1) == True):
         break
   return x_1, v_1, v_2, S, e, h_

def RK4_O2(x, v, step, n):
   h = step
   i = 1

   x1 = x
   v1 = v
   x2 = x
   v2 = v

   x_1 = [x1]
   v1_ = []
   v1_1 = [v1[0]]
   v2_1 = [v1[1]]

   x_2 = [x2]
   v2_ = []
   v1_2 = [v2[0]]
   v2_2 = [v2[1]]

   S1 = [0]
   S2 = [0]
   e1 = [0]
   e2 = [0]
   h_ = [h]

   while i < n + 1:
      v1_ = np.array(v1)
      k1_1 = O2(x1, v1_)
      k1_1 = np.array(k1_1)
      k2_1 = O2(x1 + h/2, v1_ + 0.5 * h * k1_1)
      k2_1 = np.array(k2_1)
      k3_1 = O2(x1 + h/2, v1_ + 0.5 * h * k2_1)
      k3_1 = np.array(k3_1)
      k4_1 = O2(x1 + h, v1_ + h * k3_1)
      k4_1 = np.array(k4_1)

      x1 = x1 + step
      v1 = v1_ + step/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
      x_1.append(x1)
      v1_1.append(v1[0])
      v2_1.append(v1[1])
      
      j = i - 1
      while j < i + 1:
         v2_ = np.array(v2)
         k1_2 = O2(x2, v2_)
         k1_2 = np.array(k1_2)
         k2_2 = O2(x2 + h/4, v2_ + 0.25 * h * k1_2)
         k2_2 = np.array(k2_2)
         k3_2 = O2(x2 + h/4, v2_ + 0.25 * h * k2_2)
         k3_2 = np.array(k3_2)
         k4_2 = O2(x2 + h/2, v2_ + 0.5 * h * k3_2)
         k4_2 = np.array(k4_2)

         x2 = x2 + step/2
         v2 = v2_ + step/12 * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
         j = j + 1
      x_2.append(x2)
      v1_2.append(v2[0])
      v2_2.append(v2[1])

      S_1 = v2[0] - v1[0]
      S_2 = v2[1] - v1[1]
      e1_ = (S_1/31) * 32
      e2_ = (S_2/31) * 32

      if(((epsilon/32) <= abs(S_1) and abs(S_1) <= epsilon) and ((epsilon/32) <= abs(S_2) and abs(S_2) <= epsilon)):
         e1.append(e1_)
         e2.append(e2_)
         S1.append(S_1)
         S2.append(S_2)
         h_.append(h)

         h = h
         i = i + 1
      elif((abs(S_1) <= (epsilon/32)) and (abs(S_2) <= (epsilon/32))):
         e1.append(e1_)
         e2.append(e2_)
         S1.append(S_1)
         S2.append(S_2)
         h_.append(h)
         
         h = 2*h
         i = i + 1
      else:
         x_1.pop(-1)
         v1_1.pop(-1)
         v2_1.pop(-1)
         x1 = x_1[-1]
         v1 = [v1_1[-1], v2_1[-1]]

         x_2.pop(-1)
         v1_2.pop(-1)
         v2_2.pop(-1)
         x2 = x_2[-1]
         v2 = [v1_2[-1], v2_2[-1]]

         h = h/2
         i = i - 1
         j = j - 1

      if (control_gran(B, epsilon_gr, x1) == True):
         break
   return x_1, v1_1, v2_1, v1_2, v2_2, S1, S2, e1, e2, h_

""" g = RK4_test(12, 3, 3, 12) """


def ClearPlot():
   fig = plt.figure()
   plt.figure().clear()
   plt.close()
   plt.cla()
   plt.clf() 


""" 
zadacha_test = RK4_test(x0, v0, h, Nmax)
zadacha_test_x = zadacha_test[0]
zadacha_test_v1 = zadacha_test[1]
zadacha_test_v2 = zadacha_test[2]
zadacha_test_S = zadacha_test[3]
zadacha_test_e = zadacha_test[4]
zadacha_test_h = zadacha_test[5] 
"""


def Plot_test(x0, u0, v0, h, n):
   tt = test_true(x0, u0, h, n)
   tt_x = tt[0]
   tt_u = tt[1]

   tf = RK4_test(x0, v0, h, n)
   tf_x = tf[0]
   tf_v = tf[1]

   plt.plot(tt_x, tt_u, 'o-', linewidth = 2.0, label = 'u(x)')
   plt.plot(tf_x, tf_v, 'o-', linewidth = 2.0, label = 'v(x)')
   plt.xlabel('x')
   plt.ylabel('u(x)   v(x)')
   plt.legend()
   plt.grid()
   plt.savefig('Plot_test.png', bbox_inches='tight')
   ClearPlot()


def Plot_O1(x0, v0, h, n):
   o1 = RK4_O1(x0, v0, h, n)
   o1_x = o1[0]
   o1_v = o1[1]

   plt.plot(o1_x, o1_v, 'o-', linewidth = 2.0, label = 'v(x)')
   plt.xlabel('x')
   plt.ylabel('v(x)')
   plt.legend()
   plt.grid()
   plt.savefig('Plot_o2.png', bbox_inches='tight')
   ClearPlot()


def Plot_O2_graf(x0, v0, h, n):
   o2 = RK4_O2(x0, v0, h, n)
   o2_x = o2[0]
   o2_v1 = o2[1]
   o2_v2 = o2[3]

   plt.plot(o2_x, o2_v1, 'o-', linewidth = 2.0, label = 'v1(x)')
   plt.plot(o2_x, o2_v2, 'o-', linewidth = 2.0, label = 'v2(x)')
   plt.xlabel('x')
   plt.ylabel('v1(x)    v2(x)')
   plt.legend()
   plt.grid()
   plt.savefig('Plot_o2_graf.png', bbox_inches='tight')
   ClearPlot()


def Plot_o2_phase(x0, v0, h, n):
   o2 = RK4_O2(x0, v0, h, n)
   o2_v1 = o2[1]
   o2_v2 = o2[3]

   plt.plot(o2_v1, o2_v2, 'o-', linewidth = 2.0, label = 'v(x)')
   plt.xlabel('x')
   plt.ylabel('v(x)')
   plt.legend()
   plt.grid()
   plt.savefig('Plot_o2_phase.png', bbox_inches='tight')
   ClearPlot()


asf = Plot_O1(0, 2, 0.01, 5)











def test_true(x0, u0, h, n):
   x = x0
   u = u0
   x = np.linspace(x0, n + 1, h)
   i  = 0
   u_ = []
   while i < len(x):
      u = u0 * math.exp(-3/2 * x[i])
      u_.append(u)
      i = i + 1
   return x, u_ 
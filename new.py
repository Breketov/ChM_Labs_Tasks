import math
import matplotlib.pyplot as plt
import numpy as np
granitsa = 10
epsilon = 0.0000001
a = 3
b = 2
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
   while i < n:
      k1_1 = test(x1, v1)
      k2_1 = test(x1 + h/2, v1 + 0.5 * h * k1_1)
      k3_1 = test(x1 + h/2, v1 + 0.5 * h * k2_1)
      k4_1 = test(x1 + h, v1 + h * k3_1)

      x1 = x1 + h
      v1 = v1 + h/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
      x_1.append(x1)
      v_1.append(v1)
      
      j = i - 1
      while j < i+1:
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
      S.append(S_)
      
      if((epsilon/32) <= abs(S[i]) and abs(S[i]) <= epsilon):
         h = h
         i = i + 1
      elif(abs(S[i]) <= (epsilon/32)):
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
   while i < n:
      k1_1 = O1(x1, v1)
      k2_1 = O1(x1 + h/2, v1 + 0.5 * h * k1_1)
      k3_1 = O1(x1 + h/2, v1 + 0.5 * h * k2_1)
      k4_1 = O1(x1 + h, v1 + h * k3_1)

      x1 = x1 + h
      v1 = v1 + h/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
      x_1.append(x1)
      v_1.append(v1)
      
      j = i - 1
      while j < i+1:
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
      S.append(S_)
      
      if((epsilon/32) <= abs(S[i]) and abs(S[i]) <= epsilon):
         h = h
         i = i + 1
      elif(abs(S[i]) <= (epsilon/32)):
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

def RK4_O2(x, v, step, n):
   h = step
   i = 1

   x1 = x
   v1 = v
   x2 = x
   v2 = v

   x_1 = [x]
   v1_1 = [v[0]]
   v2_1 = [v[1]]

   x_2 = [x]
   v1_2 = [v[0]]
   v2_2 = [v[1]]

   S = [0]
   while i < n:
      k1 = zadachi.o2(x1, v1)
      k2 = zadachi.o2(x1 + step / 2, v_1 + 0.5 * step * k1)
      k3 = zadachi.o2(x1 + step / 2, v_1 + 0.5 * step * k2)
      k4 = zadachi.o2(x1 + step, v_1 + step * k3)
      x_1.append(x1)
      v1_1.append(v1[0])
      v2_1.append(v1[1])

            x1 = x1 + step
            v1 = v_1 + step/6 * (k1_ + 2*k2_ + 2*k3_ + k4_)
            print(x1, '   ', v1[0], '   ', v1[1])
        
        print(x2, '   ', v2[0], '   ', v2[1])
        for i in range(0, 2*(n + 1)):
            v_2 = np.array(v2)
            k1 = zadachi.o2(x2, v_2)
            k1_ = np.array(k1)
            k2 = zadachi.o2(x2 + step / 2, v_2 + 0.5 * step * k1_)
            k2_ = np.array(k2)
            k3 = zadachi.o2(x2 + step / 2, v_2 + 0.5 * step * k2_)
            k3_ = np.array(k3)
            k4 = zadachi.o2(x2 + step, v_2 + step * k3_)
            k4_ = np.array(k4)
            x_2.append(x2)
            v1_2.append(v2[0])
            v2_2.append(v2[1])

            x2 = x2 + step
            v2 = v_2 + step/6 * (k1_ + 2*k2_ + 2*k3_ + k4_)
            print(x1, '   ', v2[0], '   ', v2[1])

        for i in range(0, n+1):
            v1__1 = np.array(v1_1)
            v1__2 = np.array(v1_2)
            v2__1 = np.array(v2_1)
            v2__2 = np.array(v2_2)
            S1 = v1__1[i] - v1__2[2*i]
            S2 = v2__1[i] - v2__2[2*i]
            
            print(S1, '   ', S2)
        return x_1, v1_1, v2_1, x_2, v1_2, v2_2, S1, S2

test = RK4_test(0, 2, 0.1, 5)

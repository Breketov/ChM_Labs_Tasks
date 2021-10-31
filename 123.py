import math
import matplotlib.pyplot as plt
import numpy as np

#* Здесm всякие параметры, которые будут потом браться из интерфейса
granitsa = 10
epsilon = 0.001
a = 3
b = 2

def O2(x, u):
   du_ = u[1]
   dx_= -a* (u[1]**2) - b * u[0]
   abc = np.array(du_, dx_)
   return abc


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
      k2_1 = O2(x1 + h/2, v1_ + 0.5 * h * k1_1)
      k3_1 = O2(x1 + h/2, v1_ + 0.5 * h * k2_1)
      k4_1 = O2(x1 + h, v1_ + h * k3_1)

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
         
         h = h
         i = i + 1
      elif((abs(S_1) <= (epsilon/32)) and (abs(S_2) <= (epsilon/32))):
         e1.append(e1_)
         e2.append(e2_)
         S1.append(S_1)
         S2.append(S_2)

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
         v2 = [v1_2[-1], v2_2]

         h = h/2
         i = i - 1
         j = j - 1

      if((granitsa - epsilon <= x1) and (x1 <= granitsa)):
         break
   return x_1, v1_1, v2_1, v1_2, v2_2, S1, S2, e1, e2, h_

o2 = RK4_O2(0, [2, 3], 0.001, 5)
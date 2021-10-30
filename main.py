import math
import matplotlib.pyplot as plt
import numpy as np

granitsa = 10
epsilon = 0.001
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

      if((epsilon/32) <= abs(S[i]) and abs(S[i]) <= epsilon):
         e.append(e_)
         S.append(S_)
         h_.append(h)

         h = h
         i = i + 1
      elif(abs(S[i]) <= (epsilon/32)):
         e.append(e_)
         S.append(S_)
         h_.append(h)

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
   
      if((granitsa - epsilon <= x1) and (x1 <= granitsa)):
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
      x_1.append(x1)
      v_1.append(v1)
      
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
      if((epsilon/32) <= abs(S[i]) and abs(S[i]) <= epsilon):
         e.append(e_)
         S.append(S_)
         h_.append(h)

         h = h
         i = i + 1
      elif(abs(S[i]) <= (epsilon/32)):
         e.append(e_)
         S.append(S_)
         h_.append(h)

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

      if((granitsa - epsilon <= x1) and (x1 <= granitsa)):
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

""" test = RK4_test(0, 2, 0.1, 5) """

o2 = RK4_O2(0, [2, 3], 0.001, 5)


""" 
u0 = 2
x = np.linspace(0, 0.05, 6)
i = 0
u_test = []
while i < len(x):
   u_ = u0 * math.exp(-3/2 * x[i])
   u_test.append(u_)
   i = i + 1


e = RK4.RK4_test(0, 2, 0.01, 6)
e_x = e[0]
e_v = e[1]

plt.plot(e_x, e_v, 'r--', linewidth = 0.5)
plt.plot(x, u_test, 'b--', linewidth = 0.5)
plt.axis([0, 0.05, 0, 2])
plt.show() 
"""
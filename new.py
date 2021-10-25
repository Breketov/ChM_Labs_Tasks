import numpy as np
import math

class zadachi():
   def __init__(self) -> None:
      pass
   
   def test(x, u):
      dudx = -1* 3/2 * u
      return dudx
    
   def o1(x, u):
      dudx = 1/((1+(x**2))**(1/3))*(u**2) + u - (u**3)*math.sin(10*x)
      return dudx

   def o2(x, u):
      du_ = u[1]
      dx_= -a* (u[1]**2) - b * u[0]
      return du_, dx_

class RK4(zadachi):
   def __init__(self) -> None:
      super().__init__()
   
   def test(x, v, step, n):
      x_ = []
      v_ = []
      print(x, '   ', v)
      for i in range (0, n + 1):
         k1 = zadachi.test(x, v)
         k2 = zadachi.test(x + step / 2, v + 0.5 * step * k1)
         k3 = zadachi.test(x + step / 2, v + 0.5 * step * k2)
         k4 = zadachi.test(x + step, v + step * k3)
         x_.append(x)
         v_.append(v)
         x =  x + step
         v =  v + step/6 * (k1 + 2*k2 + 2*k3 + k4)

         print(x, '   ', v)
      return x_, v_

test = RK4.test(0, 2, 0.01, 5)
print(test)
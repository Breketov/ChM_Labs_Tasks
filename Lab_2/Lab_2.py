import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


ksi = 0.3
ny1 = 1
ny2 = 0
n = 12
C1 = -0.0963958371
C2 = -2.2369374963

C3 = 1.1565176427
C4 = -0.1565176427


C4 = -0.1086898371251591
C1 = -2.441089498977808
C2 = 0.1077561656444751
C3 = 0.8031153039014358

#* Аналитическое решение
def u1(x):
   return C1*np.exp(-np.sqrt(ksi/(ksi**2 + 2))*x) + C2*np.exp(np.sqrt(ksi/(ksi**2 + 2))*x) + 1/ksi
def u2(x):
   return C3*np.exp(-x) + C4*np.exp(x) + np.sin(np.pi*x)/((ksi**2)*(1 + np.pi**2))


""" 
x + y = -7/3, 
z*(1/e) + t*e = 0, 
x*e^(-sqrt(0.3/(2.09))*0.3) + y*e^(sqrt(0.3/(2.09))*0.3) - z*(1/e^(0.3)) - t*e^(0.3) = sin(pi*0.3)/(0.09*(1+pi^2)) - 10/3, 
sqrt(0.3/(2.09))*2.09*x*e^(-sqrt(0.3/(2.09))*0.3) - sqrt(0.3/(2.09))*2.09*y*e^(sqrt(0.3/(2.09))*0.3) - z*0.09*(1/e^(0.3)) + t*0.09*e^(0.3) = -pi*cos(pi*0.3)/(0.09*(1+pi^2))
 """

#* [0, ksi]
def k1(x):
   return x**2 + 2
#* [ksi, 1]
def k2(x):
   return x**2


#* [0, ksi]
def q1(x):
   return x
#* [ksi, 1]
def q2(x):
   return x**2


#* [0, ksi]
def f1(x):
   return 1
#* [ksi, 1]
def f2(x):
   return np.sin(np.pi*x)

def node(i, n):
   return i/n

def test_coef():
   a = [0]
   d = [0]
   fi = [0]
   for i in range(1, n + 1):
      if ((node(i - 1, n) < 0.3) and (node(i, n) < 0.3)):
         a.append(2.09)
      elif ((node(i - 1, n) > 0.3) and (node(i, n) > 0.3)):
         a.append(0.09)
      else:
         a.append(1/(n*(0.3 - node(i - 1, n))/2.09 + n*(node(i, n) - 0.3)/0.09))

   for i in range(1, n):
      if ((node(i + 0.5, n) < 0.3) and (node(i - 0.5, n) < 0.3)):
         d.append(0.3)
      elif ((node(i + 0.5, n) > 0.3) and (node(i - 0.5, n) > 0.3)):
         d.append(0.09)
      else:
         d.append(n*0.3*(0.3 - node(i - 0.5, n)) + n*0.09*(node(i + 0.5, n) - 0.3))
   
   for i in range(1, n):
      if ((node(i + 0.5, n) < 0.3) and (node(i - 0.5, n) < 0.3)):
         fi.append(1)
      elif ((node(i + 0.5, n) > 0.3) and (node(i - 0.5, n) > 0.3)):
         fi.append(np.sin(np.pi*0.3))
      else:
         fi.append(n*(0.3 - node(i - 0.5, n)) + n*np.sin(np.pi*0.3)*(node(i + 0.5, n) - 0.3))

   return a, d, fi


def SLAY():
   12









n = 12
""" test_coef() """









x1 = np.linspace(0, 0.3, 11)
plt.plot(x1, u1(x1),'o-', linewidth = 2.0, label='траектория до точки разрыва')
x2 = np.linspace(0.3, 1, 11)
plt.plot(x2, u2(x2),'o-', linewidth = 2.0, label='траетория после точки разрыва')
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x)')
plt.grid()
plt.savefig('График истинной траектории.png')
plt.show()
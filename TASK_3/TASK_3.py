import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def task(x, u):
   du1dx = -500.005*u[0] + 499.995*u[1]
   du2dx = 499.995*u[0] - 500.005*u[1]
   return np.array([du1dx, du2dx])

def true_task(x):
   u = []
   for i in range(0, len(x)):
      u.append(list(3*np.exp(-1000*x[i])*np.array([-1, 1]) + 10*np.exp(-0.01*x[i])*np.array([1, 1])))
   return np.array(u)

def plot():
   x = np.linspace(0, 1000, 51)
   u = true_task(x)
   plt.plot(x, u[:, 0], 'o-', label='u1(t) - Истинная траектория')
   plt.plot(x, u[:, 1], 'x-', label='u2(t) - Истинная траектория')
   plt.legend()
   plt.xlabel('t')
   plt.ylabel('u1(t)   u2(t)')
   plt.grid()
   plt.show()

   x = np.linspace(0, 0.01, 51)
   u = true_task(x)
   plt.plot(x, u[:, 0], 'o-', label='u1(t) - Истинная траектория')
   plt.plot(x, u[:, 1], 'x-', label='u2(t) - Истинная траектория')
   plt.axis([0, 0.01, 6, 14])
   plt.legend()
   plt.xlabel('t')
   plt.ylabel('u1(t)   u2(t)')
   plt.grid()
   plt.show()

   x = np.linspace(0, 0.5, 51)
   u = true_task(x)
   plt.plot(x, u[:, 0], 'o-', label='u1(t) - Истинная траектория')
   plt.plot(x, u[:, 1], 'x-', label='u2(t) - Истинная траектория')
   plt.axis([0, 0.5, 6, 14])
   plt.legend()
   plt.xlabel('t')
   plt.ylabel('u1(t)   u2(t)')
   plt.grid()
   plt.show()

u0 = [7, 13]
x0 = 0
def Euler(h0, x0, v0, Nmax):
   i = 1
   x1 = x0
   v1 = np.array(v0)
   x_1 = [x1]
   v_1 = [v1]
   n = [0]
   E = np.array([[1, 0], [0, 1]])
   A = np.array([[-500.005, 499.995], [499.995, -500.005]])
   while i < 1000:
      B = np.linalg.inv(E - h0*A)
      v1 = B @ v1
      x1 = x1 + h0
      x_1.append(x1)
      v_1.append(v1)
      n.append(i)
      i = i + 1
   return n, x_1, np.array(v_1)

Nmax = 21
h0 = 0.005
v0 = u0
n, x1, v1 = Euler(h0, x0, v0, Nmax)

plt.plot(x1, v1[:, 0], 'o-', label='u1(t) - Истинная траектория')
plt.plot(x1, v1[:, 1], 'x-', label='u2(t) - Истинная траектория')
plt.legend()
plt.xlabel('t')
plt.ylabel('u1(t)   u2(t)')
plt.grid()
plt.show()


#plot()
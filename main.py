import numpy as np
from numpy.core.function_base import linspace
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import cmath

u0 = float(input("Введите начальное условие u при х = 0: "))

def test(u, x):
   dudx = -3/2 * u
   return dudx

x = np.linspace(0, 5, 100)
u = odeint(test, u0, x)

plt.plot(x, u, 'o-', linewidth = 2.0)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.grid()
plt.show()
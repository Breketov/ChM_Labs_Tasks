import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


ksi = 0.4
ny1 = 1
ny2 = 0


#* [1, ksi]
def k1(x):
   return x**2 + 2
#* [ksi, 0]
def k2(x):
   return x**2


#* [1, ksi]
def q1(x):
   return x
#* [ksi, 0]
def q2(x):
   return x**2


#* [1, ksi]
def f1(x):
   return 1
#* [ksi, 0]
def f2(x):
   return np.sin(np.pi*x)

def func_1()


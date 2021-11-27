import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def task(x, u):
   du1dx = -500.005*u[0] + 499.995*u[1]
   du2dx = 499.995*u[0] + 500.005*u[1]


u = [7, 13]

def Euler(x0, v0, h0, Nmax):
   i = 1
   x = [x0]
   v = [v0]
   h = [h0]


   while i < Nmax + 1:
      x[i + 1] = x[i] + h0
      v[i + 1] = v[i] + h0*task(x[i+1], v[i+1]) 
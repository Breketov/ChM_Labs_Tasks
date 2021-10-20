from math import comb
from tkinter import *
from tkinter.ttk import Combobox  

window = Tk()
window.title("Лабораторная №1 по ЧМ")
window.geometry('1024x720')


zadachi = Combobox(window, state='readonly')
zadachi['values'] = ['Тестовая', 'Основная №1', 'Основная №2']
zadachi.current(0)
zadachi.grid(column = 0, row = 0)



window.mainloop()


btn = Button(window, )


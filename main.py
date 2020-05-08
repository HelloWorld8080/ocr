import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from tkinter import *
import tkinter.filedialog
import a
import pred
from a import cv_show


def xz():
    filename=tkinter.filedialog.askopenfilename()
    print(filename)
    imgs=a.imghandle(filename)
    print("img***",imgs)
    co=0;
    for img in imgs:
        print(img)
        cv_show('img',img)
        co+=1
    print(co)    
    pred.pred(imgs)
def en():
    filename=tkinter.filedialog.askopenfilenames()
    a.enhance(filename)
root = Tk()
root.title('demo')
root.geometry('640x480')
lb = Label(root,text='')
lb.pack()
btn1=Button(root,text='数据增强',command=en)
btn2=Button(root,text='图片识别',command=xz)
btn2.pack()
btn1.pack()
root.mainloop()
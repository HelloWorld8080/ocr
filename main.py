import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import tkinter as tk 
import tkinter.filedialog as tfl
import a
import pred
from a import cv_show
from distributed.worker import weight
from skimage.viewer.utils import canvas
def xz():
    filename=tfl.askopenfilename()
    print(filename)
    imgs=a.imghandle(filename) 
    pred.pred(imgs)
def tr():
    trwin=tk.Toplevel(root)
    trwin.geometry('300x200')
    trwin.title('模型训练')
    label=tk.Label(trwin,text='共计训练4586张图片',bg='green').pack()
    label=tk.Label(trwin,text='精确度：  99.57%',bg='green').pack()
    label=tk.Label(trwin,text='损失率：  1.0203e4',bg='green').pack()
def en():
    filename=tfl.askopenfilenames()
    a.enhance(filename)
root = tk.Tk()
root.title('卡号识别系统')
root.geometry('300x200')
# canvas=tk.Canvas(root,bg='blue',height=800,weight=400)
# image_file=tk.PhotoImage(file='ind')
btn1 = tk.Button(root,text='数据增强',bg='green',width=20,height=2,command=en)
btn2 = tk.Button(root,text='图片识别',bg='green',width=20,height=2,command=xz)
btn3 = tk.Button(root,text='模型建立',bg='green',width=20,height=2,command=tr)
btn1.place(x=60,y=10,anchor='nw')
btn2.place(x=60,y=60,anchor='nw')
btn3.place(x=60,y=110,anchor='nw')
root.mainloop()

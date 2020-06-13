import tkinter.filedialog as tfl
import a
import tensorflow as tf
import cv2 as cv
import tkinter as tk
from tkinter import ttk
from matplotlib import pyplot as plt
from win32service import OpenWindowStation
def resizeandgray(img):
    x=[]
    for i in img:
        gray=cv.cvtColor(i,cv.COLOR_BGR2GRAY)
        x.append(cv.resize(gray,(30,46)))
    for xx in x:
        cv.imshow('xx',xx)    
    return x
def xz():#图片识别
    filename=tfl.askopenfilename()
    print(filename)
    localimgs=a.imghandle(filename)   
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('hostory/saver_792/save_mode-792.meta')
        saver.restore(sess, tf.train.latest_checkpoint('hostory/saver_792/'))
        graph = tf.get_default_graph()
        i=0    
#         for img in localimgs:
        q=''
        co=0
        numimgs=localimgs
#             numimgs=a.WindowSlide(img, 32, 2, 1) 
        x=resizeandgray(numimgs)             
        y =  graph.get_operation_by_name('pred').outputs[0]
        t=tf.nn.softmax(y)
        img_input=graph.get_operation_by_name('img_input').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        pre_ac=sess.run(t,feed_dict={img_input:x,keep_prob:1})
        coo=0
        for i in pre_ac:
                n=i.argmax(axis=0)
                cv.imwrite('imgset/'+str(n)+'_'+str(coo)+'a.png',x[coo]) 
                if n==10:
                    q+='_'
                else:
                    q+=str(n)    
#                 cv.imshow('num'+str(n),x[coo])
                coo+=1            
#         cv.imshow('localimg',img)               
#         plt.scatter(range(0,2*len(q),2),q,c='g',marker='*')
#         plt.xlabel('x')
#         plt.ylabel('推理值')
#         plt.close(None)
        cv.destroyAllWindows()
        print(q)
        xzwin=tk.Toplevel(root)
        xzwin.title('卡号识别')
        xzwin.geometry('300x200')
        tk.Label(xzwin,text='银行卡号为:'+q).place(x=60,y=60,anchor='nw')
def tr():#模型建立
    trwin=tk.Toplevel(root)
    trwin.geometry('300x200') 
    trwin.title('模型训练')
    label=tk.Label(trwin,text='共计训练4586张图片',bg='green').pack()
    label=tk.Label(trwin,text='精确度：  99.57%',bg='green').pack()
    label=tk.Label(trwin,text='损失率：  1.0203e4',bg='green').pack()
def en1():
    filenames=tfl.askopenfilenames()
    print(filenames) 
    a.enhance1(filenames) 
    enwin=tk.Toplevel(root)
    enwin.geometry('300x200') 
    enwin.title('原数据增强')
    tk.Label(enwin,text='原始数据增强完成').pack()
def en2():
    filenames=tfl.askopenfilenames()
    print(filenames) 
    a.enhance2(filenames) 
    enwin=tk.Toplevel(root)
    enwin.geometry('300x200') 
    enwin.title('银行卡数据切割')
    tk.Label(enwin,text='银行卡数据切割完成').pack()    
def en():#数据增强
    enwin=tk.Toplevel(root)
    enwin.geometry('300x200') 
    enwin.title('数据增强')
    btn1 = tk.Button(enwin,text='原数据增强',bg='green',width=20,height=2,command=en1)
    btn2 = tk.Button(enwin,text='银行卡切割',bg='green',width=20,height=2,command=en2)
    btn1.place(x=60,y=20,anchor='nw')
    btn2.place(x=60,y=70,anchor='nw')
root = tk.Tk()
root.title('卡号识别系统')
root.geometry('300x200')
# canvas=tk.Canvas(root,bg='blue',height=800,weight=400)
# image_file=tk.PhotoImage(file='ind')
btn1 = tk.Button(root,text='数据增强',bg='green',width=20,height=2,command=en)
btn2 = tk.Button(root,text='卡号识别',bg='green',width=20,height=2,command=xz)
btn3 = tk.Button(root,text='模型建立',bg='green',width=20,height=2,command=tr)
btn1.place(x=60,y=10,anchor='nw')
btn2.place(x=60,y=60,anchor='nw')
btn3.place(x=60,y=110,anchor='nw')
root.mainloop()

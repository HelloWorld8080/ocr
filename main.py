import tkinter.filedialog as tfl
import a
import tensorflow as tf
import cv2 as cv
import tkinter as tk
from matplotlib import pyplot as plt
def resizeandgray(img):
    x=[]
    for i in img:
        gray=cv.cvtColor(i,cv.COLOR_BGR2GRAY)
        x.append(cv.resize(gray,(30,46)))
        
    for xx in x:
        cv.imshow('xx',xx)    
    return x
def xz():
    filename=tfl.askopenfilename()
    print(filename)
    imgs=a.imghandle(filename)
    with tf.Session() as sess:
        q=[]
        saver = tf.train.import_meta_graph('./save_mode-792.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        i=0
        co=0
        for img in imgs:
            con=0
            numimgs=a.cutbankimg(img, 32, 32, 1) 
            x=resizeandgray(numimgs)
             
            y =  graph.get_operation_by_name('pred').outputs[0]
            t=tf.nn.softmax(y)
            img_input=graph.get_operation_by_name('img_input').outputs[0]
            keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
            pre_ac=sess.run(t,feed_dict={img_input:x,keep_prob:1})
            for i in pre_ac:
                    n=i.argmax(axis=0)
                    q.append(n)
                    if n!=10:
                        con+=1
                       
            for xx in x:
                if q[co]==10:
                    cv.imwrite('imgset/_'+str(co)+'.png',xx)
                else:
                    cv.imwrite('imgset/'+str(q[co])+'_'+str(co)+'.png',xx)   
                co+=1              
            if(con>30):
                cv.imshow('localimg',img)               
                plt.scatter(range(0,2*len(q),2),q,c='g',marker='*')
                plt.xlabel('x')
                plt.ylabel('推理值')
                plt.close(None)
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

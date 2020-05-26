import numpy as np
import tensorflow as tf
import cv2 as cv
from matplotlib import pyplot as plt
import tkinter as tk
def resizeandgray(img):
    x=[]
    for i in img:
        gray=cv.cvtColor(i,cv.COLOR_BGR2GRAY)
        x.append(cv.resize(gray,(30,46)))
        
    for xx in x:
        cv.imshow('xx',xx)    
    return x

def pred(img):
    with tf.Session() as sess:
        q=''
        x=resizeandgray(img)
        saver = tf.train.import_meta_graph('./save_mode-792.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
    
        y =  graph.get_operation_by_name('pred').outputs[0]
        t=tf.nn.softmax(y)
        img_input=graph.get_operation_by_name('img_input').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        pre_ac=sess.run(t,feed_dict={img_input:x,keep_prob:1})
        for i in pre_ac:
                n=i.argmax(axis=0)
                if n!=10:
                    q+=str(n)   
        pwin=tk.Tk()
        pwin.title('字符识别')
        pwin.geometry('300x200')
        tk.Label(pwin,text='识别数字为： '+q,bg='green').pack()
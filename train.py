from os import listdir
import cv2 as cv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from cProfile import label
from matplotlib.pylab import style

#本程序参考了TensorFlow中文官方教程

#parm
learning_rate=0.001 #学习率
training_iters=10 #训练周期s
batch_size=50 #批处理
display_step=10 #迭代次数用于统计精准度
#network
out_class=11
dropout=0.8
img_input=tf.placeholder(tf.float32,[None,46,30],name='img_input')#输入数据
test_input=tf.placeholder(tf.float32,[None,out_class],name='test_input')#训练样本下标
keep_prob=tf.placeholder(tf.float32,name='keep_prob')
def conv2d(x_input,W,b,strides=1):#cnn模型
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_input,W,strides=[1,strides,strides,1],padding='SAME'),b))
def maxpool2d(x_input,k):
        return tf.nn.max_pool(x_input,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')
def convnet(x,weights,biases,dropout):
        x=tf.reshape(x,shape=[-1,46,30,1])
        conv1= conv2d(x,weights['wc1'],biases['bc1'])
        conv2=conv2d(conv1,weights['wc2'],biases['bc2'])
        maxpool1=maxpool2d(conv2,k=2)
        maxpool2=maxpool2d(maxpool1,k=3)
        fullshape=maxpool2.shape.as_list()[1]*maxpool2.shape.as_list()[2]*maxpool2.shape.as_list()[3]
        
        fc=tf.reshape(maxpool2,shape=[-1,fullshape])
        fullweights={
            # full connected,out=1024
            'wd1':tf.Variable(tf.random_normal([fullshape,512]),name='wd1'),#全连接层权重
            # full connected,out=11
            'wd2':tf.Variable(tf.random_normal([512,out_class]),name='wd2')
            }
        fullbiases={
            'bd1':tf.Variable(tf.random_normal([512]),name='bd1'),#全连接层权重
            'bd2':tf.Variable(tf.random_normal([out_class]),name='bd2'),
            }
        fullconnect=tf.add(tf.matmul(fc,fullweights['wd1']),fullbiases['bd1'],name='fullconnect')
        fc=tf.nn.relu(fullconnect)
        fc=tf.nn.dropout(fc,dropout)
        return tf.add(tf.matmul(fc,fullweights['wd2']),fullbiases['bd2'],name='pred')
weights={#权重
        # conv1 5*5*1*32,out=32
        'wc1':tf.Variable(tf.random_normal([5,5,1,32]),name='wc1'),#卷积层权重
        # conv2 5*5*32*64,out=64
        'wc2':tf.Variable(tf.random_normal([5,5,32,64]),name='wc2'),
}
biases={
        # conv1
        'bc1':tf.Variable(tf.random_normal([32]),name='bc1'),
        'bc2':tf.Variable(tf.random_normal([64]),name='bc2'),#卷积层权重
       
}
 
def cutimg(img_value):#切割图片
    x=[]
    img=[]#图片列表
    img_value=cv.cvtColor(img_value,cv.COLOR_BGR2GRAY)
    for i in range(0, img_value.shape[1]//30 ):
        n=i*30
        x=img_value[0:,n:n+30]
        cv.imshow('x',np.array(x))
        img.append(np.array(x))
def img_load(path='imgset'):#加载图片
    labels=[]
    imgs=[]
    img_list=listdir(path)
    for imgname in img_list:
        image=cv.imread(path+imgname)
        imgs.append(image)
        x=np.zeros((out_class))
        if imgname=='_': 
            x[out_class-1]=1
        else:
            x[int(imgname)]=1
        labels.append(x)      
    return imgs,labels    
def String_add(string_list):
    n=0
    for i in string_list:
        string_list[n]='img/'+i
        n+=1
def imgset_cuthandle(path,key_list):
        for imgname in key_list:
            image=cv.imread(path+imgname,cv.COLOR_BGR2GRAY)
            for i in range(0,4):
                n=i*30
                img=image[0:,n:n+30]
                if imgname[i]=='_': 
                    cv.imwrite('imgset/_.png',img) 
                else:
                    cv.imwrite('imgset/'+imgname[i]+'.png',img)        
style.use('ggplot')    
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
path='img/'      
img_list=listdir(path)#获取训练数据
imgset_cuthandle(path,img_list)
imgs,labels=img_load('imgset')#加载数据集
x1=tf.constant(np.array(imgs))
t1=tf.constant(np.array(labels))
dataset=tf.data.Dataset.from_tensor_slices((x1, t1))#建立dataset集
datasets=dataset.shuffle(10).batch(batch_size).repeat(training_iters)
pred=convnet(img_input,weights,biases,keep_prob)#推理
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=test_input),name='cost')#计算损失函数
optimizer= tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)#权重优化
correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(test_input,1),name='correct_prediction')   
accuracy= tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='accuracy')#计算精确度
saver=tf.train.Saver(max_to_keep=1)#保存模型
init=tf.global_variables_initializer()
train_loss=[]
training_acc=[]
test_acc=[]
with tf.Session() as sess:#开始训练
        sess.run(init)
        iterator = datasets.make_initializable_iterator()
        init_op = iterator.make_initializer(datasets)
        step=1
        sess.run(init_op)
        writer=tf.summary.FileWriter("./summary/linear-regression-2/",sess.graph)
        iterator = iterator.get_next()
        try:
                while True:
                        x_out,y_out=sess.run(iterator)
                        sess.run(optimizer,feed_dict={img_input:x_out,test_input:y_out,keep_prob:dropout})
                        if step%display_step==0:
                                x_out,y_out=sess.run(iterator)
                                loss_train,acc_train=sess.run([cost,accuracy],feed_dict={img_input:x_out,test_input:y_out,keep_prob:1})
                                train_loss.append(loss_train)
                                training_acc.append(acc_train)
                        step+=1
        except tf.errors.OutOfRangeError:#训练结束,保存结果并绘制训练情况
                saver.save(sess,"save_mode",global_step=step)
                print(step)
                eval_indices=range(0,step*batch_size,display_step*batch_size)
                plt.scatter(eval_indices[0:len(train_loss)],train_loss,c='b',marker='*',label='softmax损失值散点图')
                plt.xlabel('迭代次数')
                plt.ylabel('损失率')
                plt.close(None)
               
                plt.scatter(eval_indices[0:len(training_acc)],training_acc,c='r',marker='x',label='训练准确率散点图')
                plt.xlabel('迭代次数')
                plt.ylabel('精确度')
                plt.close(None)
writer.close()
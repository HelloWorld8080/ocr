import numpy as np
import cv2 as cv
import math
import tensorflow as tf
import random
from os import listdir
from matplotlib import pyplot as plt
from scipy import ndimage
from cv2.cv2 import morphologyEx, MORPH_CLOSE, MORPH_OPEN, MORPH_TOPHAT, dilate
from skimage.feature._canny import canny
def cv_show(name,img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
def enhance1(load):#数据增强模块
    with tf.Session() as sess:
        for i in load:
            for s in range(0,20):
                raw_img = tf.gfile.FastGFile(i,'rb').read()
                n=random.randint(0,11)
                img_data = tf.image.decode_image(raw_img)
                if n==0:         # 
                    img_data=np.rot90(sess.run(img_data))
                    strload=i[0:i.find('.',-5,-1)-1]+'_'+str(s)+str(n)+'.png'
                elif n==1:
                    img_data = tf.image.rgb_to_grayscale(img_data)
                elif n==2:
                    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
                    img_data = tf.image.adjust_brightness(img_data, delta=-.7)
                elif n==3:
                    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
                    img_data = tf.image.random_brightness(img_data, max_delta=0.6)
                elif n==4:
                    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
                    img_data = tf.image.random_contrast(img_data, lower=0, upper=4)
                elif n==5:
                    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
                    img_data = tf.image.random_hue(img_data, 0.5)
                elif n==6:
                    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
                    img_data = tf.image.random_saturation(img_data, lower=0, upper=2)
                elif n==7:
                    img_data = tf.image.central_crop(sess.run(img_data),random.random())
                elif n==8:
                    img_data = tf.image.resize_image_with_pad(img_data,random.randint(sess.run(tf.shape(img_data))[0]/2,sess.run(tf.shape(img_data))[0]*2),random.randint(sess.run(tf.shape(img_data))[1]/2,sess.run(tf.shape(img_data))[1]*2))
                elif n==9:
                    img_data = tf.image.flip_left_right(img_data) 
                elif n== 10:
                    img_data = tf.image.flip_up_down(img_data)
                img_data = tf.image.convert_image_dtype(img_data, tf.int16)
                strload=i[0:i.find('.',-5,-1)-1]+'_'+str(s)+str(n)+'.png'
                cv.imwrite(strload,img_data.eval())
def enhance2(filenames):
    co=0
    for filename in  filenames:
        cardimg=cv.imread(filename)
        rawimgs=WindowSlide(cardimg, 46, 46, 0)
        for rawimg in rawimgs:
            colimgs=WindowSlide(rawimg,30, 30, 1)
            for colimg in colimgs:
                gray=cv.cvtColor(colimg,cv.COLOR_BGR2GRAY)
                cv.imwrite('imgset/'+'_'+str(co)+'.png', gray)
                co+=1                
def cutimg(img_value,ROI_w,ROI_h,ROI_x,ROI_y,type):#裁剪图片
    img=[]
    t=0
    for i in range(0,math.ceil(ROI_w/25)):
        if type!=3 and i%4==0 and i>0:
                t+=10
        n=i*25+t    
        x=np.zeros((ROI_h,25,img_value.shape[2]),dtype=np.int16)
        for j in range(0,ROI_h):
            if ROI_w-n<25:
                return img
            else :
                x[j][0:]=img_value[ROI_y+j][n+ROI_x:n+ROI_x+25]
#         cv_show('x', x)                 
        img.append(x)
    return img

def scan(image):
    def order_points(pts):
        # 一共4个坐标点
        rect = np.zeros((4, 2), dtype = "float32")
        
        # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
        # 计算左上，右下
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
    
        # 计算右上和左下
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(image, pts):
        # 获取输入坐标点
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
    
        # 计算输入的w和h值
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
    
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
    
        # 变换后对应坐标位置
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
    
        # 计算变换矩阵
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
        # 返回变换后结果
        return warped
        #坐标也会相同变化
    image = cv.resize(image, (680,500), interpolation=cv.INTER_AREA)
    orig = image.copy()
    # 预处理
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(gray, 70, 100)
    kernel = np.ones((3,3), np.uint8)
    close = cv.morphologyEx(edged,MORPH_CLOSE,kernel)
    # 展示预处理结果
    print("STEP 1: 边缘检测")
    cv.imshow("Image", image)
    cv.imshow("Edged", edged)
    cv.imshow("close", close)
    # 轮廓检测
    cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[1]
    cv.drawContours(image, cnts, -1, (0, 255, 0), 2)
    cv.imshow('imagecon',image)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]
    screenCnt=[]
    # 遍历轮廓
    for c in cnts:
        # 计算轮廓近似
        peri = cv.arcLength(c, True)
        # C表示输入的点集
        # epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
        # True表示封闭的
        approx = cv.approxPolyDP(c, 0.01 * peri, True)
        cv.drawContours(orig, [approx], -1, (0, 100, 200), 2)
        cv.imshow('approx',orig)
        # 4个点的时候就拿出来
        if len(approx) == 4:
            screenCnt = approx
            break
    
    # 展示结果
    w=0
    h=0
    if(len(screenCnt)==4):
       x, y, w, h = cv.boundingRect(screenCnt)    
    if w>500 and h>300 :
        print("STEP 2: 获取轮廓")
        cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        cv.imshow("Outline", image)
        
        # 透视变换
        warped = four_point_transform(orig, screenCnt.reshape(4,2))
        cv.imshow('warped',warped)
        return warped
    return []
def ts(e):
        return e[0]
def CardNumLocal(orimg,type):
    # 添加矩形框元素
    def add_cont(x, y, w ,h):
        p = []
        p.append(x)
        p.append(y)
        p.append(w)
        p.append(h)
        return p
    def bubble_sort(a, w, s):
        '''w: 要取的x,y,w,h元素，对应0，1，2，3'''
        '''s: 0取最小值， 1取最大值'''
        b = []
        temp = 0
        for i in range(len(a)):
            b.append(a[i][w])
        b.sort()
        if s:
            return b[len(b)-1]
        else:
            return b[0]
    def cutimg1(img_value,type):
        handle=[]
        if type==0:
            th=34
            co=0
            for i in range(0,4):
                t=7+th
                n=i*(th*4+t)+10
                if img_value.shape[1]-n<4*th:
                    return handle
                cutiimg=img_value[0:, n:n+4*th] 
                cv.imshow('cutiimg',cutiimg)
                for j in range(0,4):
                    n=j*th
                    cutjimg=cutiimg[0:, n:n+th]
                    cv.imshow('cutjimg',cutjimg)
                    handle.append(cutjimg)    
        return handle
    locanimgs=[]     
    tent = 1
    point = []
    target = []
    img=orimg.copy()
    cv.imshow('img',img)
    kernel3 = np.ones((3, 3), np.uint8)
    kernel4 = np.ones((4, 4), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    kernel9 = np.ones((9, 9), np.uint8)
    if type==0:
        gray = cv.GaussianBlur(img, (5, 5), 0)
        newimg = cv.Canny(gray, 70, 100)
        cv.imshow('newimg',newimg)
        dst0_9=morphologyEx(newimg,MORPH_CLOSE,kernel5)
        dst0_9=morphologyEx(dst0_9,MORPH_OPEN,kernel3)
        cv.imshow('dst0_9',dst0_9)
        contours = cv.findContours(dst0_9, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]
        for cnt in contours:
            x,y,w,h=cv.boundingRect(cnt)
            if 30>w>20 and 35>h>25:
                point.append((x,y,w,h))        
        point.sort(key=ts)        
        for rect in point:
            x,y,w,h=rect[0],rect[1],rect[2],rect[3]         
            locanimg=orimg[y-3:y+h+3,x-3:x+w+3]
            locanimgs.append(locanimg)
            
        for locat in locanimgs:
            cv.imshow('locat',locat)        
        finalx = bubble_sort(point, 0, 1)
        startx = bubble_sort(point, 0, 0) - 3
        starty = bubble_sort(point, 1, 0) - 3
        finalx = finalx + bubble_sort(point, 2, 1) + 3
        finaly = starty + bubble_sort(point, 3, 1) + 3    
        return locanimgs,startx,starty,finalx,finaly
    elif type==1:   
        newimg = cv.Canny(img, 70, 100)
        cv.imshow('newimg',newimg)
        dst1=morphologyEx(newimg,MORPH_CLOSE,kernel5)
        cv.imshow('dst1', dst1)
        dst2=morphologyEx(dst1,MORPH_OPEN,kernel3)
        cv.imshow('dst2', dst2)
        contours = cv.findContours(dst2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv.boundingRect(cnt)
            if 40>w>15 and 50>h>15:
                point.append(add_cont(x,y,w,h))
                cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv.imshow('imggg',img)
        imggg1=orimg.copy()
                
        for o in range(len(point)):
            for i in range(len(point)):
                if i != o:
                    xx= abs(point[o][1] - point[i][1])
                    if 0<xx<=10:
                        tent += 1
                    if  tent >10 :
                        tent = 1
                        target.append(point[o])
                        x,y,w,h=point[o][0],point[o][1],point[o][2],point[o][3]
                        cv.rectangle(imggg1,(x,y),(x+w,y+h),(255,0,0),2)
                        cv.imshow('imggg1',imggg1)              
        finalx = bubble_sort(point, 0, 1)
        startx = bubble_sort(point, 0, 0) - 3
        starty = bubble_sort(point, 1, 0) - 3
        finalx = finalx + bubble_sort(point, 2, 1)
        finaly = starty + bubble_sort(point, 3, 1) + 8
        lcan_dst2=dst2[starty:finaly, startx:finalx]
         
        cv.imshow('lcan_dst2',lcan_dst2)
        
        
        t_lcan_dst2=morphologyEx(lcan_dst2,MORPH_TOPHAT,kernel4)
        cv.imshow('b_lcan_dst2', t_lcan_dst2)
        da_dst2=dilate(lcan_dst2-t_lcan_dst2,kernel5)
        cv.imshow('da_dst2',da_dst2)
        lcanimg=orimg[starty:finaly, startx-10:finalx+10]
        cv.imshow('lcanimg',lcanimg)
        locanimgs=cutimg1(lcanimg, 0)
        for lli in locanimgs:
            cv.imshow('lli',lli)       
        return locanimgs,startx,starty,finalx,finaly
def getlight(grayimg):
    light=0
    cout=0   
    for imge in grayimg:
        for imgee in imge:
            light+=imgee
            cout+=1
    imglight=light//cout
    return imglight

def line_detect_possible(orimage):
    image=orimage.copy()
    th=100
    image = cv.resize(image, (680,500), interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 150)
    cv.imshow("edges", edges)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, 100, minLineLength =250, maxLineGap = 30)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (abs(x1-x2)<20 or abs(y1-y2)<20):
            cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.imshow("line_detect_possible", image)
    cv.imshow("line_detect_possible", image)

def WindowSlide(img,th,step,style):#th切割大小，step切割步长，style切割方向：0为向下切割。1为向右切割
    imgs=[]
    n=0
    if style==0:
        while True:
            if n+th>img.shape[0]:
                return imgs
            else:
                cutimg=img[n:n+th,0:]
                imgs.append(cutimg)
            n+=step    
    elif style==1:
        while True:
            if n+th>img.shape[1]:
                return imgs
            else:
                cutimg=img[0:,n:n+th]
                imgs.append(cutimg)
            n+=step
def CardNumLocal2(img):
    # 添加矩形框元素
    def add_cont(x, y, w ,h):
        p = []
        p.append(x)
        p.append(y)
        p.append(w)
        p.append(h)
        return p
    # 起泡法排序返回最大or最小值
    def bubble_sort(a, w, s):
        '''w: 要取的x,y,w,h元素，对应0，1，2，3'''
        '''s: 0取最小值， 1取最大值'''
        b = []
        temp = 0
        for i in range(len(a)):
            b.append(a[i][w])
        b.sort()
        if s:
            return b[len(b)-1]
        else:
            return b[0]

    tent = 0
    startx = 0
    finalx = 0
    finaly = 0
    finalw = 0
    finalh = 0
    point = []
    target = []
    img0 = img
    img0 = cv.resize(img0, (800,400), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    cv.namedWindow("org")
    kernel = np.ones((3, 3), np.uint8)
    newimg = cv.Canny(img, 170, 100)
    newimg = cv.morphologyEx(newimg, cv.MORPH_CLOSE, kernel)
    image_contours, contours, hier = cv.findContours(newimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv.boundingRect(cnt)
        if 50>w>10 and 50>h>10:
            point.append(add_cont(x,y,w,h))
    for o in range(len(point)):
        for i in range(len(point)):
            if 0 < abs(point[o][1] - point[i][1]) < 5:
                tent += 1
            elif abs(point[o][1] - point[i][1]) == 0:
                if point[o][0] != point[i][0]:
                    tent += 1
            if tent > 6:
                tent = 0
                target.append(point[o])
    newimg[0:target[0][1]-3,0:800] = 0
    newimg[target[0][1]+target[0][3]+3:400, 0:800] = 0
    point = []
    target = []
    image_contours, contours, hier = cv.findContours(newimg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv.boundingRect(cnt)
        if 50>w>3 and 50>h>3:
            point.append(add_cont(x,y,w,h))
    finalx = bubble_sort(point, 0, 1)
    startx = bubble_sort(point, 0, 0) - 3
    finaly = bubble_sort(point, 1, 0) - 3
    finalw = finalx + bubble_sort(point, 2, 1)
    finalh = finaly + bubble_sort(point, 3, 1) + 10
    print("startx = "+ str(startx))
    print("starty = "+ str(finaly))
    print("width = "+ str(finalw))
    print("height = "+ str(finalh))
    # 画出矩形框
    cv.rectangle(img0,(startx,finaly),(finalw, finalh),(0, 255, 0),2)
    
    cv.imshow("org", img0)
    
    cv.waitKey()
    cv.destroyAllWindows()
def CardNumLocal3(img):
     # 添加矩形框元素
    def add_cont(x, y, w ,h):
        p = []
        p.append(x)
        p.append(y)
        p.append(w)
        p.append(h)
        return p
    # 起泡法排序返回最大or最小值
    def bubble_sort(a, w, s):
        '''w: 要取的x,y,w,h元素，对应0，1，2，3'''
        '''s: 0取最小值， 1取最大值'''
        b = []
        temp = 0
        for i in range(len(a)):
            b.append(a[i][w])
        b.sort()
        if s:
            return b[len(b)-1]
        else:
            return b[0]
    localimgs=[]    
    orgimg=img.copy() 
   
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)   
    gauss = cv.GaussianBlur(gray, (5, 5), 0)    
    canny=cv.Canny(gauss,70,100)
    cv.imshow('canny',canny)
    point=[]
    contours = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]
    coimg=orgimg.copy()           
    for i in range(len(contours)):
        x,y,w,h = cv.boundingRect(contours[i])
        if 35>w>10 and 40>h>10 and 150<y<250:
            point.append(add_cont(x,y,w,h))
            cv.rectangle(coimg,(x,y),(x+w,y+h),(255,0,0),2)
            cv.imshow('coimg',coimg)
    point.sort(key=ts)
    target=[]
    for o in range(len(point)):
        tent=0
        for i in range(len(point)):
            if 0 < abs(point[o][1] - point[i][1]) < 5:
                tent += 1
            elif abs(point[o][1] - point[i][1]) == 0:
                if point[o][0] != point[i][0]:
                    tent += 1
            if tent > 12:
                tent = 0
                target.append(point[o])  
    loc=orgimg.copy()
    target.sort(key=ts)
    coimg1=orgimg.copy()
    for x,y,w,h in target:
        cv.rectangle(coimg1,(x,y),(x+w,y+h),(255,0,0),2)
        cv.imshow('coimg1',coimg1)
    startx=target[0][0]-3
    starty=target[0][1]-3
    finalx=target[-1][0]+target[-1][2]+3
    finaly=target[-1][1]+target[-1][3]+3
    for x,y,w,h in target:
        localimg=loc[y-3:y+h+3,x-3:x+w+3]
        cv.imshow('localimg',localimg)
        localimgs.append(localimg)
    return localimgs,startx,starty,finalx,finaly        
def imghandle(img_name):#图片处理
#     handle=[]
    orgimg = cv.imread(img_name)
#     line_detect_possible(orgimg)
    cv.imshow('orgimg',orgimg)
    imgout=scan(orgimg)
    localimgs=[]
    if len(imgout)==0:
        imgout=orgimg.copy()
        imgout = cv.resize(imgout, (600,400), interpolation=cv.INTER_AREA)
        localimgs,startx,starty,finalx,finaly=CardNumLocal3(imgout.copy())
    else:    
        imgout = cv.resize(imgout, (800,400), interpolation=cv.INTER_AREA)
        localimgs,startx,starty,finalx,finaly=CardNumLocal(imgout.copy(),1)#卡号定位处理  
#     radio=imgout.shape[0]/imgout.shape[1]
#     img=cv.resize(imgout,( 600,int(600*radio)))
#     cv.imshow('img',img)
#     hei=img.shape[0]
#     img1=img[hei//2-10:hei//2+90,0:]
#     cv.imshow('img1',img1)    
#     localimgs=WindowSlide(img1, 52, 5,0)
#     for img in localimgs:
#         cv.imshow('cutimg',img)
#     return localimgs    
#     if imgout=='noscan':
#         imgout=orgimg.copy()
#         imgout = cv.resize(imgout, (800,400), interpolation=cv.INTER_AREA)
#         localimgs,startx,starty,finalx,finaly=CardNumLocal(imgout.copy(),0)#卡号定位处理
#     else:
#         imgout = cv.resize(imgout, (800,400), interpolation=cv.INTER_AREA)
#         localimgs,startx,starty,finalx,finaly=CardNumLocal(imgout.copy(),1)#卡号定位处理
    cv.rectangle(imgout, (startx,starty), (finalx,finaly), (255,0,0), 2)
    cv.destroyAllWindows()
    plt.imshow(imgout)
    plt.show()
    return  localimgs   
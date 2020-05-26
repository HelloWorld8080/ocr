

'''
Created on 2020年5月20日

@author: 刘循
'''
import os
import cv2 as cv
import numpy as np
import tensorflow as tf





filename='imgg/_081b_0.png' 
img =cv.imread(filename)
org=img.copy()
cv.imshow('imgg',org)
for i in range(0,4):
    n=i*30
    
    cv.imshow('img',img[0:,n+30])
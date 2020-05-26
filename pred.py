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
   
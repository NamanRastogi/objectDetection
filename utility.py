
# coding: utf-8

# In[3]:


import argparse
import cv2
import numpy as np
import tensorflow as tf
from collections import namedtuple


# In[4]:

# initialize variables
def init_vars(sess):
    vars = []
    tensors = []
    for var in tf.global_variables():
        vars.append(var)
        tensors.append(tf.is_variable_initialized(var))
    bools = sess.run(tensors)
    init = zip(bools, vars)
    init = [var for i, var in init if not i]
    sess.run(tf.variables_initializer(init))

# openCV stores as bgr
def rgb2bgr(vec):
    return (vec[2], vec[1], vec[0])

# In[5]:


Label = namedtuple('Label', ['name', 'color'])
Size = namedtuple('Size', ['w', 'h'])
Point = namedtuple('Point', ['x', 'y'])
Sample = namedtuple('Sample', ['filename', 'boxes', 'img_size'])
Box = namedtuple('Box', ['label', 'labelid', 'center', 'size'])
Score = namedtuple('Score', ['idx', 'score'])
Overlap = namedtuple('Overlap', ['best', 'good'])


# In[7]:
      
# min-max box bound to center-width bound
def mm2cw(x_min, x_max, y_min, y_max, img_size):
    width = float(x_max-x_min)
    height = float(y_max-y_min)
    center_x = float(x_min) +  width/2
    center_y = float(y_min) + height/2
    
    width /= img_size.w
    height /= img_size.h
    center_x /= img_size.w
    center_y /= img_size.h
    return Point(center_x, center_y), Size(width, height)

def cw2mm(center, size, img_size):
    width = size.w * img_size.w/2
    height = size.h * img_size.h/2
    center_x = center.x * img_size.w
    center_y = center.y * img_size.h
    return int(center_x-width), int(center_x+width), int(center_y-height), int(center_y+height)

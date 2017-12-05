
# coding: utf-8

# In[12]:


import numpy as np
import os, sys
from utility import *
from collections import namedtuple
from math import sqrt, log, exp


# In[5]:

SSD_conf = namedtuple('SSD_conf', ['image_size', 'num_maps', 'map_sizes', 'num_anchors'])
SSD_confs = {
    'vgg300': SSD_conf(
        image_size = Size(300, 300),
        num_maps = 6,
        map_sizes = [
            Size(38, 38),
            Size(19, 19),
            Size(10, 10),
            Size(5, 5),
            Size(3, 3),
            Size(1, 1)
        ],
        num_anchors = 11639
    )
}

# min and max scales for default boxes
SCALE_MIN = 0.2
SCALE_MAX = 0.9
SCALE_DIFF = SCALE_MAX - SCALE_MIN

Anchor = namedtuple('Anchor', ['center', 'size', 'x', 'y', 'scale', 'map'])


# In[13]:


def get_conf(name):
    return SSD_confs[name]

# compute default anchor boxes
def get_anchors(conf):
    scales = []
    for k in range(1, conf.num_maps+1):
        scale = SCALE_MIN + SCALE_DIFF/(conf.num_maps-1)*(k-1)
        scales.append(scale)
    
    # compute the width and height of anchor boxes for every scale
    ratios = [1, 2, 3, 1/2, 1/3]
    ratios = list(map(lambda x: sqrt(x), aspect_ratios))
    
    box_sizes = {}
    for i in range(len(scales)):
        s = scales[i]
        box_sizes[s] = []
        for ratio in ratios:
            w = s * ratio
            h = s / ratio
            box_sizes[s].append((w,h))
        if i < len(scales)-1:
            s_prime = sqrt(scales[i]*scales[i+1])
            box_sizes[s].append((s_prime, s_prime))
    
    # compute actual boxes for every scale
    anchors = []
    for k in range(len(scales)):
        s = scales[k]
        fk = conf.map_sizes[k][0]
        for i in range(fk):
            x = (i+0.5)/float(fk)
            for j in range(fk):
                y = (j+0.5)/float(fk)
                for size in box_sizes[s]:
                    box = Anchor(Point(x, y), Size(size[0], size[1]), j, i, s, k)
                    anchors.append(box)
    
    return anchors

def jaccard_overlap(params1, params2):
    x_min1, x_max1, y_min1, y_max1 = params1
    x_min2, x_max2, y_min2, y_max2 = params2
    
    if x_max2 <= x_min1: return 0
    if x_max1 <= x_min2: return 0
    if y_max2 <= y_min1: return 0
    if y_max1 <= y_min2: return 0
    
    x_min = max(x_min1, x_min2)
    x_max = min(x_max1, x_max2)
    y_min = max(y_min1, y_min2)
    y_max = min(y_max1, y_max2)
    
    w = x_max - x_min
    h = y_max - y_min
    intersection = float(w*h)
    
    w1 = x_max1 - x_min1
    h1 = y_max1-y_min1
    w2 = x_max2-x_min2
    h2 = y_max2-y_min2
 
    union = float(w1*h1) + float(w2*h2) - intersection
 
    return intersection/union

def get_overlap(box, anchors, threshold):
    img_size = Size(1000, 1000)
    box_params = cw2mm(box.center, box.size, img_size)
    best = None
    good = []
    for i in range(len(anchors)):
        anchor = anchors[i]
        anchor_params = cw2mm(anchor.center, anchor.size, img_size)
        dist = jaccard_overlap(box_params, anchor_params)
        if dist == 0:
            continue
        elif not best or best.score < dist:
            best = Score(i, dist)
        
        if dist > threshold:
            good.append(Score(i, dist))
            
    return Overlap(best, good)

def get_enc_location(box, anchor):
    arr = np.zeros((4))
    arr[0] = (box.center.x-anchor.center.x)/anchor.size.w
    arr[1] = (box.center.y-anchor.center.y)/anchor.size.h
    arr[2] = log(box.size.w/anchor.size.w)
    arr[3] = log(box.size.h/anchor.size.h)
    
    return arr

def get_location(box, anchor):
    x = box[0] * anchor.size.w + anchor.center.x
    y = box[1] * anchor.size.h + anchor.center.y
    w = exp(box[2]) * anchor.size.w
    h = exp(box[3]) * anchor.size.h
    return Point(x, y), Size(w, h)

def get_boxes(pred, anchors, lid2name={}):
    num_classes = pred.shape[1] - 4
    bg_class = num_classes - 1
    box_class = np.argmax(pred[:, :num_classes], axis=1)
    detections = np.nonzero(box_class != bg_class)[0]
    
    boxes = []
    for idx in detections:
        center, size = get_location(pred[idx, num_classes:], anchors[idx])
        cid = box_class[idx]
        cname = None
        if cid in lid2name:
            cname = lid2name[cid]
        boxes.append(Box(cname, cid, center, size))
        
    return boxes
    
    


# In[ ]:





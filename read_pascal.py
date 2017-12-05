
# coding: utf-8

# In[6]:


import lxml.etree
import random
import math
import cv2
import os
import numpy as np 
import sys
from utility import *
from glob import glob
from tqdm import tqdm


# In[7]:


labels = [
    Label('aeroplane', rgb2bgr((0,0,0))),
    Label('bicycle', rgb2bgr((111,  74,   0))),
    Label('bird', rgb2bgr(( 81,   0,  81))),
    Label('boat', rgb2bgr((128,  64, 128))),
    Label('bottle', rgb2bgr((244,  35, 232))),
    Label('bus', rgb2bgr((230, 150, 140))),
    Label('car', rgb2bgr(( 70,  70,  70))),
    Label('cat', rgb2bgr((102, 102, 156))),
    Label('chair', rgb2bgr((190, 153, 153))),
    Label('cow', rgb2bgr((150, 120,  90))),
    Label('diningtable', rgb2bgr((153, 153, 153))),
    Label('dog', rgb2bgr((250, 170,  30))),
    Label('horse', rgb2bgr((220, 220,   0))),
    Label('motorbike', rgb2bgr((107, 142,  35))),
    Label('person', rgb2bgr((152, 251, 152))),
    Label('pottedplant', rgb2bgr(( 70, 130, 180))),
    Label('sheep', rgb2bgr((220,  20,  60))),
    Label('sofa', rgb2bgr((  0,   0, 142))),
    Label('train', rgb2bgr((  0,   0, 230))),
    Label('tvmonitor', rgb2bgr((119,  11,  32)))
]


# In[10]:


class Data:
    def __init__(self, id='VOC2007'):
        self.num_classes = len(labels)
        self.colors = {l.name: l.color for l in labels}
        self.label_id2name = {i: l.name for i, l in enumerate(labels)}
        self.name2label_id = {l.name: i for i, l in enumerate(labels)}
        self.num_train = 0
        self.num_valid = 0
        self.num_test = 0
        self.train_data = []
        self.valid_data = []
        self.test_data = []
        self.id = id

		def get_file_list(root, dataset_name):
				img_root = root + '/JPEGImages/'
				label_root = root + '/Annotations/'
				label_files = glob(label_root + '/*xml')
				data_points = []
				
				# process data
				for fn in tqdm(label_files, desc=dataset_name, unit='data_points'):
				    with open(fn, 'r') as f:
				        # read filename from annotated xml
				        doc = lxml.etree.parse(f)
				        filename = img_root + doc.xpath('/annotation/filename')[0].text
				        
				        # read image 
				        if not os.path.exists(filename):
				            continue
				        img = cv2.imread(filename)
				        img_size = Size(img.shape[0], img.shape[1])
				        
				        # get boxes for all objects
				        boxes = []
				        objects = doc.xpath('/annotation/object')
				        for obj in objects:
				            # skip difficult
				            difficult = obj.xpath('difficult')
				            if difficult:
				                difficult = int(difficult[0].text)
				                if difficult:
				                    continue
				            
				            label = obj.xpath('name')[0].text
				            x_min = int(float(obj.xpath('bndbox/xmin')[0].text))
				            x_max = int(float(obj.xpath('bndbox/xmax')[0].text))
				            y_min = int(float(obj.xpath('bndbox/ymin')[0].text))
				            y_max = int(float(obj.xpath('bndbox/ymax')[0].text))
				            center, size = mm2cw(x_min, x_max, y_min, y_max, img_size)
				            box = Box(label, center, size)
				            boxes.append(box)
				            
				            # skip data points without labels
				            if not boxes:
				                continue
				            data_point = Sample(filename, boxes, img_size)
				            data_points.append(data_point)
				            
				return data_points
        
    def load_data(self, data_dir):
        train_root = data_dir + '/train/'+self.id
        test_root = data_dir + '/test/'+self.id
        
        train_data = self.get_file_list(train_root, 'train')
        random.shuffle(train_data)
        train_len = len(train_data)
        self.train_data = self.get_file_list(test_root, 'test')
        self.valid_data = train_data[:int(0.1*train_len)]
        self.train_data = train_data[int(0.1*train_len):]
        self.num_train = len(self.train_data)
        self.num_valid = len(self.valid_data)
        
def get_data():
    return Data()


# In[ ]:





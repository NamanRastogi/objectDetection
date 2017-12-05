
# coding: utf-8

# In[4]:


import argparse
import sys
import cv2
import os
import pickle
import numpy as np
from utility import *
from tqdm import tqdm
from ssd_funcs import *


# In[5]:


# draw bounding box on images
def draw_box(data_dir, data, colors, name):
    result_dir = data_dir+'_annotated/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
      
    for pt in tqdm(data, desc=name, unit='points'):
        img = cv2.imread(pt.filename)
        basefn = os.path.basename(pt.filename)
        for box in pt.boxes:
            x_min, x_max, y_min, y_max = cw2mm(box.center, box.size, sample.img_size)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), colors[box.label], 2)
            cv2.rectangle(img, (x_min-1, y_min), (x_max+1, y_min-20), colors[box.label], cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, box.label, (x_min+5, y_min-5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(result_dir+basefn, img)

# process output vector
def set_vec(overlap, box, anchor, matches, num_classes, vec):
    matches[overlap.idx] = overlap.score
    vec[overlap.idx, 0:num_classes+1] = 0
    vec[overlap.idx, box.labelid] = 1
    vec[overlap.idx, num_classes+1:] = get_location(box, anchor)

# input to loss function 
def write_train_data(data_dir, data, anchors, num_classes, name):
    result_dir = data_dir+'/gt/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # dim of target
    row = len(anchors)
    col = num_classes + 5
    
    # generate target for each point
    data_list = []
    for pt in tqdm(data, desc=name, unit='points'):
        vec = np.zeros((row, col), dtype=np.float32)
        overlaps = {}
        for box in sample.boxes:
            overlaps[box] = get_overlap(box, anchors, 0.5)
        
        vec[:, num_classes] = 1
        vec[:, num_classes+1] = 0
        vec[:, num_classes+2] = 0
        vec[:, num_classes+3] = 0
        vec[:, num_classes+4] = 0
        
        matches = {}
        for box in pt.boxes:
            for overlap in overlaps[box].good:
                anchor = anchors[overlap.idx]
                set_vec(overlap, box, anchor, matches, num_classes, vec)
                
        matches = {}
        for box in pt.boxes:
            overlap = overlaps[box].best
            anchor = anchors[overlap.idx]
            set_vec(overlap, box, anchor, matches, num_classes, vec)
            
        data_list.append((pt, gt_fn))
    
    # write data
    with open(data_dir + '/' + name.strip() + '-data.pkl', 'wb') as f:
        pickle.dump(data_list, f)


# In[6]:


def main():
    parser = argparse.ArgumentParser(description='Train the SSD')
    parser.add_argument('--data-dir', default='data_dir', help='data directory')
    args = parser.parse_args()
    
    data = get_file_list(args.data_source)
    data.load_data(args.data_dir, 0.1)
    
    annotate(args.data_dir, data.train_samples, data.colors, 'train')
    annotate(args.data_dir, data.valid_samples, data.colors, 'valid')
    annotate(args.data_dir, data.test_samples,  data.colors, 'test ')
    
    conf = get_conf('SSD_conf')
    anchors = get_anchors(conf)
    write_train_data(args.data_dir, data.train_samples, anchors, data.num_classes, 'train')
    
    with open(args.data_dir+'/training-data.pkl', 'wb') as f:
        data = {
            'conf': conf,
            'num-classes': data.num_classes,
            'colors': data.colors,
            'lable_id2name': data.lable_id2name,
            'name2label_id': data.name2label_id
        }
        pickle.dump(data, f)
    
    return 0


# In[11]:


if __name__ == '__main__':
    sys.exit(main())


# In[ ]:





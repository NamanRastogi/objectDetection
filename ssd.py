
# coding: utf-8

# In[3]:


import zipfile
import shutil
import os
import tensorflow as tf
from urllib.request import urlretrieve
from tqdm import tqdm


# In[4]:




def conv(x, size, shape, stride, name, padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('weights',
                           shape=[shape, shape, x.get_shape()[3], size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(size), name='biases')
        x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def layer(x, size, kernel_size, name):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', 
                           shape=[3, 3, x.get_shape()[3], size],
                           initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(size), name='biases')
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
    return tf.reshape(x, [-1, kernel_size.w*kernel_size.h, size])

def smooth_l1_loss(x):
    square_loss = 0.5*x**2
    absolute_loss = tf.abs(x)
    return tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss-0.5)


# In[11]:


class SSD:
    def __init__(self, session):
        self.session = session
        
    def create_from_vgg(self, vgg_dir, num_classes, conf):
        self.num_classes = num_classes + 1
        self.num_vars = num_classes + 5
        self.preset = preset
        self.load_vgg(vgg_dir)
        self.create_vgg()
        self.create_ssd()
        self.get_feature_maps()
        self.create_classifiers()
    
    def create_from_metagraph(self, metagraph_file, checkpoint_file):
        sess = self.session
        saver = tf.train.import_meta_graph(metagraph_file)
        saver.restore(sess, checkpoint_file)
        self.image_input = sess.graph.get_tensor_by_name('image_input:0')
        self.keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        self.result = sess.graph.get_tensor_by_name('result/result:0')
    
    
    def load_vgg(self, vgg_dir):
        sess = self.session
        graph = tf.saved_model.loader.load(sess, ['vgg16'], vgg_dir+'/vgg')
        self.image_input = sess.graph.get_tensor_by_name('image_input:0')
        self.keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        self.vgg_conv4_3 = sess.graph.get_tensor_by_name('conv4_3/Relu:0')
        self.vgg_conv5_3 = sess.graph.get_tensor_by_name('conv5_3/Relu:0')
        self.vgg_fc6_w = sess.graph.get_tensor_by_name('fc6/weights:0')
        self.vgg_fc6_b = sess.graph.get_tensor_by_name('fc6/biases:0')
        self.vgg_fc7_w = sess.graph.get_tensor_by_name('fc7/weights:0')
        self.vgg_fc7_b = sess.graph.get_tensor_by_name('fc7/biases:0')
        
    def create_vgg(self):
        self.mod_pool5 = tf.nn.max_pool(self.vgg_conv5_3, ksize=[1, 3, 3, 1],
                                       strides=[1,1,1,1], padding='SAME',
                                       name='mod_pool5')
        
        with tf.variable_scope('mod_conv6'):
            x = tf.nn.conv2d(self.mod_pool5, self.vgg_fc6_w, 
                            strides=[1,1,1,1], padding='SAME')
            x = tf.nn.bias_add(x, self.vgg_fc6_b)
            self.mod_conv6 = tf.nn.relu(x)
            
        with tf.variable_scope('mod_conv7'):
            x = tf.nn.conv2d(self.mod_conv6, self.vgg_fc7_w, 
                            strides=[1,1,1,1], padding='SAME')
            x = tf.nn.bias_add(x, self.vgg_fc7_b)
            self.mod_conv7 = tf.nn.relu(x)
                            
    def create_ssd(self):
        self.ssd_conv8_1 = conv_map(self.mod_conv7, 256, 1, 1, 'conv8_1')
        self.ssd_conv8_2 = conv_map(self.ssd_conv8_1, 512, 3, 2, 'conv8_2')
        self.ssd_conv9_1 = conv_map(self.ssd_conv8_2, 128, 1, 1, 'conv9_1')
        self.ssd_conv9_2 = conv_map(self.ssd_conv9_1, 256, 3, 2, 'conv9_2')
        self.ssd_conv10_1 = conv_map(self.ssd_conv9_2, 128, 1, 1, 'conv10_1')
        self.ssd_conv10_2 = conv_map(self.ssd_conv10_1, 256, 3, 1, 'conv10_2', 'VALID')
        self.ssd_conv11_1 = conv_map(self.ssd_conv10_2, 128, 1, 1, 'conv11_1')
        self.ssd_conv11_2 = conv_map(self.ssd_conv11_1, 256, 3, 1, 'conv11_2', 'VALID')
        
    def get_feature_maps(self):
        self.maps = [
            self.vgg_conv4_3,
            self.mod_conv7,
            self.ssd_conv8_2,
            self.ssd_conv9_2,
            self.ssd_conv10_2,
            self.ssd_conv11_2
        ]
        
    def create_classifiers(self):
        self.classifiers = []
        for i in range(len(self.maps)):
            fmap = self.maps[i]
            map_size = self.preset.map_sizes[i]
            for j in range(5):
                name = 'classifier{}_{}'.format(i, j)
                tmp = classifier(fmap, self.num_vars, map_size, name)
                self.classifiers.append(tmp)
            if i < len(self.__maps) - 1:
                name = 'classifier{}_6'.format(i)
                tmp = classifier(fmap, self.num_vars, map_size, name)
                self.classifiers.append(tmp)
        
        with tf.variable_scope('result'):
            self.result = tf.concat(self.classifiers, axis=1, name='result')
            self.classifier = self.result[:,:,:self.num_classes]
            self.locator = self.result[:,:,self.num_classes:]
        
    def get_optimizer(self, gt, learning_rate=0.0005):
        gt_cl = gt[:,:,:self.num_classes]
        gt_loc = gt[:,:, self.num_classes:]
        
        batch_size = tf.shape(gt_cl)[0]
        
        with tf.variable_scope('match_cnt'):
            total_cnt = tf.ones([batch_size], dtype=tf.int64) * tf.to_int64(self.preset. num_anchors)
            neg_cnt = tf.count_nonzero(gt_cl[:,:,-1], axis=1)
            pos_cnt = total_cnt - neg_cnt
        
        with tf.variable_scope('match_masks'):
            pos_mask = tf.equal(gt_cl[:,:,-1], 0)
            neg_mask = tf.logical_not(pos_mask)
        
        with tf.variable_scope('confidence_loss'):
            ce = tf.nn.softmax_cross_entropy_with_logits(labels=gt_cl, logits=self.classifier)
            pos = tf.where(pos_mask, ce, tf.zeros_like(ce))
            pos_sum = tf.reduce_sum(pos, axis=-1)
            neg = tf.where(neg_mask, ce, tf.zeros_like(ce))
            neg_top = tf.nn.top_k(neg, self.conf.num_anchors)[0]
            neg_cnt_max = tf.minimum(neg_cnt, 3*pos_cnt)
            neg_cnt_max_t = tf.expand_dims(neg_cnt_max, 1)
            rng = tf.range(0, self.conf.num_anchors, 1)
            range_row = tf.to_int64(tf.expand_dims(rng, 0))
            neg_max_mask = tf.less(range_row, neg_cnt_max_t)
            neg_max = tf.where(neg_max_mask, neg_top, tf.zeros_like(neg_top))
            neg_max_sum = tf.reduce_sum(neg_max, axis=-1)
            confidence_loss = tf.add(pos_sum, neg_max_sum)
        
        with tf.variable_scope('localization_loss'):
            loc_diff = tf.subtract(self.locator, gt_loc)
            loc_loss = smooth_l1_loss(loc_diff)
            loc_loss_sum = tf.reduce_sum(loc_loss, axis=-1)
            pos_locs = tf.where(pos_mask, loc_loss_sum, tf.zeros_like(loc_loss_sum))
            localization_loss = tf.reduce_sum(pos_locs, axis=-1)
            
        with tf.variable_scope('total_loss'):
            sum_losses = tf.add(confidence_loss, localization_loss)
            pos_cnt_safe = tf.where(tf.equal(pos_cnt, 0),
                                         tf.ones([batch_size])*10e-15,
                                         tf.to_float(pos_cnt))
            total_losses = tf.where(tf.less(pos_num,1),
                                   tf.zeros([batch_size]),
                                   tf.div(sum_losses, pos_cnt_safe))
            loss = tf.reduce_mean(total_losses)
            
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            optimizer = optimizer.minimize(loss)
        
        return optimizer, loss


# In[ ]:





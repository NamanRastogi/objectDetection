
# coding: utf-8

# In[2]:


import argparse
import math
import sys
import os
import tensorflow as tf
import numpy as np
from ssd import SSD
from utility import *
from tqdm import tqdm


# In[3]:
class TrainingData:
    def __init__(self, data_dir):
        with open(data_dir+'/training-data.pkl', 'rb') as f:
            data = pickle.load(f)
        with open(data_dir+'/train-samples.pkl', 'rb') as f:
            train_samples = pickle.load(f)
        with open(data_dir+'/valid-samples.pkl', 'rb') as f:
            valid_samples = pickle.load(f)
    
    # gets batch of data for training
    def get_batch(self, file_list):
        random.shuffle(file_list)
        for i in range(0, len(file_list), 32):
            data = file_list[i:i+32]
            images = []
            labels = []
            
            # read input and target
            for pt in data:
                image_file = pt[0].filename
                label_file = pt[1]
                
                image = cv2.resize(cv2.imread(image_file), image_size)
                label = np.load(label_file)
                
                images.append(image.astype(np.float32))
                labels.append(label)
                  
        return np.array(images), np.array(labels)

def main():
    parser = argparse.ArgumentParser(description='Train Network')
    parser.add_argument('--data-dir', default='data', help='data directory')
    args = parser.parse_args()
    
    td = TrainingData(args.data_dir)
    
    with tf.Session() as sess:
        net = SSD(sess)
        net.create_from_vgg(args.vgg_dir, td.num_classes, td.conf)
        
        labels = tf.placeholder(tf.float32, shape=[None, None, td.num_classes+5])
        optimizer, loss = net.get_optimizer(labels)
        summary_writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
        saver = tf.train.Saver(max_to_keep=10)
        n_batches = int(math.ceil(td.num_train/args.batch_size))
        init_vars(sess)
        
        validation_loss = tf.placeholder(tf.float32)
        validation_loss_summary_op = tf.summary.scalar('validation_loss', validation_loss)
        training_loss = tf.placeholder(tf.float32)
        training_loss_summary_op = tf.summary.scalar('training_loss', training_loss)
        
        for e in range(args.epochs):
            generator = td.train_generator(args.batch_size)
            description = 'Epoch {}/{}'.format(e+1, args.epochs)
            training_loss_total = 0
            for x, y in tqdm(generator, total=n_train_batches, desc=description, unit='batches'):
                feed = {net.image_input: x,
                       labels: y, net.keep_prob: 1}
                loss_batch, _ = sess.run([loss, optimizer], feed_dict=feed)
                training_loss_total += loss_batch * x.shape[0]
            training_loss_total /= td.num_train
            
            generator = tf.valid_generator(args.batch_size)
            validation_loss_total = 0
            for x, y in generator:
                feed = {net.image_input: x,
                       labels: y, net.keep_prob: 1}
                loss_batch, _ = sess.run([loss], feed_dict=feed)
                validation_loss_total += loss_batch * x.shape[0]
            validation_loss_total /= td.num_valid
                
            feed = {validation_loss: validation_loss_total,
                    training_loss:   training_loss_total}
            loss_summary = sess.run([validation_loss_summary_op,
                                     training_loss_summary_op],
                                    feed_dict=feed)
            summary_writer.add_summary(loss_summary[0], e)
            summary_writer.add_summary(loss_summary[1], e)    
            
            if (e+1) % args.checkpoint_interval == 0:
                checkpoint = '{}/e{}.ckpt'.format(args.name, e+1)
                saver.save(sess, checkpoint)

        checkpoint = '{}/final.ckpt'.format(args.name)
        saver.save(sess, checkpoint)
        
    return 0


# In[ ]:


if __name__ == '__main__':
    sys.exit(main())


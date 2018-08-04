#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:40:30 2018

@author: kaustubh
"""

import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

#Parameters
learning_rate = 0.01
n_step = 500
batch = 128
display = 100

n_hid1 = 256
n_hid2 = 256
n_input = 784
n_class = 10

X = tf.placeholder("float",[None,n_input])
Y = tf.placeholder("float",[None,n_class])

weights = {
        'h1' : tf.Variable(tf.random_normal([n_input,n_hid1])),
        'h2' : tf.Variable(tf.random_normal([n_hid1,n_hid2])),
        'out' : tf.Variable(tf.random_normal([n_hid2,n_class]))
        }

biases = {
        'b1' : tf.Variable(tf.random_normal([n_hid1])),
        'b2' : tf.Variable(tf.random_normal([n_hid2])),
        'out' : tf.Variable(tf.random_normal([n_class]))
        }

def nn(x):
    layer1 = tf.add(tf.matmul(x,weights['h1']), biases['b1'])
    layer11 = tf.nn.relu(layer1)
    layer2 = tf.add(tf.matmul(layer11,weights['h2']), biases['b2'])
    layer22 = tf.nn.relu(layer2)
    output = tf.add(tf.matmul(layer22,weights['out']), biases['out'])
    return output

model = nn(X)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1,n_step+1):
        batch_x,batch_y = mnist.train.next_batch(batch)
        sess.run(train, feed_dict={X: batch_x,Y: batch_y})
        if step % display ==0 or step ==1:
            loss1, acc = sess.run([loss,accuracy], feed_dict = {X:batch_x,Y:batch_y})
            print("Step "+str(step) + ", Minibatch Loss= "+\
                  "{:.4f}".format(loss1) +", Training Accuracy= "+\
                  "{:.3f}".format(acc))
    print("Optimization finished!")
    print("Testing Accuracy: ", \
          sess.run(accuracy, feed_dict={X:mnist.test.images,
                                        Y:mnist.test.labels}))

#plt.imshow(np.reshape(mnist.test.images[1],[28,28]),cmap='gray')

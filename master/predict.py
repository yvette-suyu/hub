import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import time

learningrate = 0.01
batch_size = 20
n_epochs = 20

# step 1
docu = 'E:\document\code\matlab\generate trajectory\dataset\zong/'
for ff,_,f in os.walk(docu):
    for name in f:
        print('name',name)
        datapath = os.path.join(docu, name)



# num_trajectories
# # step 2
# # X = tf.placeholder(tf.float32, shape=(1,784), name = 'X')
# # X = tf.placeholder(tf.float32, [batch_size,784], name = 'X_placeholder')
# t = tf.placeholder(tf.int32, [batch_size,100], name = 'time')
#
# # Y = tf.placeholder(tf.int8, name = 'Y')
# # Y = tf.placeholder(tf.int32, [batch_size,10], name = 'Y_placeholder')
# V_ob = tf.placeholder(tf.float32, [batch_size,100], name = 'V_ob')
#
# # step 3
# # w = tf.Variable(0.0, name='w')
# # w = tf.Variable(tf.random_normal(shape=[784,10], stddev=0.01), name='weights')
# w_0 = tf.Variable(tf.random_normal(shape=[1,100], stddev=0.01), name='W_0')
# # b = tf.Variable(0.0, name='b')
# # b = tf.Variable(tf.zeros([1,10]), name='bias')
# w_1 = tf.Variable(tf.random_normal(shape=[1,1], stddev=0.01), name='W_1')
# # step 4
# V_predicted = tf.add(w_0,tf.matmul(w_1,t))
#
#
# # step 5
# # loss = tf.nn.softmax_cross_entropy_with_logits(logits, Y, name='loss')
# loss = tf.square(V_ob - V_predicted, name='loss')
#
# # step 6
# # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate).minimize(loss)
# # step 7
#
# with tf.Session() as sess:
#
#     writer = tf.summary.FileWriter('./graphs/',sess.graph) # add!
#     #
#     start_time = time.time()
#     sess.run(tf.global_variables_initializer())
#     n_batches = int(num_trajectories/batch_size)
#     for i in range(n_epochs):
#         total_loss = 0
#
#         for _ in range(n_batches):
#             t_batch,v_batch = mnist.train.next_batch(batch_size)
#             _,loss_batch = sess.run([optimizer,loss],feed_dict={t:t_batch,V_ob:v_batch})
#             total_loss += loss_batch
#             print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
#
#     print('Total time : {0} seconds'.format(time.time() - start_time))
#
#     print('Optimizer Finished!')
#
#     # test model
#
#     writer.close()
#     w_0, w_1 = sess.run([w_0,w_1])
#
# print('w_0,w_1',w_0,w_1)
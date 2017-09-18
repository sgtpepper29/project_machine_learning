from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
import matplotlib.pyplot as plt
import numpy as np
import random as ran

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

def TRAIN_SIZE(num):
    #print ('Total Training Images in Dataset = ' + str(mnist.train.images.shape))
    #print ('--------------------------------------------------')
    x_train = mnist.train.images[:num,:]
    #print ('x_train Examples Loaded = ' + str(x_train.shape))
    y_train = mnist.train.labels[:num,:]
    #print ('y_train Examples Loaded = ' + str(y_train.shape))
    print('')
    return x_train, y_train

#x_train= 55000 imagenes de 784 pixeles cada una, asociado a las imagenes
#y_train= asosciado a los labels (10, del 0 al 9)
def TEST_SIZE(num):
    #print ('Total Test Examples in Dataset = ' + str(mnist.test.images.shape))
    #print ('--------------------------------------------------')
    x_test = mnist.test.images[:num,:]
    #print ('x_test Examples Loaded = ' + str(x_test.shape))
    y_test = mnist.test.labels[:num,:]
    #print ('y_test Examples Loaded = ' + str(y_test.shape))
    return x_test, y_test

  
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def display_compare(num,y_conv,sess):

    
    # Aca se cargará solo un ejemplo de la base de datos mnist
    x_train = mnist.train.images[num,:].reshape(1,784)
    y_train = mnist.train.labels[num,:]

    label = y_train.argmax()
    y2=tf.nn.softmax(y_conv)
    prediction = sess.run(y2, feed_dict={x: x_train, keep_prob:0.5}).argmax()
    plt.title('Predicción: %d Etiqueta Real: %d' % (prediction, label))
    plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    plt.show()

def var_capas(pasos_ent, capas, neuronas):

    print('')
    print ('--------------------------------------------------')
    print('Cantidad de capas: %d' %(capas))
    print ('--------------------------------------------------')
    print('Cantidad de neuronas: %d' %(neuronas))
    print ('--------------------------------------------------')
    print('')
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    x_image2 = x_image
    ini=1
    lay=2
    
    for i in range(lay):
        fin=(i+1)*32
        W_conv = weight_variable([5, 5, ini, fin])
        b_conv = bias_variable([fin])
        h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
        h_pool = max_pool_2x2(h_conv)
        x_image = h_pool
        ini=fin

      #W_conv2 = weight_variable([5, 5, 32, 64])
      #b_conv2 = bias_variable([64])
    
      #h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
      #h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])

    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(x_image, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

      
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)) 


    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
   
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(pasos_ent+1):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
              train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
              print('Paso del entrenamiento: %d - Precisión = %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print('Precisión del Test: %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        
        display_compare(ran.randint(0, 55000),y_conv,sess)

def una_capa(pasos_ent,capas, neuronas):
  
  W = tf.Variable(tf.zeros([784,10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.nn.softmax(tf.matmul(x,W) + b)

  print('')
  print ('--------------------------------------------------')
  print('Cantidad de capas: %d' %(capas))
  print ('--------------------------------------------------')
  print('')
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))  
  x_train, y_train = TRAIN_SIZE(55000)
  x_test, y_test = TEST_SIZE(10000)
  LEARNING_RATE = 0.1

  #aca se determina la cantidad de steps
  TRAIN_STEPS = pasos_ent

  init = tf.global_variables_initializer()

  sess.run(init)

  training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  
  #if(FLAGS.entrena_o_no==1):
  for i in range(TRAIN_STEPS+1):
          sess.run(training, feed_dict={x: x_train, y_: y_train})
          if i%100 == 0:
            print('Paso del entrenamiento:' + str(i) + ' - Precisión =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) + ' - Pérdida = ' + str(sess.run(cross_entropy, {x: x_train, y_: y_train})))
  print('Precisión del Test: %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))          
  display_compare(ran.randint(0, 55000),y,sess)

  
def main(_):

  pasos_ent= FLAGS.pasos_entrenamiento
  cant_capas=FLAGS.capas
  cant_neu=FLAGS.neuronas
  
  if FLAGS.capas == 1:
    una_capa(pasos_ent,cant_capas)
  else:
    var_capas(pasos_ent,cant_capas, cant_neu)
         
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'pasos_entrenamiento',
      type=int,
      help='cantidad de pasos del entrenamiento'
  )

  parser.add_argument(
      'capas',
      type=int,
      help='cantidad de capas'
  )
  
  parser.add_argument(
      'neuronas',
      type=int,
      help='cantidad de neuronas'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)




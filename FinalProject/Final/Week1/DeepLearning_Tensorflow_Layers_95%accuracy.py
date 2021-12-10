#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 11:14:30 2021

@author: runnithan
"""


#Complete the tensorflow network as shown in the textbook (Chapter 2), 
#which should print out the accuracy every 100 iterations of training and 
#use layer to define the network, and reach 95% accuracy.

%tensorflow_version 1.x
import tensorflow as tf


##Let us import the mnist data set. The mnist dataset is a collection of images from 0 to 9.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##The mnist dataset is divided into two parts:
##55,000 rows of training dataset and 10,000 rows of test dataset
##Each mnist dataset contains 2 parts : a picture containing handwritten digits 
##and a corresponding tag
##If you print the dimension of the mnist.train.images , you can see it is shaped as
##[55000,784] tensor. So, Each there is 55000 images. Each image is 28 * 28 pixels. 
##So a total of 28 pixels * 28 pixels for a total of 784 pixels, which intern is represented as 
##784 element array. Each number in the array represent the intensity of the associated 
##pixel's grayscale(on a scle of 0-1)
#print("dimension of mnist train images", mnist.train.images.shape)

##The mnist tag is a number between 0 and 9 and represents the number in a given
##picture. The tag data is represented in on-hot vector format. The number n
##will be represented as a 10-dimensional vector with only i in the n-th dimension and the 
##rest will be 0.
#print("dimension of mnist train labels", mnist.train.labels.shape)
#print("dimension of mnist test images", mnist.test.images.shape)
#print("dimension of mnist test labels", mnist.test.labels.shape)

batchSz=100
##our model requires weight and offset(TF parameters). There is 3 stage in createing TF parameters:
##1. Create a tensor with intial values
##2. Turn the tensor into variable
##3. Then intialize the variables/parameters

###1. tf.random_normal([784, 10] creates a tensor of shape [784, 10], whose ten values are
##random number generated from a normal distribution with sd 0.1 
###2. tf.variable() takes the tensor that is created by the above function  and 
##adds a piece of the TF graph that creates A VARIABLE of same shape and value.

##Let us create placeholder. We enter the value when Tensorflow runs the calculation
##This place holder is for the image and the labels
img=tf.placeholder(tf.float32, [None,784])
ans = tf.placeholder(tf.float32, [None,10])

print("Image variable shape ", img.shape)
print("Ans variable shape ", ans.shape)


###Increasing the accuracy to 95% using layers
##Let us build a 2 hidden layered neural network. The output layer will be
##a dense layer of 10 nodes- as there are 10 classes


# number of nodes in input layer
n_input = 784
# number of nodes in 1st hidden layer
n_hidden1 = 500
# number of nodes in 2nd hidden layer
n_hidden2 = 500
# number of nodes in 2nd hidden layer
n_hidden3 = 500
# number of nodes in output layer
n_class = 10
# number of epochs to run
n_epoch = 10


##Let us define the learned variables, the weights and the bias
W1 = tf.Variable(tf.random_normal([n_input, n_hidden1],stddev=.1))
b1 = tf.Variable(tf.random_normal([n_hidden1],stddev=.1))

#The softmax function will convert the output from the last layer of neural networks 
#into a probability value and the higher probability will be the actual output

layer1 = tf.nn.softmax(tf.add(tf.matmul(img,W1),b1))  
#print("layer 1: ", layer1.shape)

W2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2],stddev=.1))
b2 = tf.Variable(tf.random_normal([n_hidden2],stddev=.1)) 

layer2 = tf.nn.softmax(tf.add(tf.matmul(layer1,W2),b2))
#print("layer 2: ", layer2.shape)

W3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3],stddev=.1))
b3 = tf.Variable(tf.random_normal([n_hidden3],stddev=.1)) 

 
layer3 = tf.add(tf.matmul(layer2,W3),b3)##Output layer

W4 = tf.Variable(tf.random_normal([n_hidden3, n_class],stddev=.1))
b4 = tf.Variable(tf.random_normal([n_class],stddev=.1)) 

 
prediction = tf.add(tf.matmul(layer3,W4),b4)##Output layer


#print("output layer : ", prediction.shape)
##computing softmax cross entropy
prbs = tf.nn.softmax(prediction)

##Inorder to train our model we need indicators, for evaluating whether a model is good or not
##In ML indicators are defined to indicate whether a model is bad and this is called as cost or loss
##We then try to minimize the indicator 

xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs),
                                       reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer().minimize(xEnt)
  
##Start the session for the graph
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(n_epoch):
            epoch_loss = 0
            for _ in range(1000):
                epoch_x, epoch_y = mnist.train.next_batch(batchSz)
                _, c = sess.run([optimizer, xEnt], feed_dict={img: epoch_x, ans: epoch_y})
                epoch_loss += c
            print('epoch', epoch, 'completed out of',
                  n_epoch, 'loss:', epoch_loss)
    print(mnist.test.images.shape)
    numCorrect= tf.equal(tf.argmax(prbs,1), tf.argmax(ans,1))
    accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))
    print('accuracy:', accuracy.eval({img: mnist.test.images, ans: mnist.test.labels}))
        

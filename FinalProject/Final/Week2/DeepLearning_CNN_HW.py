# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


###############################################################################################
######################A TF program for a Convolutional Neural Neywork using mnist dataset########################
##############################################################################################

###Let us import tensorflow and then read the Mnist data
%tensorflow_version 1.x
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the fashion-mnist pre-shuffled train data and test data
from tensorflow.examples.tutorials.mnist import input_data


fashion_mnist = input_data.read_data_sets('data/fashion',one_hot=True,\
                                 source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

###a tensor is a multidimensional array. In tensorflow every line of code that you write
##has to go through a computational graph.

##placeholders allow you to feed input on the run. Definig a node as placeholder assures the node that it is expected
##to receive a value later or during run time. Runtime means the input is fed to the placeholder when we run the
##computational graph

#Variables allow you to modify the graph. It can ptoduce new outputs with respect to the same inputs. 
##Variables are not not intialized when you call tf.Variable()
##Variables survive across multiple executions of a graph, unlike normal tensors [Normal tensors are instantiated 
##when a graph is run and are deleted immediately afterwards]

###Let us check the Training set and Test set shapes

print("Shape of the Images in the taining data is: " , fashion_mnist.train.images.shape)
print("Shape of the Labels in the taining data is: " , fashion_mnist.train.labels.shape)
print("Shape of the Images in the test data is: " , fashion_mnist.test.images.shape)
print("Shape of the Labels in the test data is: " , fashion_mnist.test.labels.shape)


##Let us see if the image pixels is re-sccaled between 0 and 1.
print("The minimum pixel value is: ", np.min(fashion_mnist.train.images[5]))

print("The maximum pixel value is: ", np.max(fashion_mnist.train.images[5]))

##Inorder to feed the sample to the CNN model, we need to reshape each training and test dataset
##to 28 * 28 * 1
##We need to reshape the data because the CNN expects certain input shape:
##(number of images , image x dimension, image y dimension, number of channels)
train_image = fashion_mnist.train.images.reshape(-1, 28,28,1)#Here -1 means the entire column, no matter what the batch size is in runtime
test_image = fashion_mnist.test.images.reshape(-1, 28,28,1)
print(train_image.shape)
print(test_image.shape)


train_labels = fashion_mnist.train.labels
test_labels = fashion_mnist.test.labels
##There are 4 main operations in the ConvNet
##1. Convolution
##2. Non Linearity
##3. Poolong or Sub sampling
##4. Classification() ##Fully connected layer
##The Conv2D layer is a convolutional layer required to craeting a convolutional kernel 
##that is convovled with the layer input to produce atensor of outputs. A convolutional layer has
##several filters that perform the convolutional operation.


##Let us start by defining the training iterations:
##training_iters - indicates the number of times you train your network
n_epoch = 10
##learning_rate - the weights get updated by multiplying by leaning rate
learning_rate = 0.001
##batchSz ## this means your training images will be divided into fixed batch size
batchSz = 100
#Let us have 10 training iteration, learning rate of 0.001 and batch_size of 100

##Let us create placeholders
image = tf.placeholder("float", [None, 28,28,1])
ans = tf.placeholder("float",[None, 10])

##In the convolutional network you will pass 4 arguments
##1. input img
##2. weights w
##3. bias b
##4. strides ##This is set to 1 by default

##Let us create the weight and bias of the first layer

##Let us create 16 - 4*4 filters
##So the firrst weight w1 has an argument shape , and it takes 4 values - the first and second are filter size
#4*4, the third is the number of channels. The fourth one is 
##number of convulutional filter we need
##so let us create 16 convolutional filter of size 4* 4
W1 = tf.Variable(tf.random_normal([4,4,1,16],stddev=.1))
##The bias variable will be having 16 bias parameter
b1 = tf.Variable(tf.random_normal([16],stddev=.1))

##Let us perform convolution using padding 1,2,2,1
conv1 = tf.nn.conv2d(image, W1 , strides = [1, 2, 2, 1], padding = 'SAME')
conv1 = tf.nn.relu(conv1 + b1)

##Let us perform pooling 


##Let us create 32 - 2*2 filters
##So the second weight w2 has an argument shape , and it takes 4 values - the first and second are filter size
#4*4, the third is the number of channels from the previous output. Since we passed 16 convolutoional filter
#to the input image, we eill be having 16 channels as output from the first convolutional layer operation. The fourth one is 
##number of convulutional filter we need
##so let us create 32 convolutional filter of size 4* 4
W2 = tf.Variable(tf.random_normal([2,2,16,32],stddev=.1))
##The bias variable will be having 32 bias parameter
b2 = tf.Variable(tf.random_normal([32],stddev=.1))

conv2 = tf.nn.conv2d(conv1, W2 , strides = [1, 2, 2, 1], padding = 'SAME')
conv2 = tf.nn.relu(conv2 + b2)

##Fully connected layer input

##Reshape conv2 output to fit fully connected layer input
conv2 = tf.reshape(conv2, [-1, 1568] )
W = tf.Variable(tf.random_normal([1568,10],stddev=.1))
b = tf.Variable(tf.random_normal([10],stddev=.1)) 

print(conv2.shape)
print(W.shape)
print(b)

##computing softmax cross entropy
prbs = tf.nn.softmax(tf.matmul(conv2,W)+b)
##Inorder to train our model we need indicators, for evaluating whether a model is good or not
##In ML indicators are defined to indicate whether a model is bad and this is called as cost or loss
##We then try to minimize the indicator 

xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs),
                                       reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(xEnt)
numCorrect= tf.equal(tf.argmax(prbs,1), tf.argmax(ans,1))
accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))
##Start the session for the graph
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    train_accuracy = []
    test_accuracy = []

    train_loss = []
    test_loss = []
    for epoch in range(n_epoch):
        for batch in range(len(train_image)//batchSz):
                epoch_x = train_image[batch*batchSz:min((batch+1)*batchSz,len(train_image))]
                epoch_y = train_labels[batch*batchSz:min((batch+1)*batchSz,len(train_labels))]
                opt = sess.run([optimizer], feed_dict={image: epoch_x, ans: epoch_y})
                loss, acc = sess.run([xEnt, accuracy], feed_dict={image: epoch_x, ans: epoch_y})
        
   
    test_acc = sess.run([accuracy], feed_dict={image: test_image,ans : test_labels}) 
    #print(test_acc.shape)  
    test_accuracy.append(test_acc)
    train_accuracy.append(acc)
    print('accuracy:', test_acc)

  

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


###############################################################################################
######################A TF program for a feed-forward NN Mnist program########################
##############################################################################################


##used tutorial https://tensorflowguide.readthedocs.io/en/latest/tensorflow/mnist.html
###Let us import tensorflow and then read the Mnist data
%tensorflow_version 1.x
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

##mnist dataset is acollection of imaages of digits from 0 to 9
from tensorflow.examples.tutorials.mnist import input_data
##reading the data without oneshot labels
data = input_data.read_data_sets("MNIST_data/")
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print("The labels of the data, if we read without one hot encoding:")
print(data.train.labels)##will print [7 3 4 ... 5 6 8]

##Let us read with onne sht labels
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print("The labels of the data, if we read with one hot encoding:")
print(mnist.train.labels[0:5,:])
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
##will print the label for the first 5 elements


##Let us try plotting the first images in the train data
#plt.imshow(mnist.train.images[1].reshape(28,28), cmap =plt.get_cmap('gray'))

#Now let us plot all the numbers in the train dataset
print("Plotting the Training set")
for i in range(9):
    #plot this as 3 columns and 3 rows
    plt.subplot(330+1+i)
    plt.suptitle("mnist Train set")
    plt.imshow(mnist.train.images[i].reshape(28,28), cmap =plt.get_cmap('gray'))
   
plt.show()
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
###Plotting the test set with numbers as label

mnist.test.numbers = np.array([label.argmax() for label in mnist.test.labels])
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print("Coverting the one hot labels to numbers:")
print(mnist.test.numbers)
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")

#plot this as 3 columns and 3 rows
print("-------------------------------------------------------------")
print("plotting the Test set")
for i in range(9):
    plt.subplot(330+1+i)  
    plt.suptitle("mnist Test set")
    plt.imshow(mnist.test.images[i].reshape(28,28), cmap =plt.get_cmap('gray'))
    xlabel = "True : {}".format(mnist.test.numbers[i])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
    plt.title(xlabel)
   
plt.show()
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")


####################################################
#################Create a tensorflow graph#########
####################################################

##Create a placeholder for input value

##The fundamental data structures in a tensorflow is Tensors. When we define the 
##placeholder Z, along with it's type, it has a shape too. If we are defining a scalar tensor,
##we can give only data type.
##If a vector, we need to guve the data type and the shape. Here since we are creating a
##two dimensional tensor, we will give row and column
##x and y are just placeholders, we will enter values when tensorflow runs the calculation
x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None, 10])



#let us create another variable for storing the numbered labels, i.e., 
#corresponding values to one hot labels
y_number = tf.placeholder(tf.int64,[None])

#print(x , y)
##create variables of tensorflow
weights = tf.Variable(tf.zeros([784,10]))
bias = tf.Variable(tf.zeros([10]))
#print(weights)
#print(bias)

##Create the model
##A simple neoural network model is output = (input * weight) + bias

##Create the model
logits = tf.matmul(x , weights) + bias #the matmul gets the dot product between the two inputs
#print(logits)
##converting the result to probability density using softmax function,
##so that the sum of the vector will be 1

y_pred = tf.nn.softmax(logits)
#print(y_pred)

##store the index-number which has the highest value
y_pred_number = tf.argmax(y_pred , dimension = 1)
#print(y_pred_number)

##Execute the model without an optimization
##Here the weights are intialised as 0 
##therefpre all the values will be detected as 0


#Create a session
session = tf.Session()

##Then let us initialise the variables
session.run(tf.global_variables_initializer())

##Now let us assighn the values from the train dataset to the placeholder

feed_dict = {
    x:mnist.train.images,
    y:mnist.train.labels
}

result = session.run(y_pred_number, feed_dict = feed_dict)
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print("Predictions Before Optimisation")
#print(sum(result)) # here the sum is zero as all the predictions are zer0

###Plot the images with prediction before gradient optimizer
print("Before Optimisation:",sum(result)) # here the sum is zero as all the predictions are zer0
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print("plotting the Test set with the prediction before optimisation")
for i in range(9):
    plt.subplot(330+1+i)  
    plt.suptitle("mnist Test set prediction")
    plt.imshow(mnist.test.images[i].reshape(28,28), cmap =plt.get_cmap('gray'))
    xlabel = "True : {}, Predict: {}".format(mnist.test.numbers[i], result[i])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
    plt.title(xlabel)
   
plt.show()
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")

##Now let us use Gradient optimizer for prediction
##Define the cross entropy function for optimization.

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
# use optimizer on cost function "cross-entropy"
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#print(optimizer)

##Let us evaluate the accuracy of the prediction
##Correct prediction is when the predicted is actual
##prediction is in y_pred_number
##actual is y_number

correct_prediction = tf.equal(y_pred_number, y_number)

accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))

# create session
session = tf.Session()
# initialize variables
session.run(tf.global_variables_initializer())

# assign value to placeholder
feed_dict = {
    x: mnist.train.images,
    y: mnist.train.labels
}


result = session.run(optimizer, feed_dict=feed_dict)
#print("After Optimisation:",result)

feed_dict_accuracy = {
    x: mnist.test.images,
    y: mnist.test.labels,
    y_number: mnist.test.numbers
}

prediction, accu = (session.run([y_pred_number, accuracy], feed_dict = feed_dict_accuracy))
print("Test Accuracy after Gradient Optimisation but before Batch processing: ", accu)

for i in range(9):
    plt.subplot(330+1+i)  
    plt.suptitle("mnist Test set prediction using Gradient optimizer")
    plt.imshow(mnist.test.images[i].reshape(28,28), cmap =plt.get_cmap('gray'))
    xlabel = "True : {}, Predict: {}".format(mnist.test.numbers[i], prediction[i])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
    plt.title(xlabel)
   
plt.show()
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")

##Running in batch mode

###initialise a variablefor 100 batchsize 
batchSz = 100
##Running in batch mode
batchSz = 100
x = tf.placeholder(tf.float32, [batchSz, 784])  # store data of image
y = tf.placeholder(tf.float32, [batchSz, 10])     # store one_hot labels

weights = tf.Variable(tf.random_normal([784,1],stddev = .1))
bias = tf.Variable(tf.random_normal([10],stddev = .1))

logits = tf.matmul(x , weights) + bias
y_pred = tf.nn.softmax(logits)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred),
                                   reduction_indices=[1]))

#cross_entropy = -tf.reduce_sum(y*tf.log(y_pred))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

##computing the accuracy of the model
##Count the number of correct Answers and then divide by the number of images processed
correct_prediction = tf.equal(tf.argmax(y_pred , dimension = 1), tf.argmax(y , dimension = 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))

rep = 1000
session = tf.Session()

session.run(tf.global_variables_initializer())
###Train the model
for i in range(rep):
  batch = mnist.train.next_batch(batchSz)
  session.run(optimizer, feed_dict={x: batch[0], y: batch[1]}) 


sumAcc = 0
for i in range(rep):
  batch = mnist.test.next_batch(batchSz)
  sumAcc+=session.run(accuracy, feed_dict={x: batch[0], y: batch[1]}) 

print("Test Accuracy after batch processing: %r" % (sumAcc/1000))








import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2 as cv
import os
import math

import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
import pandas as pd
from tensorflow.python.framework import ops


np.random.seed(1)

tf.reset_default_graph()
ops.reset_default_graph()
# %matplotlib inline

#final_data_1 and 2 are the numpy array files for the images in the folder img and annotations.csv file
#total of 5 GB due to conversion of values to int
Z2= np.load('final_data_1.npy')
Z1= np.load('final_data_2.npy')

print(Z2[:,0])
print(Z1.shape)
index = 1647

plt.imshow(Z1[index,1:].reshape(200,200,3),cmap="hot")
print("label = " + str(Z1[index,0]))
#cv.imshow("Example",X_train[index].reshape(200,200,3))
#Print shapes

Z=np.append(Z2[0:1700],Z1[0:1700],axis=0)
Z1=None
del Z1
Z2=None
del Z2
print(Z.shape)

Z=np.take(Z,np.random.permutation(Z.shape[0]),axis=0,out=Z)

def get_data():
    X=((Z[:,1:Z.shape[1]]).reshape(Z.shape[0],200,200,3))/255
    Y=(Z[:,0]).reshape(Z.shape[0],1)
    y=[]
    for i in Y:
        if i[0]==None:
            continue
        else:
            y.append(int(i[0]))

    Y=np.array(y)
    Y=Y.reshape((Y.shape[0],1)) 
    print("Y.shape")
    print(Y.shape)
    print("X's shape")
    print(X.shape)
    #X=X.transpose([1,2,3,0])
    print(X.shape)    
    Y_one_hot = np.zeros((Y.shape[0],2))

    for i in range(Y.shape[0]):
        Y_one_hot[i,Y[i,0]] = 1
    Y=Y_one_hot
    cv_start_index = (int)(0.97*X.shape[0])
    print("HEY")
    print(Y_one_hot.shape)
    
    print(X.shape, Y_one_hot.shape)
    X_train = X[0:cv_start_index,:]
    Y_train = Y_one_hot[0:cv_start_index,:]
    X_CV = X[cv_start_index:,:]
    Y_CV = Y_one_hot[cv_start_index:,:]

    X_train = X_train.reshape(X_train.shape[0],200,200,3)
    X_CV = X_CV.reshape(X_CV.shape[0],200,200,3)

    return (X_train,Y_train,X_CV,Y_CV)

X_train,Y_train,X_CV,Y_CV=get_data()
print(X_train.shape,Y_train.shape,X_CV.shape,Y_CV.shape)

index = 1036
plt.imshow(X_train[index].reshape(200,200,3),cmap="hot")
print("label = " + str(Y_train[index]))
print(Y_train)
#cv.imshow("Example",X_train[index].reshape(200,200,3))
#Print shapes
print("X_train : " + str(X_train.shape))
print("Y_train : " + str(Y_train.shape))
print("X_CV : " + str(X_CV.shape))
print("Y_CV : " + str(Y_CV.shape))
print(X_train[index].shape)

def get_placeholders():
    X = tf.placeholder(tf.float32,shape=[None,X_train.shape[1],X_train.shape[2],X_train.shape[3]],name="X")
    Y = tf.placeholder(tf.float32,shape=[None,Y_train.shape[1]],name="Y")
    keep_prob = tf.placeholder(tf.float32)
    return X,Y,keep_prob

X,Y,keep_prob = get_placeholders()
print (X)
print (Y)
print (keep_prob)

def initialize_variables():
    parameters = {}
    
    tf.set_random_seed(1)
    
    W1 = tf.get_variable("W1",[5,5,3,32],initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2",[5,5,32,64],initializer = tf.contrib.layers.xavier_initializer(seed=0))
    flatten = tf.get_variable("flatten",[5,5,64,1],initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3",[2500,512],initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4",[512,32],initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W5 = tf.get_variable("W5",[32,2],initializer = tf.contrib.layers.xavier_initializer(seed=0))
    
    b1 = tf.Variable(tf.zeros([1,1,1,32]),name="b1")
    b2 = tf.Variable(tf.zeros([1,1,1,64]),name="b2")
    b_flatten=tf.Variable(tf.zeros([1,1,1,1]),name="b_flatten")
    b3 = tf.Variable(tf.zeros([512]),name="b3")
    b4 = tf.Variable(tf.zeros([32]),name="b4")
    b5 = tf.Variable(tf.zeros([2]),name="b5")
    
    parameters = {
        "W1":W1,
        "W2":W2,
        "W3":W3,
        "W4":W4,
        "W5":W5,
        "b1":b1,
        "b2":b2,
        "b3":b3,
        "b4":b4,
        "b5":b5,
        "flatten":flatten,
        "b_flatten":b_flatten
    }
    
    return parameters

tf.reset_default_graph()
par = initialize_variables()
par

def forward_pass(X_batch,parameters,keep_prob_):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    W4 = parameters["W4"]
    W5 = parameters["W5"]
    flatten=parameters["flatten"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    b4 = parameters["b4"]
    b5 = parameters["b5"]
    b_flatten=parameters["b_flatten"]
    #temp=1.0
    print("X_batch")
    print(X_batch.shape)
    
    Z1 = tf.nn.conv2d(X_batch,W1,strides = [1,1,1,1],padding = 'SAME') + b1
    print("Z1")
    print(Z1.shape)
    A1 = tf.nn.relu(Z1)
    A1=tf.nn.dropout(A1, keep_prob_)

    print("A1")
    print(A1.shape)
    P1 = tf.nn.max_pool(A1,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
    print("P1")
    print(P1.shape)
    
    
    Z2 = tf.nn.conv2d(P1,W2,strides = [1,1,1,1],padding = 'SAME') + b2
    print("Z2")
    print(Z2.shape)
    A2 = tf.nn.relu(Z2)
    A2=tf.nn.dropout(A2, keep_prob_)

    print("A2")
    print(A2.shape)
    P2 = tf.nn.max_pool(A2,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
    print("P2")
    print(P2.shape)
    
    
    P2_flat_ = tf.nn.conv2d(P2,flatten,strides = [1,1,1,1],padding='SAME')+b_flatten
    
    P2_flat = tf.nn.dropout((tf.nn.relu(P2_flat_)),keep_prob_)
    print("shape before flatten fucntion")
    print(P2_flat.shape)
    P2_flat=tf.contrib.layers.flatten(P2_flat)
    print("P2_flat")
    print(P2_flat.shape)
    
    Z3=tf.matmul(P2_flat,W3) + b3
    A3 = tf.nn.relu(Z3)
    A3=tf.nn.dropout(A3, keep_prob_)
    
    A4 = tf.nn.relu(tf.matmul(A3,W4) + b4)
    A4=tf.nn.dropout(A4, keep_prob_)
    Y_hat = tf.matmul(A4,W5) + b5
    
    return Y_hat

def compute_cost(Y_hat,Y):
    
    print("Y , Y_hat "+str(Y)+str(Y_hat))
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_hat,labels=Y))
    print(Y,Y_hat, cost)
    
    return cost

def sample_minibatches(X,Y,batch_size,seed = 0):
    np.random.seed(seed)
    
    shuffled_X = X 
    shuffled_Y = Y

    num_batches = (int)(X.shape[0]/(batch_size))
    minibatches = []
    
    for i in range(1,num_batches+1):
        minibatch_X = shuffled_X[(i-1)*batch_size:i*batch_size,:,:,:]
        minibatch_Y = shuffled_Y[(i-1)*batch_size:i*batch_size,:]
        minibatches.append((minibatch_X,minibatch_Y))
    
    if X.shape[0]%num_batches != 0:
        remainder = X.shape[0]%num_batches
        last_index = (int)(X.shape[0]/num_batches)
        
        minibatch_X = shuffled_X[last_index*num_batches:last_index*num_batches + remainder,:,:,:]
        minibatch_Y = shuffled_Y[last_index*num_batches:last_index*num_batches + remainder,:]
        
        minibatches.append((minibatch_X,minibatch_Y))
    
    return minibatches

def check_accuracy(Y_hat,Y):
    correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(Y_hat,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def model(learning_rate,batch_size,num_epochs,keep_prob_):
  

#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    tf.reset_default_graph()
    ops.reset_default_graph()
    tf.set_random_seed(1)
    
    num_minibatches = math.ceil(X_train.shape[0]/batch_size)
    
    seed = 5
    
    X,Y,keep_prob = get_placeholders()
    parameters = initialize_variables()
    Y_hat = forward_pass(X,parameters,keep_prob)
    cost = compute_cost(Y_hat,Y)
    accuracy=check_accuracy(Y_hat,Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    costs = []
    train_accuracies = []
    test_accuracies = []
    
    train_accuracy = 0.0
    test_accuracy = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(num_epochs):
            
            cost_ = 0.
            a_train = 0.
            a_test = 0.
            minibatches = sample_minibatches(X_train,Y_train,batch_size=batch_size,seed=seed)
            seed = seed + 1
            count=0        
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y) = minibatch
#                 print(minibatch_X.shape)
#                 print(minibatch_Y.shape)
                _,cur_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y,keep_prob:keep_prob_})
                train_accuracy = sess.run([accuracy],feed_dict={X: minibatch_X, Y: minibatch_Y,keep_prob:keep_prob_})
                #cost_ = ((count*cost_)+cur_cost)/(count+1)
                test_accuracy = sess.run([accuracy],feed_dict={X: X_CV, Y: Y_CV,keep_prob:keep_prob_})
                #a_train = ((count*a_train)+train_accuracy[0])/(count+1)
                #a_test =(test_accuracy[0]/1)
                cost_+=cur_cost
                a_train+=train_accuracy[0]
                a_test+=test_accuracy[0]
                #print("Train Accuracy : " + str(a_train))
                #print("Test Accuracy : " + str(a_test))
                #print("cost", str(cost_))
                #costs.append(cost_)
                #train_accuracies.append(a_train)
                #test_accuracies.append(a_test)
            cost_=cost_/num_minibatches
            a_train = a_train /num_minibatches
            a_test = a_test/num_minibatches
            costs.append(cost_)
            train_accuracies.append(a_train)
            test_accuracies.append(a_test)
            
            print("EPOCH : "+str(epoch))
            print("Train Accuracy : " + str(a_train))
            print("Test Accuracy : " + str(a_test))
            print("cost", str(cost_))
                
            #print("train accuracy")
            #print(type(train_accuracy))
            #print("Train Accuracy : " + str(a_train))
            #print("Test Accuracy : " + str(a_test))

            

    
        
    return costs,parameters,train_accuracies, test_accuracies

costs,params,train_acc,test_acc = model(learning_rate=0.00003,batch_size =100,num_epochs = 100,keep_prob_=0.80)

costs = np.array(costs)
costs = costs.reshape(len(costs),1)
train_acc = np.array(train_acc)
train_acc = train_acc.reshape(len(train_acc),1)
test_acc = np.array(test_acc)
test_acc = test_acc.reshape(len(test_acc),1)
steps = []
for i in range(0,costs.shape[0],1):
    steps.append(i)
steps = np.array(steps).reshape(costs.shape[0],1)  
print("Train Accuracy : "+str(train_acc[len(train_acc)-1,0]))
print("Test Accuracy : "+str(test_acc[len(test_acc)-1,0]))
plt.plot(steps,costs)
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.title("Cost vs Epochs")
plt.show()
plt.plot(steps, train_acc)
plt.ylabel('Train Accuracy')
plt.xlabel('Epochs')
plt.title("Train Accuracy vs Epochs")
plt.show()
plt.plot(steps,test_acc)
plt.ylabel('Test Accuracy')
plt.xlabel('Epochs')
plt.title("Test Accuracy vs Epochs")
plt.show()


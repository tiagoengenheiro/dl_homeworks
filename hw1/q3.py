import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import json
import pickle

import utils


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        self.W1=np.ones((hidden_size,n_features))
        self.b1=np.array([[-2],[-1]]) #2,1
        self.W2=np.array([[-1,1]]) #1,2
        self.b2=np.array([[-1]])

    def predict(self, X):

        z1=np.dot(self.W1,X.T)+np.tile(self.b1,X.shape[0]) #cada coluna para cada exemplo
        sign=np.vectorize(lambda x: 1 if x>=0 else -1)
        h1=sign(z1) 
        z2=np.dot(self.W2,h1)+np.tile(self.b2,X.shape[0])  #W2 = 4,200 200,97477 -> 4,97477
        output=sign(z2) 
        return output
    

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def forward(self, x_i): 
        self.z1=np.dot(self.W1,np.expand_dims(x_i,axis=1))+self.b1 #2,2 x 2,1 -> 2,1
        sign=np.vectorize(lambda x: 1 if x>=0 else -1)
        self.h1=sign(self.z1) # -> 2,1
        self.z2=np.dot(self.W2,self.h1)+self.b2 #->1,2 x 2,1 -> 1,1
        output=sign(self.z2)
        return np.squeeze(output) [()]
        
    def compute_loss(self,y_hat,y_i):
        loss = np.maximum(0, 1 - np.dot(y_hat,y_i))
        return loss
    
    def backward(self,y_hat,y_i,x_i):
        if y_hat!=y_i:
            self.W2+=np.dot(y_i,self.h1.T) #1,1 > 1,2
            self.W1[(y_i+1)//2]+=np.dot(y_i,x_i) #->  
            self.W1[(y_hat+1)//2]-=np.dot(y_i,x_i)

    def update_weights(self,lr):
        self.W1-=lr*self.dL_dW1
        self.b1-=lr*self.dL_db1
        self.W2-=lr*self.dL_dW2
        self.b2-=lr*self.dL_db2


    def train_epoch(self, X, y, learning_rate=1):
        loss=0
        for _,(x_i,y_i) in enumerate(zip(X,y)):
            y_hat=self.forward(x_i) 
            print(y_hat)
            loss+=self.compute_loss(y_hat,y_i)
            #self.backward(y_hat,y_i,x_i)
            #self.update_weights(learning_rate)
        loss/=X.shape[0] 
        return np.squeeze(loss) [()]


train_X = np.array([[1,1,1],[-1,1,1],[1,-1,1],[-1,-1,-1]])
train_y=np.array([-1,1,1,-1])
n_classes = 1
n_feats = train_X.shape[1]
print("N_classes,n_feats",n_classes,n_feats)
model = MLP(n_classes, n_feats, 2)
epochs=np.arange(1, 1)

for i in epochs:

    train_order = np.random.permutation(train_X.shape[0])
    train_X = train_X[train_order]
    train_y = train_y[train_order]
    loss = model.train_epoch(
        train_X,
        train_y,
        learning_rate=1)
    
print(model.predict(train_X))



# W1=np.array([[1,1],[1,1]])
# b1=np.array([[-1],[+1]]) #2,1
# W2=np.array([[-1,1]]) #1,2
# b2=np.array([[-1]])

# for x_i,y_i in zip(train_X,train_y):
#     z1=np.dot(W1,np.expand_dims(x_i,axis=1))+b1 #2,2 x 2,1 -> 2,1
#     sign=np.vectorize(lambda x: 1 if x>=0 else -1)
#     h1=sign(z1) # -> 2,1
#     z2=np.dot(W2,h1)+b2 #->1,2 x 2,1 -> 1,1
#     output=sign(z2)
#     output = np.squeeze(output) [()]
#     print(x_i,y_i,output)
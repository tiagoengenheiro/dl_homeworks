#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import json
import pickle

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


def sign(x):
    if x>=0:
        return 1
    else:
        return -1
    

class PerceptronM(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        y_hat=np.argmax(np.dot(self.W,x_i))
        if y_hat!=y_i:
            self.W[y_i]+=x_i
            self.W[y_hat]-=x_i


class LogisticRegressionM(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        scores=np.expand_dims(np.dot(self.W,x_i),axis=1) #4,1
        scores=np.apply_along_axis(np.exp,axis=1,arr=scores) 
        scores=scores/np.sum(scores,axis=0) #axis 0 é por coluna
        e_y=np.zeros((self.W.shape[0],1))
        e_y[y_i]=1
        scores=scores-e_y
        gradient=np.dot(scores,np.expand_dims(x_i,axis=1).T)
        self.W-=learning_rate*gradient

        
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        self.W1=0.1+0.1*np.random.randn(hidden_size,n_features)
        self.b1=np.zeros((hidden_size,1),dtype=np.float64)
        self.W2=0.1+0.1*np.random.randn(n_classes,hidden_size)
        self.b2=np.zeros((n_classes,1),dtype=np.float64)

    def softmax(self,x):
        f=x-np.max(x)
        return np.exp(f)/np.sum(np.exp(f),axis=0)

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        z1=np.dot(self.W1,X.T)+np.tile(self.b1,X.shape[0]) #cada coluna para cada exemplo
        h1=np.maximum(z1,0) #relu
        #print("h1",h1.shape)
        z2=np.dot(self.W2,h1)+np.tile(self.b2,X.shape[0])  #W2 = 4,200 200,97477 -> 4,97477
        #print("z2",z2.shape)
        output=self.softmax(z2)    
        #print("output",output.shape)
        #print("argmax",np.argmax(output,axis=0).shape)
        return np.argmax(output,axis=0)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        #print(y_hat[10])
        #print(y[10])
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def forward(self, x_i): 
        self.z1=np.dot(self.W1,np.expand_dims(x_i,axis=1))+self.b1 
        #print("z1",self.z1.shape)
        #print("z1",self.z1)
        self.h1=np.maximum(self.z1,0) #Relu
        #print("h1",self.h1.shape)
        self.z2=np.dot(self.W2,self.h1)+self.b2
        #print("z2",self.z2.shape)
        #print("z2",self.z2[0][0])
        output=self.softmax(self.z2)
        #print("output",output.shape)
        #print("output",output[0][0])
        return output
        
    def compute_loss(self,probs,y_i):
        loss= -np.log(probs[y_i])# if probs[y_i]!=0 else -np.log(10**(-30))
        return loss
    
    def backward(self,probs,y_i,x_i):
        y_onehot=np.zeros((self.W2.shape[0],1)) #(4,1)
        y_onehot[y_i]=1
        dL_dz2=probs-y_onehot
        self.dL_dW2=np.dot(dL_dz2,self.h1.T) #(4,1) (1,200) -> (4,200)
        self.dL_db2=dL_dz2 
        dL_dh1=np.dot(self.W2.T,dL_dz2) #W2=(4,200).T=(200,4) (4,1) -> (200,1)
        g_derivative=np.vectorize(lambda x: 1 if x>0 else 0)
        dL_dz1=dL_dh1*g_derivative(self.z1) #z1=(200,1)
        self.dL_dW1=np.dot(dL_dz1,np.expand_dims(x_i,axis=1).T) #(200,1) (1,784) -> (200,784)
        self.dL_db1=dL_dz1
    
    def update_weights(self,lr):
        self.W1-=lr*self.dL_dW1
        self.b1-=lr*self.dL_db1
        self.W2-=lr*self.dL_dW2
        self.b2-=lr*self.dL_db2


    def train_epoch(self, X, y, learning_rate=0.001):
        loss=0
        for _,(x_i,y_i) in enumerate(zip(X,y)):
            probs=self.forward(x_i) #(4,1)
            loss+=self.compute_loss(probs,y_i)
            self.backward(probs,y_i,x_i)
            self.update_weights(learning_rate)
        loss/=X.shape[0] 
        return np.squeeze(loss) [()]


def normalize(array_X):
    print("mean",np.mean(array_X,axis=0).shape)
    print("std",np.std(array_X,axis=0).shape)
    return array_X-np.mean(array_X,axis=0)/np.std(array_X,axis=0)

def plot(epochs, train_accs, val_accs,opt):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.savefig(f"images/q1/{opt.model}_lr_{opt.learning_rate}.png")
    #plt.show()
    

def plot_loss(epochs, loss,opt):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.savefig(f"images/q1/{opt.model}_loss_lr_{opt.learning_rate}.png")
    #plt.show()

#from sklearn.linear_model import LogisticRegression,Perceptron


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"] 
    # t=100
    # train_X,train_y=train_X[:t],train_y[:t]
    # dev_X,dev_y=dev_X[:t//10],dev_y[:t//10]
    # test_X,test_y=test_X[:t//10],test_y[:t//10]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    #Normalization
    # train_X=normalize(train_X)
    # dev_X=normalize(dev_X)
    # test_X=normalize(test_X)

    
    if opt.model == 'perceptron':
        model = PerceptronM(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegressionM(n_classes, n_feats,)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    # clf = LogisticRegression(fit_intercept=False, penalty=None,solver='newton-cholesky')
    # clf = Perceptron(random_state=42,max_iter=20)
    # clf.fit(train_X, train_y)
    # #print("LR train",clf.score(train_X, train_y))
    # #print("LR dev",clf.score(dev_X, dev_y))
    # #print("LR test",clf.score(test_X, test_y))
    
    
    for i in epochs:
        #print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        start_time=time.perf_counter()
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        print(f"Epoch took {round(time.perf_counter()-start_time,3)}s")
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f} \n'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    plot(epochs, train_accs, valid_accs,opt)
    if opt.model == 'mlp':
        with open("mlp_specs.pickle","wb") as file:
            result={"epochs":epochs,
                    "train_accs":train_accs,
                    "valid_accs":valid_accs,
                    "opt":opt,
                    "train_loss":train_loss,
                    "test_acc":model.evaluate(test_X, test_y),
            }
            #json.dump(result,file)
            pickle.dump(result,file)
        plot_loss(epochs, train_loss,opt)

if __name__ == '__main__':
    main()


import numpy as np


X = np.array([[1,1],[-1,1],[-1,1],[-1,-1]])
y=np.array([-1,1,1,-1])

W1=np.array([[1,1],[-1,-1]])
A=0
B=0
b1=np.array([-A,B]) #
W2=np.array([1,1])
b2=np.array([-1])

z1=np.dot(W1,X.T)+np.expand_dims(b1,axis=1) #cada coluna para cada exemplo
sign=np.vectorize(lambda x: 1 if x>=0 else -1)
tanh=np.vectorize(lambda x: np.tanh(5*x))
relu=np.vectorize(lambda x: x if x>=0 else 0)
print("z1",z1)
h1=tanh(z1)
print("h1",h1)
z2=np.dot(W2,h1)+np.expand_dims(b2,axis=1)  #W2 = 4,200 200,97477 -> 4,97477
output=tanh(z2) 
print(output)

#A=-1 B=1 or A=B=0
# X = np.array([[1,1],[-1.001,1],[-1,1],[-1,-1]])
# y=np.array([-1,1,1,-1])

# W1=np.array([[1,1],[-1,-1]])
# A=0
# B=1
# b1=np.array([-A,B]) #
# W2=np.array([1,1])
# b2=np.array([-2])

# z1=np.dot(W1,X.T)+np.expand_dims(b1,axis=1) #cada coluna para cada exemplo
# sign=np.vectorize(lambda x: 1 if x>=0 else -1)
# relu=np.vectorize(lambda x: x if x>=0 else 0)
# print("z1",z1)
# h1=sign(z1)
# print("h1",h1)
# z2=np.dot(W2,h1)+np.expand_dims(b2,axis=1)  #W2 = 4,200 200,97477 -> 4,97477
# output=sign(z2) 
# print(output)


# X = np.array([[1,1],[-1,1],[-1,1],[-1,-1]])
# y=np.array([-1,1,1,-1])

# W1=np.array([[1,1],[-1,-1]])
# A=-1
# B=1
# b1=np.array([-A,B]) #
# W2=np.array([-1,-1])
# b2=np.array([B-A])

# z1=np.dot(W1,X.T)+np.expand_dims(b1,axis=1) #cada coluna para cada exemplo
# sign=np.vectorize(lambda x: 1 if x>=0 else -1)
# relu=np.vectorize(lambda x: x if x>=0 else 0)
# print("z1",z1)
# h1=relu(z1)
# print("h1",h1)
# z2=np.dot(W2,h1)+np.expand_dims(b2,axis=1)  #W2 = 4,200 200,97477 -> 4,97477
# output=sign(z2) 
# print(output)

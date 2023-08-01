#note that the weights vector is of the form
# w = (b|w)^T

#y is vector of classes and x is data matrix

#We will implment the softmargin form of the SVM
#classification algorithm and optimise it using
#the PEGASOS algorithm

#This will be used to train m choose 2 seperating hyperplanes
#Between each of the classes of our dataset

#The decision tree algorithm will also be implemented so that
#We can get classification for multiple classes


import numpy as np


def SVM_Hinge_Loss(w,x,y,l):
    x = np.delete(x,0,1)
    zeros = np.zeros(len(y))
    return (np.sum(np.maximum(zeros,1-y*(x@w[1:]-w[0])))/len(y))+l*np.dot(w[1:],w[1:])

#Stopping conditions are maximum iteration and a loss tolerance, epsilon

def PEGASOS(x,y,lr,maxiter,epsilon):
    w = np.zeros(len(x[0]))
    order = np.arange(0,len(y),1)
    i=1

    while(i<=maxiter):
        eg = np.random.choice(order, size=1, replace=False)[0] #example we are selecting
        eta = 1/(lr*i)
        if (y[eg]*np.dot(w,x[eg])<1):
            w = (1-eta*lr)*w+eta*y[eg]*x[eg]
        else:
            w = (1-eta*lr)*w
        loss = SVM_Hinge_Loss(w,x,y,lr)
        print("Iteration:",i,"Loss:",loss)
        if(loss<epsilon):
            break
        i += 1
    
    return w
        
#now we implement the binary decision tree algorithm

def h_sum(m,i,j):
    h = 0
    for k in range(j+1):
        h += 2**(m-2-k)
    h -= (i+j-2)
    return h

def BDTree(x,H,m):
    j, l, h = 1, 2, 1
    for i in range(1,m):
        if np.dot(H[h],x)>=0:
            c = j
            l+=1
            h+=1
        else:
            c = l
            j=l
            l+=1
            h += h_sum(m,i,j)
    
    return c


#Now we test the algorithm on the iris dataset
from sklearn import datasets
#we define a function which takes two classes as input and outputs
#the data for both of them with the classes being 1 and -1
iris = datasets.load_iris()
X = iris.data
y = iris.target
#For iris dataset class labels take on the values, 0,1 and 2


def Data_Clean(X,y,c1,c2):
    X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
    c1_indexs = np.where(y==c1)
    c2_indexs = np.where(y==c2)
    rows = np.concatenate((c1_indexs[0],c2_indexs[0]),axis=0)
    X = X[rows, :]
    y = y[rows]
    y[y==c2]=-1
    y[y==c1]=1
    return X, y

X1, y1 = Data_Clean(X,y,0,1)

w = PEGASOS(X1,y1,0.005,10000,10**(-5))

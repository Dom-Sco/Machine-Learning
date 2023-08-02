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
        if (i%1000==1 or i==maxiter):
            print("Iteration:",i,"Loss:",loss)
        if(loss<epsilon):
            break
        i += 1
    
    return w
        
#now we implement the binary decision tree algorithm

def h_sum(m,i,j):
    h = 0
    
    for k in range(j-i):
        h += m-j-k
        
    return h

def BDTree(x,H,m):
    j, l, h = 0, 1, 0
    for i in range(1,m):
        print(h)
        if np.dot(H[h],x)>=1:
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
X2, y2 = Data_Clean(X,y,0,2)
X3, y3 = Data_Clean(X,y,1,2)

w1 = PEGASOS(X1,y1,0.005,40,10**(-5))
w2 = PEGASOS(X2,y2,0.005,40,10**(-5))
w3 = PEGASOS(X3,y3,0.005,40,10**(-5))

#Now we define a function for inference
def test(w,X,y):
    X = np.delete(X,0,1)
    out = X@w[1:]+w[0]
    out[out>=1] = 1
    out[out<=-1] = -1
    return out

inference1 = test(w1,X1,y1)
inference2 = test(w2,X2,y2)
inference3 = test(w3,X3,y3)

print("Accuracy for 0 and 1 is:",(sum(inference1==y1)/len(y1))*100)
print("Accuracy for 0 and 2 is:",(sum(inference2==y2)/len(y2))*100)
print("Accuracy for 1 and 2 is:",(sum(inference3==y3)/len(y3))*100)

H = [w1,w2,w3]
X_full = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)

predictions = []

for i in range(X.shape[0]):
    predictions.append(BDTree(X_full[i],H,3))

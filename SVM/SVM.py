#note that the weights vector is of the form
# w = (b|w)^T

#y is vector of classes and x is data matrix

#We will implment the softmargin form of the SVM
#classification algorithm and optimise it using
#the PEGASOS algorithm

#This will be used to train m choose 2 seperating hyperplanes
#Between each of the classes of our dataset

#To decide between multiple classes we will get the output of each of the m choose 2
#classifiers and take the class that appears the most as the prediction


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
        
#now we implement the prediction algorithm


def SVM_Prediction(x,H,m):
    j, l = 0, 1
    classes = []
    for i in range(len(H)):
        if np.dot(H[i],x)>=1:
            classes.append(j)
        else:
            classes.append(l)
        
        l+=1
        if (l==m):
            j+=1
            l = j+1
    
    return max(classes,key=classes.count)


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

w1 = PEGASOS(X1,y1,0.005,400,10**(-5))
w2 = PEGASOS(X2,y2,0.005,400,10**(-5))
w3 = PEGASOS(X3,y3,0.005,400,10**(-5))

H = [w1,w2,w3]
X_full = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)

predictions = []

for i in range(X.shape[0]):
    predictions.append(SVM_Prediction(X_full[i],H,3))
    
print("Overall accuracy is:",(sum(predictions==y)/len(y))*100,"%")

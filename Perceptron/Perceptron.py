import numpy as np

def perceptron(w, x, y, lr, maxiter):
    j = 0
    while (j < maxiter):
        misscnt = 0
        output = np.sign(x @ w)
        for i in range(len(output)):
            if (output[i]!=y[i]):
                w  = w + lr * y[i] * x[i]
                misscnt += 1 
        if (misscnt == 0):
            break
        j += 1 
    return [w, j, misscnt]
    
#for most logic functions    
w = np.array([0,0,0])
data = np.array([[1,0,0],
                 [1,0,1],
                 [1,1,0],
                 [1,1,1]])
#AND
y_and = np.array([-1,-1,-1,1]) #-1 if output is in lower, 1 if in upper
w_and = perceptron(w, data, y_and, 0.2, 100)
#OR
y_or = np.array([-1,1,1,1]) #-1 if output is in lower, 1 if in upper
w_or = perceptron(w, data, y_or, 0.2, 100)
#NAND
y_nand = np.array([1,1,1,-1]) #-1 if output is in lower, 1 if in upper
w_nand = perceptron(w, data, y_nand, 0.2, 100)
#NOR
y_nor = np.array([1,-1,-1,-1]) #-1 if output is in lower, 1 if in upper
w_nor = perceptron(w, data, y_nor, 0.2, 100)






#XOR
w = np.array([0,0,0,0])
data = np.array([[1,0,0],
                 [1,0,1],
                 [1,1,0],
                 [1,1,1]])
y = np.array([1,-1,-1,1]) #-1 if output is in lower, 1 if in upper

#Use kernel trick to make data linearly seperable
kernData = np.hstack((data,np.abs(data[:,1]-data[:,2]).reshape(4,1)))

w = perceptron(w, kernData, y, 0.2, 100)

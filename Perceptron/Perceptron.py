import numpy as np
import matplotlib.pyplot as plt


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

#plots for logic functions

fig, axs = plt.subplots(2, 2)

#AND
X1 = np.array([0,0,1])
Y1 = np.array([0,1,0])
X2 = np.array([1])
Y2 = np.array([1])
axs[0,0].scatter(X1,Y1, c = "blue", label="0")
axs[0,0].scatter(X2,Y2, c = "red", label="1")
axs[0,0].set_title("And Function")
axs[0,0].legend()

#OR
X1 = np.array([0])
Y1 = np.array([0])
X2 = np.array([0,1,1])
Y2 = np.array([1,0,1])
axs[0,1].scatter(X1,Y1, c = "blue", label="0")
axs[0,1].scatter(X2,Y2, c = "red", label="1")
axs[0,1].set_title("Or Function")
axs[0,1].legend()

#NAND
X2 = np.array([0,0,1])
Y2 = np.array([0,1,0])
X1 = np.array([1])
Y1 = np.array([1])
axs[1,0].scatter(X1,Y1, c = "blue", label="0")
axs[1,0].scatter(X2,Y2, c = "red", label="1")
axs[1,0].set_title("Nand Function")
axs[1,0].legend()

#NOR
X2 = np.array([0])
Y2 = np.array([0])
X1 = np.array([0,1,1])
Y1 = np.array([1,0,1])
axs[1,1].scatter(X1,Y1, c = "blue", label="0")
axs[1,1].scatter(X2,Y2, c = "red", label="1")
axs[1,1].set_title("Nor Function")
axs[1,1].legend()
plt.show()

#plots with decision boundary

fig, axs = plt.subplots(2, 2)

#AND
X1 = np.array([0,0,1])
Y1 = np.array([0,1,0])
X2 = np.array([1])
Y2 = np.array([1])
axs[0,0].set_xlim(-0.5, 1.5), axs[0,0].set_ylim(-0.5, 1.5)
axs[0,0].scatter(X1,Y1, c = "blue", label="0")
axs[0,0].scatter(X2,Y2, c = "red", label="1")
axs[0,0].axline((0,(-w_and[0][0]-w_and[0][1]*0)/w_and[0][2]) ,(1 ,(-w_and[0][0]-w_and[0][1]*1)/w_and[0][2]))
axs[0,0].set_title("And Function")
axs[0,0].legend()

#OR
X1 = np.array([0])
Y1 = np.array([0])
X2 = np.array([0,1,1])
Y2 = np.array([1,0,1])
axs[0,1].set_xlim(-0.5, 1.5), axs[0,1].set_ylim(-0.5, 1.5)
axs[0,1].scatter(X1,Y1, c = "blue", label="0")
axs[0,1].scatter(X2,Y2, c = "red", label="1")
axs[0,1].axline((0,(-w_or[0][0]-w_or[0][1]*0)/w_or[0][2]) ,(1 ,(-w_or[0][0]-w_or[0][1]*1)/w_or[0][2]))
axs[0,1].set_title("Or Function")
axs[0,1].legend()

#NAND
X2 = np.array([0,0,1])
Y2 = np.array([0,1,0])
X1 = np.array([1])
Y1 = np.array([1])
axs[1,0].set_xlim(-0.5, 1.5), axs[1,0].set_ylim(-0.5, 1.5)
axs[1,0].scatter(X1,Y1, c = "blue", label="0")
axs[1,0].scatter(X2,Y2, c = "red", label="1")
axs[1,0].axline((0,(-w_nand[0][0]-w_nand[0][1]*0)/w_nand[0][2]) ,(1 ,(-w_nand[0][0]-w_nand[0][1]*1)/w_nand[0][2]))
axs[1,0].set_title("Nand Function")
axs[1,0].legend()

#NOR
X2 = np.array([0])
Y2 = np.array([0])
X1 = np.array([0,1,1])
Y1 = np.array([1,0,1])
axs[1,1].set_xlim(-0.5, 1.5), axs[1,1].set_ylim(-0.5, 1.5)
axs[1,1].scatter(X1,Y1, c = "blue", label="0")
axs[1,1].scatter(X2,Y2, c = "red", label="1")
axs[1,1].axline((0,(-w_nor[0][0]-w_nor[0][1]*0)/w_nor[0][2]) ,(1 ,(-w_nor[0][0]-w_nor[0][1]*1)/w_nor[0][2]))
axs[1,1].set_title("Nor Function")
axs[1,1].legend()
plt.show()



#XOR
fig = plt.figure()
X1 = np.array([0,1])
Y1 = np.array([0,1])
X2 = np.array([0,1])
Y2 = np.array([1,0])
plt.scatter(X1,Y1, c = "blue", label="0")
plt.scatter(X2,Y2, c = "red", label="1")
plt.title("Xor Function")
plt.legend()
plt.show()


fig = plt.figure()
ax = plt.axes(projection ='3d')
(X, Y) = np.meshgrid(np.linspace(-0.5,1.5,200),np.linspace(-0.5,1.5,200))
Z = -(w[0][0]+w[0][1]*X+w[0][2]*Y)/w[0][3];
ax.scatter(kernData[:,1][1:3], kernData[:,2][1:3], kernData[:,3][1:3], c = "red", label = "1")
ax.scatter(np.array([kernData[:,1][0],kernData[:,1][3]]),np.array([kernData[:,2][0],kernData[:,2][3]]),np.array([kernData[:,3][0],kernData[:,3][3]]), c = "green", label = "0")
ax.plot_surface(X, Y, Z)
plt.legend()
plt.title("Xor with Kernel Trick")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

#Defining loss functions and activation functions

def SE_loss(y,g):
    return (g - y)**2 , 2 * (g - y)

def RELU(z, l, L):
    if l == L: 
        return z, np. ones_like (z)
    else:
        val = np. maximum (0,z) # RELU function element -wise
        J = np.array(z>0, dtype = float ) # derivative
    return val, J


#MLP class takes layers as input which is a list that contains the number
#of "neurons" in each layer. It also takes act_fns which is a list of activation
#functions for each layer

class MLP:
    def __init__(self, layers, act_fns):
        self.layers = layers
        self.act_fns = act_fns
    
    #initialise takes self.layers and uses that to produce
    #initial weights and biases via drawing from standard
    #normal distribution
    def initialise(self, w_sig=1):
        W, b = [[]]* len(self.layers), [[]]* len(self.layers)
        for l in range (1, len(self.layers)):
            W[l]= w_sig * np.random.randn(self.layers[l], self.layers[l -1])
            b[l]= w_sig * np.random.randn(self.layers[l], 1)
        
        return W,b
    
    #Forward pass on the network
    def forward(self, x, W, b):
        p = len(self.layers)
        
        a, z, gr= [0]*p, [0]*p, [0]*p
        a[0] = x.reshape(-1,1)
        
        for l in range(1, p):
            z[l] = W[l] @ a[l-1] + b[l]
            a[l], gr[l] = eval(self.act_fns[l])(z[l], l, p-1)
        
        return a, z, gr
    
    #Backward pass on the network
    def backward(self, W, b, X, y):
        n = len(y)
        L = len(self.layers)-1
        delta = [0] * len(self.layers)
        dC_db, dC_dW = [0] * len(self.layers), [0] * len(self.layers)
        loss = 0
        
        for i in range(n): #looping over training examples
            a, z, gr = self.forward(X[i,:].T, W, b)
            cost , gr_C = SE_loss(y[i], a[L])
            loss += cost/n
            delta [L] = gr[L] @ gr_C
            
            for l in range (L ,0 , -1): # l = L ,... ,1
                dCi_dbl = delta [l]
                dCi_dWl = delta [l] @ a[l-1].T
                # ---- sum up over samples ----
                dC_db [l] = dC_db [l] + dCi_dbl /n
                dC_dW [l] = dC_dW [l] + dCi_dWl /n
                # -----------------------------
                delta[l-1] = gr[l-1] * W[l].T @ delta [l]
        
        return dC_dW , dC_db , loss
    
    
    def list2vec(self, W,b):
        # converts list of weight matrices and bias vectors into
        # one column vector
        b_stack = np. vstack ([b[i] for i in range (1, len(b))] )
        W_stack = np. vstack (W[i]. flatten (). reshape ( -1 ,1) for i in range(1, len(W)))
        vec = np. vstack ([ b_stack , W_stack ])
        return vec

    def vec2list(self, vec):
        p = self.layers
        # converts vector to weight matrices and bias vectors
        W, b = [[]]* len(p) ,[[]]* len(p)
        p_count = 0
        for l in range (1, len(p)): # construct bias vectors
            b[l] = vec[ p_count :( p_count +p[l]) ]. reshape ( -1 ,1)
            p_count = p_count + p[l]
        
        for l in range (1, len(p)): # construct weight matrices
            W[l] = vec[ p_count :( p_count + p[l]*p[l -1]) ]. reshape (p[l], p[l-1])
            p_count = p_count + (p[l]*p[l -1])
    
        return W, b
    
    def train(self, batch_size, num_epochs, lr, W, b, X, y):
        beta = self.list2vec(W,b)
        loss_arr = []
        n = len(X)
        print (" epoch | batch loss")
        print (" ----------------------------")
        for epoch in range (1, num_epochs +1):
            batch_idx = np.random.choice(n, batch_size)
            batch_X = X[batch_idx].reshape(-1 ,1)
            batch_y = y[batch_idx].reshape(-1 ,1)
            dC_dW, dC_db, loss =  self.backward(W, b, batch_X , batch_y)
            d_beta = self.list2vec(dC_dW , dC_db)
            loss_arr.append(loss.flatten()[0])
            if(epoch ==1 or np.mod(epoch ,1000) ==0):
                print (epoch ,": ",loss. flatten () [0])
            beta = beta - lr*d_beta #gradient descent
            W, b = self.vec2list(beta)
        
        dC_dW , dC_db , loss = self.backward(W,b,X,y)
        print (" entire training set loss = ",loss. flatten () [0])
        
        return W, b, loss_arr



data = np.genfromtxt('polyreg.csv',delimiter =',')
X = data [: ,0]. reshape ( -1 ,1)
y = data [: ,1]. reshape ( -1 ,1)
layers = [1,5,7,1]
act_fns = ["","RELU","RELU","RELU"]
n = MLP(layers, act_fns)
W, b = n.initialise()
W_f, b_f, loss_arr = n.train(20, 10000, 0.005, W, b, X, y)



xx = np. arange (0 ,1 ,0.01)
y_preds = np. zeros_like (xx)
L = len(n.layers)-1
for i in range (len(xx)):
    a, _, _ =  n.forward (xx[i],W_f,b_f)
    y_preds [i], = a[L]
    
plt.plot(X,y, 'r.', markersize = 4, label = 'y')
plt.plot(np. array (xx), y_preds , 'b',label = 'fit ')
plt. legend ()
plt. xlabel ('x')
plt. ylabel ('y')
plt.show ()

import numpy as np

class MLP:
    def __init__(self, layers, lr = 0.1):
        self.W = []
        self.layers = layers
        self.lr = lr
        
        for i in np.arange(0, len(layers) -2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
        
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))
        
    def __repr__(self):
        # constrruct and return a string representing the network architecture
        return "MLP: {}".format(
            "-".join(str(l) for l in self.layers))
    
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x * (1 - x)
    
    def fit(self, X, y, epochs=1000, displayUpdate=100):
        #To introduce the bias term as a trainable paramater
        #insert a column of ones as the last column in the
        #feature matrix
        X = np.c_[X, np.ones((X.shape[0]))]
        
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[Info] epoch={}, loss={:.7f}".format(
                    epoch + 1, loss))
    
    def fit_partial(self, x, y):
        #store output of each activation as x is
        #forward propagated
        A = [np.atleast_2d(x)]
        
        #Forward Propagation
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)
            
        #Backward Propagation
        error = A[-1] - y
        D = [error * self.sigmoid_deriv(A[-1])]
        
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        #reverse deltas as lopped over layers
        #in reverse order
        D = D[::-1]
        
        #Weight Update
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.lr * A[layer].T.dot(D[layer])
    
    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        
        return p
    
    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets)**2)
        return loss
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

class PMM:
    def __init__(self, counts, maxiter=10):
        self.maxiter = maxiter
        self.counts = counts
    
    #initialise parameters
    def initialise(self):
        ratios = self.counts[:-1]/self.counts[1:]
        maximums = []
        for i in range(1,len(ratios)):
            if ratios[i]>=1 and ratios[i-1]<=1:
                maximums.append(ratios[i])
        self.g = len(maximums)
        self.l = (ratios[:, None] == maximums).argmax(axis=0)
        
        landmax = np.append(self.l, len(self.counts))
        vec = np.array([(x + landmax[i])/2 for i, x in enumerate(landmax) if i > 0])
        self.pi = vec / np.sum(vec)
        
        data = []
        
        for i in range(len(counts)):
            for j in range(counts[i]):
                data.append(i)
                
        data = np.array(data)
        self.data = data
        
    #Poisson distribution
    def poi(self, x):
        results = (((self.l**x)*np.exp(-self.l))/factorial(x))
        return np.array(results)

    #Poisson mixture model (returns all frequencies for given model parameters)
    def PMM(self):
        freq = []
        for i in range(len(self.counts)):
            freq.append(np.sum(self.pi*self.poi(i)))
        return np.array(freq)

    #posterior probabilities
    def tau(self):
        taus = []
        for i in range(len(self.data)):
            mix = self.pi*self.poi(self.data[i])
            taus.append(mix/np.sum(mix))
        return np.array(taus)
    
    def update(self):
        taus = self.tau()
        tausum = np.sum(taus, axis=0)
        tausumwdata = np.sum(taus*np.tile(self.data, (np.shape(taus)[1], 1)).T, axis=0)
        self.pi = tausum / len(self.data)
        self.l = tausumwdata / tausum
        
    def fit(self):
        self.initialise()
        for i in range(self.maxiter):
            self.update()
    
    def graph(self):
        freq = self.counts/np.sum(self.counts)
        model = self.PMM()
        xaxis = np.arange(0, len(self.counts))
        plt.plot(xaxis, freq, 'o', label="Data")
        plt.plot(xaxis, model, 'o', color='red', label="Model")
        title = "Mixture of " + str(self.g) + " Poissons"
        plt.title(title)
        plt.xlabel("Events in Fixed Time Interval")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
        
        
        
        
        
        
#Made up data
counts = np.array([162,267,271,185,140,122, 104, 125, 143,167,140, 122, 98,42,8,1])

#Model fitting
m = PMM(counts)
m.initialise()
print("Theta 0:", m.pi,m.l)
m.fit()
m.graph()
print("Theta 10:", m.pi,m.l)

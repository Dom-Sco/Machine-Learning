import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

#Poisson distribution
def poi(l,x):
    results = []
    for i in range(len(x)):
        results.append((((l**x[i])*np.exp(-l))/factorial(x[i])))
    return np.array(results)

#Poisson mixture model
def PMM(pi, l, x):
    return np.sum(pi*poi(l,x))

#posterior probabilities
def tau(g, pi, l, x):
    taus = []
    for j in range(g):
        taus.append(((pi[j]*poi(l[j],x))/np.sum(pi*poi(l,x))))
    return np.array(taus)

def MMpoiUpdate(g, pi, l, data):
    n = len(data)
    taus = tau(g, pi, l, data)
    pi = []
    l = []
    for i in range(g-1):
        pi.append(np.sum(taus[i])/n)
        l.append(np.sum(taus[i]*data)/np.sum(taus[i]))
    pi.append(1-np.sum(pi))
    l.append(np.sum(taus[g-1]*data)/np.sum(taus[g-1]))
    return [np.array(pi),np.array(l)]
    
    
#MM algorithm for PMM
def MMpoi(data, pi, l, maxiter, g):
    updates = [[pi, l]]
    for i in range(maxiter):
        updates.append(MMpoiUpdate(g, pi, l, data))
    return updates


#data
counts = np.array([162,267,271,185,111,61,27,8,3,1])

data = []

for i in range(len(counts)):
    for j in range(counts[i]):
        data.append(i)
        
data = np.array(data)

#parameter learning
pi = np.array([0.8,0.2])
l = np.array([1.5, 2.3])

parameters = MMpoi(data, pi, l, 100, 2)
pi = parameters[-1][0]
l = parameters[-1][1]

#plotting
freq = counts/np.sum(counts)
model = []
xaxis = []
for i in range(len(counts)):
    xaxis.append(i)
    model.append(PMM(pi,l,[i]))

plt.plot(xaxis, freq, 'o', label="Data")
plt.plot(xaxis, model, 'o', color='red', label="Model")
plt.title("Mixture of Two Poissons")
plt.legend()
plt.show()

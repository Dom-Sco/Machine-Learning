from MLP import MLP
import numpy as np

#XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0], [1],[1], [0]])

#train
nn = MLP([2,2,1], lr = 0.5)
nn.fit(X, y, epochs=20000)

#test

for (x, target) in zip(X, y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[Info] data={}, ground-truth={}, pred={:.4f}, step={}".format(
        x, target[0], pred, step))
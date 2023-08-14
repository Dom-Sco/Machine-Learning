# based on the implementations:
#https://gist.github.com/karpathy/d4dee566867f8291f086
#https://github.com/javaidnabi31/RNN-from-scratch/blob/master/RNN_char_text%20generator.ipynb
# I use Adam instead of adagrad for training


import numpy as np


class DataReader:
    def __init__(self, path, seq_length):
        #uncomment below , if you dont want to use any file for text reading and comment next 2 lines
        #self.data = "some really long text to test this. maybe not perfect but should get you going."
        self.fp = open(path, "r")
        self.data = self.fp.read()
        #find unique chars
        chars = list(set(self.data))
        #create dictionary mapping for each char
        self.char_to_ix = {ch:i for (i,ch) in enumerate(chars)}
        self.ix_to_char = {i:ch for (i,ch) in enumerate(chars)}
        #total data
        self.data_size = len(self.data)
        #num of unique chars
        self.vocab_size = len(chars)
        self.pointer = 0
        self.seq_length = seq_length

    def next_batch(self):
        input_start = self.pointer
        input_end = self.pointer + self.seq_length
        inputs = [self.char_to_ix[ch] for ch in self.data[input_start:input_end]]
        targets = [self.char_to_ix[ch] for ch in self.data[input_start+1:input_end+1]]
        self.pointer += self.seq_length
        if self.pointer + self.seq_length + 1 >= self.data_size:
            # reset pointer
            self.pointer = 0
        return inputs, targets

    def just_started(self):
        return self.pointer == 0

    def close(self):
        self.fp.close()




class RNN:
    def __init__(self, hidden_size, vocab_size, seq_length, lr):
        # hyper parameters
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.lr = lr
        # model parameters
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.01 # input to hidden
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01 # hidden to hidden
        self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01 # hidden to output
        self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
        self.by = np.zeros((self.vocab_size, 1)) # output bias
    
    def forward(self, inputs, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size,1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t-1] + self.bh) # hidden state
            ys[t] = self.Why @ hs[t] + self.by # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t] - np.max(ys[t]))/np.sum(np.exp(ys[t] - np.max(ys[t]))) # softmax
        return xs, hs, ps
    
    def backward(self, xs, hs, ps, targets):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(self.seq_length)):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop into y
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        return dWxh, dWhh, dWhy, dbh, dby
    
    def loss(self, ps, targets):
        return sum(-np.log(ps[t][targets[t],0]) for t in range(self.seq_length)) # cross-entropy loss function
    
    def list2vec(self, W,b):
        # converts list of weight matrices and bias vectors into
        # one column vector
        b_stack = np. vstack ([b[i] for i in range (0, len(b))] )
        W_stack = np. vstack (W[i]. flatten (). reshape ( -1 ,1) for i in range(0, len(W)))
        vec = np. vstack ([ b_stack , W_stack ])
        return vec

    def vec2list(self, vec):
        bh = vec[0:len(self.bh)]
        by_end = len(self.bh)+len(self.by)
        by = vec[len(self.bh):by_end]
        Wxh_end = by_end + (self.hidden_size * self.vocab_size)
        Wxh = vec[by_end:Wxh_end].reshape(self.hidden_size, self.vocab_size)
        Whh_end = Wxh_end + (self.hidden_size * self.hidden_size)
        Whh = vec[Wxh_end:Whh_end].reshape(self.hidden_size, self.hidden_size)
        Why_end = Whh_end + (self.vocab_size * self.hidden_size)
        Why = vec[Whh_end:Why_end].reshape(self.vocab_size, self.hidden_size)
        return Wxh, Whh, Why, bh, by
        
    
    def Adam(self, dW, db, b1, b2, m, v, epsilon, t):
        theta = self.list2vec([self.Wxh, self.Whh, self.Why],[self.bh,self.by])
        g = self.list2vec(dW, db)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*(g**2)
        m_hat = m/(1-b1**t)
        v_hat = v/(1-b2**t)
        theta = theta - (self.lr*m_hat)/(np.sqrt(v_hat)+epsilon)
        Wxh, Whh, Why, bh, by = self.vec2list(theta)
        return Wxh, Whh, Why, bh, by
    
    
    def sample(self, h, seed_ix, n):
           """
           sample a sequence of integers from the model
           h is memory state, seed_ix is seed letter from the first time step
           """
           x = np.zeros((self.vocab_size, 1))
           x[seed_ix] = 1
           ixes = []
           for t in range(n):
               h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
               y = self.Why @ h + self.by
               p = np.exp(y)/np.sum(np.exp(y))
               ix = np.random.choice(range(self.vocab_size), p = p.ravel())
               x = np.zeros((self.vocab_size,1))
               x[ix] = 1
               ixes.append(ix)
           return ixes
        
    def train(self, threshold, maxiter, b1, b2, epsilon, data_reader):
        m = 0
        v = 0
        t = 1
        loss = 1000
        hprev = np.zeros((self.hidden_size,1))
        while (loss > threshold and t<=maxiter):
            if data_reader.just_started():
                    hprev = np.zeros((self.hidden_size,1))
            inputs, targets = data_reader.next_batch()
            xs, hs, ps = self.forward(inputs, hprev)
            dWxh, dWhh, dWhy, dbh, dby = self.backward(xs, hs, ps, targets)
            dW = [dWxh, dWhh, dWhy]
            db = [dbh, dby]
            self.Wxh, self.Whh, self.Why, self.bh, self.by  = self.Adam(dW, db, b1, b2, m, v, epsilon, t)
            t += 1
            loss = self.loss(ps, targets)
            if t%1000==0:
                print("Iteration:",t,"Loss:",loss)
    
    def predict(self, data_reader, start, n):

        #initialize input vector
        x = np.zeros((self.vocab_size, 1))
        chars = [ch for ch in start]
        ixes = []
        for i in range(len(chars)):
            ix = data_reader.char_to_ix[chars[i]]
            x[ix] = 1
            ixes.append(ix)

        h = np.zeros((self.hidden_size,1))
        # predict next n chars
        for t in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            p = np.exp(y)/np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p = p.ravel())
            x = np.zeros((self.vocab_size,1))
            x[ix] = 1
            ixes.append(ix)
        txt = ''.join(data_reader.ix_to_char[i] for i in ixes)
        return txt
    
path = "warpeace.txt"
seq_length = 30
data_reader = DataReader(path, seq_length)
rnn = RNN(hidden_size=50, vocab_size=data_reader.vocab_size,seq_length=seq_length,lr=1e-1)
rnn.train(0.01, 1000, 0.9, 0.999, 1e-8, data_reader)
print(rnn.predict(data_reader, 'war', 101))
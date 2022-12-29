import numpy as np
import math
import matplotlib.pyplot as plt


def plot_gallery(images, titles=None, xscale=1, yscale=1, nrow=3, cmap='gray', output=None):
    ncol = math.ceil(len(images) / nrow)
    
    plt.figure(figsize=(xscale * ncol, yscale * nrow))

    for i in range(nrow * ncol):
        plt.subplot(nrow, ncol, i + 1)
        if i < len(images):
            plt.imshow(images[i], cmap=cmap)
            if titles is not None:
                # use size and y to adjust font size and position of title
                plt.title(titles[i], size=12, y=1)
        plt.xticks(())
        plt.yticks(())

    plt.tight_layout()

    if output is not None:
        plt.savefig(output)
    plt.show()

#Binarise images
def binarise(image):
    return 2*(image > np.mean(image))-1


class HopfieldNet:
    def __init__(self):
        pass
    def train(self, images):
        """ assume each image is binary with values -1 and 1"""
        self.num_neuron = np.prod(images[0].shape)
        self.shape = images[0].shape
        self.weights = np.zeros((self.num_neuron, self.num_neuron))
        for image in images:
            v = image.flatten()
            self.weights += np.outer(v, v)
        # zero diagonal entries
        self.weights = self.weights - np.diag(np.diag(self.weights))
    def predict(self, image, sync=True, max_iter=100, verbose=False):
        v = image.flatten()
        converged = False
        for i in range(max_iter):
            if sync: # synchronous update
                v_new = np.sign(self.weights @ v)
            if np.all(v_new == v):
                converged = True
                break
            else: # semi-random update
                perm = np.random.permutation(len(v))
                v_new = np.copy(v).astype(float)
            for j in perm:
                v_new[j] = np.sign(self.weights[j] @ v_new)
                if np.all(v_new == v):
                    converged = True
                    break
            v = v_new
        if verbose:
            if converged:
                print('converged in %d iteration(s)' % (i+1))
            else:
                print('update did not converge in %d iteration(s)' % (i+1))
        return v.reshape(image.shape)

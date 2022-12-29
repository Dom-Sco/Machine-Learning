from HopfieldNetwork import *
import skimage.io

images = [skimage.io.imread('supplements/image%d.png' % i) for i in range(1,4)]
for i in range(len(images)):
    print('Image %d: size = %dx%d; mean pixel value = %5.5f' %
          (i+1, images[i].shape[0], images[i].shape[1], images[i].mean()))

images_binary = [binarise(image) for image in images]

net = HopfieldNet()
net.train(images_binary)


images_noisy = [skimage.io.imread('supplements/noisy%d.png' % i) for i in range(1, 4)]
images_reconstructed = [(net.predict(binarise(image), sync=True, max_iter=100, verbose=True)+1)/2 for image in images_noisy]

plot_gallery(images_noisy, nrow=1, xscale=2, yscale=2)
plot_gallery(images_reconstructed, nrow=1, xscale=2, yscale=2)
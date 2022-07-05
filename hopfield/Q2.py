# Q2_graded
# Do not change the above line.

# Remove this comment and type your codes here

import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import math
from PIL import Image, ImageFont
import random
from skimage.metrics import structural_similarity as ssim

# Q2_graded
# Do not change the above line.

from tensorflow import keras
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255

mnist_dict = {
              0:[],
              1:[],
              2:[],
              3:[],
              4:[],
              5:[],
              6:[],
              7:[],
              8:[],
              9:[]
             }

for i in range(len(y_train)):
    mnist_dict[y_train[i]].append(i)

# Q2_graded
# Do not change the above line.
 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets


class Hopfield:
    def __init__(self, x_train, bias):
        self.bias = bias
        train = np.array(x_train)
        self.dimension = train.shape
        self.Weights = np.zeros((self.dimension[1], self.dimension[1]))

        #normalize data and weight matrix
        mean = np.sum([np.sum(t) for t in train]) / (len(train) * self.dimension[1])
        for sample in train:
            self.Weights += np.outer(sample - mean, sample - mean)
        np.fill_diagonal(self.Weights, 0)
        self.Weights /= len(train)

    def activation_function(self, x):
        return np.sign(self.Weights.dot(x) - self.bias)

#turn image into bipolar pattern
def make_pattern(size):
    perm = np.random.permutation(x_train)
    train = [perm[i] for i in range(size)]
    train = [np.sign(t.reshape(-1) * 2 - 1) for t in train]
    return train


def add_noise(picture, noise_percent):   
    picture = picture.reshape((10,-1))

    for i in range(picture.shape[0]):
        pixel = np.random.binomial(1, noise_percent, picture.shape[1])
        for j in range(picture.shape[1]):
            if pixel[j] == 1:
                picture[i][j] *= -1
    return picture

def mse(imageA, imageB):

	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err


sizes = [6, 600, 60000]
noises = [0.1, 0.3, 0.6]


for i in range(3):

    train = make_pattern(sizes[i])
    print('Network size {}'.format(sizes[i]))

    for j in range(3):
        indexes = []
        for i in range(10):
            indexes += random.sample(mnist_dict[i], 1)

        sample_classes = x_train[indexes]
        pattern_data = np.array([np.sign(t.reshape(-1) * 2 - 1) for t in sample_classes])
        noisy_data = add_noise(pattern_data,noises[j])
        print('Noise percent {}'.format(noises[j]))

        fig, axs = plt.subplots(2, 10, figsize=(50,10))
        count_x = 0
        count_y = 0

        for noisy in noisy_data:
            axs[count_x, count_y].imshow(noisy.reshape(28, 28))
            count_y += 1

        model = Hopfield(train, bias=80)

        count_x += 1
        count_y = 0
        ims = []
        predict = []

        for d in noisy_data:
            data = model.activation_function(d)
            predict.append(data)

        predict=np.array(predict)    
        
        for i in range (predict.shape[0]):
            m = mse(pattern_data[i].reshape(28, 28), predict[i].reshape(28, 28))
            s = ssim(pattern_data[i].reshape(28, 28), predict[i].reshape(28, 28))
            axs[count_x, count_y].imshow(predict[i].reshape(28, 28))
            print(("MSE: %.2f, SSIM: %.2f" % (m, s)))
            count_y += 1
        fig.tight_layout()


# Q1_graded
# Do not change the above line.

import matplotlib.pylab as plt
import numpy as np
import gzip
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import tensorflow as tf
import random

# Q1_graded
# Do not change the above line.

# Type your code here
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train/255
unique, counts = np.unique(y_train, return_counts=True)
#How many of each number we have
result = np.column_stack((unique, counts)) 

#Creating a dictionary for keeping track of our numbers
mnist_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

for i in range(len(y_train)):
    mnist_dict[y_train[i]].append(i)

#We want 5000 numbers so we pick 500 of each number
numbers = []
for i in range(10):
    numbers += random.sample(mnist_dict[i], 500)

x_train = x_train[numbers].reshape(5000,-1)
print(x_train.shape)


# Q1_graded
# Do not change the above line.

# Type your code here

#Setting the parameters
np.random.seed(0)

learning_rate = 0.1
epochs = 100
radius = 1


grid_shape = (20,20)
neurons = 400
size = int(np.sqrt(neurons))

weights = np.random.rand(neurons, 784) 

def get_distance(x, index):
    i,j = np.indices(x, sparse=True)
    return np.sqrt((i-index[0])**2 + (j-index[1])**2)

def convert_shape(x):
    i = (int)(x/size)
    j = x % size
    return i, j

def plot_numbers(weights):
    grid = np.zeros((20 * 28, 20 * 28))
    for i in range(len(weights)):
        image = weights[i].reshape(28, 28)
        grid[(i // 20) * 28: ((i // 20) + 1) * 28,
        (i % 20) * 28: ((i % 20) + 1) * 28] = image

    plt.xticks([])
    plt.yticks([])  
    plt.imshow(grid, cmap="gray")
    plt.show()    

for epoch in range(epochs):
  if epoch%10 == 0:
      plot_numbers(weights) 

  for num in range(10):

      batch = x_train[np.random.choice(len(x_train), size=128, replace=False)]
      
      for data in batch:
          # Find the Best Matching Unit (BMU) by their Euclidean distances
          BMU = np.argmin(np.sum((weights - data) ** 2, axis=1))  
          #1D -> 2D
          BMU_index = convert_shape(BMU)
          #Compute distance to BMU
          distances = get_distance(grid_shape, BMU_index)  
          #Flatten
          distances = distances.reshape(-1, 1)  
          distances = (-1 / (2 * radius ** 2)) * distances
          #Neighbourhood function
          Neighbourhood = np.exp(distances)  
          #Upadting weights
          weights = weights + learning_rate * (Neighbourhood * (data - weights))
  learning_rate *= 0.99        
  radius *= 0.99




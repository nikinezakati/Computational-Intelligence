# Q3_graded
# Do not change the above line.

# Remove this comment and type your codes here

import pandas as pd
import numpy as np

def read_csv(filename):
  
    with open(filename) as f:
        cities = pd.read_csv(
            f,
            skiprows=1,
            sep=' ',
            names=['city', 'y', 'x'],
            dtype={'city': str, 'x': np.float64, 'y': np.float64},
            header=None,
            nrows=195
        )
        return cities

def normalize(points):
    """
    Normalizing each dimension by removing the
    initial offset and normalizing the points in a proportional interval: [0,1]
    on y, maintining the original ratio on x.
    """
    ratio = (points.x.max() - points.x.min()) / (points.y.max() - points.y.min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    return norm.apply(lambda p: ratio * p, axis=1)

# Q3_graded
# Do not change the above line.

def euclidean_distance(x, y):
    """Return the array of distances of two numpy arrays of points."""
    return np.linalg.norm(x - y, axis=1)

def select_closest(weights, x):
    """Return the index of the closest point to a given point."""
    return euclidean_distance(weights,x).argmin()

def route_distance(cities):
    """Return the cost of traversing a route of cities in a certain order."""
    points = cities[['x', 'y']]
    distances = euclidean_distance(points, np.roll(points, 1, axis=0))
    return np.sum(distances)

# Q3_graded
# Do not change the above line.

import numpy as np

def generate_network(size):
    """
    Generate a neuron network of a given size.
    Return a vector of two dimensional points in the interval [0,1].
    """
    return np.random.rand(size, 2)

def gaussian_neighborhood(center, radix, weights):
    """Get the range gaussian of given radix around a center index."""

    # Impose an upper bound on the radix to prevent NaN and blocks
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    deltas = np.absolute(center - np.arange(weights))
    distances = np.minimum(deltas, weights - deltas)

    # Compute Gaussian distribution around the given center
    return np.exp(-(distances*distances) / (2*(radix*radix)))

def get_route(cities, weights):
    """Return the route computed by a network."""
    cities['winner'] = cities[['x', 'y']].apply(
        lambda c: select_closest(weights, c),
        axis=1, raw=True)

    return cities.sort_values('winner').index

# Q3_graded
# Do not change the above line.

import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_network(cities, neurons, name='diagram.png', ax=None):
    
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        fig = plt.figure(figsize=(5, 5), frameon = False)
        axis = fig.add_axes([0,0,1,1])

        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        axis.scatter(cities['x'], cities['y'], color='red', s=4)
        axis.plot(neurons[:,0], neurons[:,1], 'r.', ls='-', color='#0063ba', markersize=2)

        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.show()
        plt.close()

    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=4)
        ax.plot(neurons[:,0], neurons[:,1], 'r.', ls='-', color='#0063ba', markersize=2)
        return ax

def plot_route(cities, route, name='diagram.png', ax=None):
    
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        fig = plt.figure(figsize=(5, 5), frameon = False)
        axis = fig.add_axes([0,0,1,1])

        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        axis.scatter(cities['x'], cities['y'], color='red', s=4)
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        axis.plot(route['x'], route['y'], color='purple', linewidth=1)

        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.show()
        plt.close()

    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=4)
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        ax.plot(route['x'], route['y'], color='purple', linewidth=1)
        return ax

# Q3_graded
# Do not change the above line.

def som(problem, iterations, learning_rate=0.8):

    cities = problem.copy()
    cities[['x', 'y']] = normalize(cities[['x', 'y']])

    # The population size is 8 times the number of cities
    n = cities.shape[0] * 8

    # Generate an adequate network of neurons:
    weights = generate_network(n)
    print('Network of {} neurons created. Starting the iterations:'.format(n))

    for i in range(iterations):

        # Choose a random city
        city = cities.sample(1)[['x', 'y']].values

        # Find the index of the closest city to it
        winner_idx = select_closest(weights, city)

        # Generate a filter that applies changes to the winner's gaussian
        gaussian = gaussian_neighborhood(winner_idx, n//10, weights.shape[0])

        # Update the network's weights 
        weights += gaussian[:,np.newaxis] * learning_rate * (city - weights)
        
        # Decay the variables
        learning_rate = learning_rate * 0.99997
        n = n * 0.9997

        # Check for plotting interval
        if not i % 5555:
              plot_network(cities, weights, name='{:05d}.png'.format(i))

        # Check if any parameter has completely decayed.
        if n < 1:
            print('Radius has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            print('Learning rate has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break
    else:
        print('Completed {} iterations.'.format(iterations))

    plot_network(cities, weights, name='final.png')

    route = get_route(cities, weights)
    plot_route(cities, route, 'route.png')
    return route


# Q3_graded
# Do not change the above line.

from google.colab import files
import pandas as pd

uploaded = files.upload()

# Q3_graded
# Do not change the above line.

problem = read_csv('Cities.csv')
route = som(problem, 100000)
problem = problem.reindex(route)
distance = route_distance(problem)
print('Route found of length {}'.format(distance))


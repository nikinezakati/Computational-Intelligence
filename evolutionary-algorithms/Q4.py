# Q4_graded
# Do not change the above line.

import random

population = []

def initialize_population(pop_size=1000):
    global population
    population = []
    for i in range(pop_size):
        population.append(random.uniform(-10, 10))

def fitness(x):
    return abs(168 * x ** 3 - 7.22 * x ** 2 + 15.5 * x - 13.2)


def sort_by_fitness():
    global population
    population.sort(key=fitness)


def reproduce(parent1, parent2, range=10):
    midpoint = random.randrange(1, range)
    child1 = (parent1 * midpoint + parent2 * (range - midpoint)) / range
    child2 = (parent2 * midpoint + parent1 * (range - midpoint)) / range
    return child1, child2

def crossover(c_range=10):
    global population
    for i in range(len(population) // 2):
        x = population[2 * i]
        y = population[2 * i + 1]
        child1, child2 = reproduce(x, y, c_range)
        population[2 * i] = child1
        population[2 * i + 1] = child2


def mutate(alpha):
    global population
    for i in range(len(population)):
        if random.random() < alpha:
            population[i] += random.uniform(-0.1, 0.1)


def GA(pop_size=1000, threshold=0.000001, crossover_range=10, alpha=0.1):
    initialize_population(pop_size)
    sort_by_fitness()
    generation = 0
    while fitness(population[0]) > threshold:
        crossover(crossover_range)
        mutate(alpha)
        sort_by_fitness()
        generation += 1
    return population[0]



print(GA())    




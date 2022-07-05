# Q5_graded
# Do not change the above line.

class Task:
  def __init__(self, time_demand: float, machine_demand: float):
    self.time_demand = time_demand
    self.machine_demand = machine_demand

class Machine:
  def __init__(self, machine_id: int, time_supply: float, time_velocity: float, machine_supply: float, machine_capacity: float):
    self.id = machine_id
    self.time_supply = time_supply
    self.time_velocity = time_velocity
    self.machine_supply = machine_supply
    self.machine_capacity = machine_capacity      

# Q5_graded
# Do not change the above line.
import numpy as np
from typing import List
from matplotlib import pyplot as plt

class ACOscheduler:
  def __init__(self, tasks, machines, population_number=100, iterations=500):
    self.tasks = tasks
    self.machines = machines
    self.task_num = len(tasks) # The number of tasks  
    self.machine_number = len(machines) # Number of machines 
    self.population_number = population_number # Population number 
    self.iterations = iterations 
    # The pheromone of the machine representing the task selection 
    self.pheromone_phs = [[100 for _ in range(self.machine_number)] for _ in range(self.task_num)]
    self.best_pheromone = None

# Generate a new solution vector 
  def gen_pheromone_jobs(self):
    ans = [-1 for _ in range(self.task_num)]
    node_free = [node_id for node_id in range(self.machine_number)]
    for let in range(self.task_num):
      ph_sum = np.sum(list(map(lambda j: self.pheromone_phs[let][j], node_free)))
      test_val = 0
      rand_ph = np.random.uniform(0, ph_sum)
      for node_id in node_free:
        test_val += self.pheromone_phs[let][node_id]
        if rand_ph <= test_val:
          ans[let] = node_id
          break
    return ans

# Evaluate the current solution vector 
  def evaluate_particle(self, pheromone_jobs: List[int]) -> int:
    time_util = np.zeros(self.machine_number)
    machine_util = np.zeros(self.machine_number)

    for i in range(len(self.machines)):
      time_util[i] = self.machines[i].time_supply
      machine_util[i] = self.machines[i].machine_supply

    for i in range(self.task_num):
      time_util[pheromone_jobs[i]] += self.tasks[i].time_demand
      machine_util[pheromone_jobs[i]] += self.tasks[i].machine_demand

    for i in range(self.machine_number):
      if time_util[i] > self.machines[i].time_velocity:
        return 100
      if machine_util[i] > self.machines[i].machine_capacity:
        return 100
    for i in range(self.machine_number):
      time_util[i] /= self.machines[i].time_velocity
      machine_util[i] /= self.machines[i].machine_capacity

    return np.std(time_util, ddof=1) + np.std(machine_util, ddof=1)


  # Calculate Fitness 
  def calculate_fitness(self, pheromone_jobs: List[int]) -> float:
    return 1 / self.evaluate_particle(pheromone_jobs)

  # Update pheromones 
  def update_pheromones(self):

    for i in range(self.task_num):
      for j in range(self.machine_number):
        if j == self.best_pheromone[i]:
          self.pheromone_phs[i][j] *= 2
        else:
          self.pheromone_phs[i][j] *= 0.5

  def scheduler_main(self):
    results = [0 for _ in range(self.iterations)]
    fitness = 0

    for it in range(self.iterations):
      best_time = 0
      for ant_id in range(self.population_number):
        pheromone_jobs = self.gen_pheromone_jobs()
        fitness = self.calculate_fitness(pheromone_jobs)
        if fitness > best_time:
          self.best_pheromone = pheromone_jobs
          best_time = fitness
          assert self.best_pheromone is not None
          self.update_pheromones()
          results[it] = best_time
          if it % 10 == 0:
            print("ACO iter: ", it, " / ", self.iterations, ", Fitness : ", fitness)
    return results


if __name__ == '__main__':
  nodes = [Machine(0, 0.862, 950, 950, 1719), Machine(1, 0.962, 2, 950, 1719), Machine(2, 1.062, 2, 1500, 1719)]
  lets = [Task(0.15, 50), Task(0.05, 100), Task(0.2, 60),
  Task(0.01, 70), Task(0.04, 80), Task(0.07, 20),
  Task(0.14, 150), Task(0.15, 200), Task(0.03, 40), Task(0.06, 90)]
  ac = ACOscheduler(lets, nodes, iterations=150)
  res = ac.scheduler_main()
  i = 0
  for _ in ac.best_pheromone:
    print(" Task :", i, " Put it on the machine ", ac.best_pheromone[i])
    i += 1


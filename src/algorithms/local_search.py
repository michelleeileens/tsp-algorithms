import numpy as np
from typing import List, Tuple, Set
import random
import math
from src.utils import time_tracked

# ------------------ Hill Climbing ------------------
class HillClimbing:
    def __init__(self, adj_matrix: np.ndarray):
        self.adj_matrix = adj_matrix
        self.n = len(adj_matrix)
        self.best_solution = None
        self.best_cost = float('inf')

    def _calculate_tour_cost(self, tour: List[int]) -> float:
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.adj_matrix[tour[i]][tour[i + 1]]
        cost += self.adj_matrix[tour[-1]][tour[0]]
        return cost

    def _generate_neighbors(self, tour: List[int]) -> List[List[int]]:
        neighbors = []
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1:
                    continue
                new_tour = tour.copy()
                new_tour[i:j] = tour[i:j][::-1]
                neighbors.append(new_tour)
        return neighbors

    def _generate_random_tour(self) -> List[int]:
        cities = list(range(1, self.n))
        random.shuffle(cities)
        return [0] + cities

    @time_tracked
    def solve(self, num_restarts: int = 5) -> Tuple[List[int], float]:
        for _ in range(num_restarts):
            current_tour = self._generate_random_tour()
            current_cost = self._calculate_tour_cost(current_tour)
            improved = True
            while improved:
                improved = False
                neighbors = self._generate_neighbors(current_tour)
                for neighbor in neighbors:
                    neighbor_cost = self._calculate_tour_cost(neighbor)
                    if neighbor_cost < current_cost:
                        current_tour = neighbor
                        current_cost = neighbor_cost
                        improved = True
                        break
            if current_cost < self.best_cost:
                self.best_solution = current_tour
                self.best_cost = current_cost
        return self.best_solution, self.best_cost

# ------------------ Simulated Annealing ------------------
class SimulatedAnnealing:
    def __init__(self, adj_matrix: np.ndarray):
        self.adj_matrix = adj_matrix
        self.n = len(adj_matrix)
        self.best_solution = None
        self.best_cost = float('inf')

    def _calculate_tour_cost(self, tour: List[int]) -> float:
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.adj_matrix[tour[i]][tour[i + 1]]
        cost += self.adj_matrix[tour[-1]][tour[0]]
        return cost

    def _generate_neighbor(self, tour: List[int]) -> List[int]:
        i, j = random.sample(range(1, len(tour)), 2)
        if i > j:
            i, j = j, i
        new_tour = tour.copy()
        new_tour[i:j] = tour[i:j][::-1]
        return new_tour

    def _acceptance_probability(self, old_cost: float, new_cost: float, temperature: float) -> float:
        if new_cost < old_cost:
            return 1.0
        return math.exp((old_cost - new_cost) / temperature)

    def _generate_initial_tour(self) -> List[int]:
        cities = list(range(1, self.n))
        random.shuffle(cities)
        return [0] + cities

    @time_tracked
    def solve(self, initial_temp: float = 100.0, alpha: float = 0.95, min_temp: float = 1e-8, max_iterations: int = 1000) -> Tuple[List[int], float]:
        current_tour = self._generate_initial_tour()
        current_cost = self._calculate_tour_cost(current_tour)
        self.best_solution = current_tour
        self.best_cost = current_cost
        temperature = initial_temp
        while temperature > min_temp:
            for _ in range(max_iterations):
                new_tour = self._generate_neighbor(current_tour)
                new_cost = self._calculate_tour_cost(new_tour)
                if self._acceptance_probability(current_cost, new_cost, temperature) > random.random():
                    current_tour = new_tour
                    current_cost = new_cost
                    if current_cost < self.best_cost:
                        self.best_solution = current_tour.copy()
                        self.best_cost = current_cost
            temperature *= alpha
        return self.best_solution, self.best_cost

# ------------------ Genetic Algorithm ------------------
class GeneticAlgorithm:
    def __init__(self, adj_matrix: np.ndarray):
        self.adj_matrix = adj_matrix
        self.n = len(adj_matrix)
        self.best_solution = None
        self.best_cost = float('inf')

    def _calculate_tour_cost(self, tour: List[int]) -> float:
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.adj_matrix[tour[i]][tour[i + 1]]
        cost += self.adj_matrix[tour[-1]][tour[0]]
        return cost

    def _generate_initial_population(self, population_size: int) -> List[List[int]]:
        population = []
        for _ in range(population_size):
            cities = list(range(1, self.n))
            random.shuffle(cities)
            tour = [0] + cities
            population.append(tour)
        return population

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        size = len(parent1)
        start, end = sorted(random.sample(range(1, size), 2))
        child = [0] * size
        child[start:end] = parent1[start:end]
        parent2_cities = [city for city in parent2 if city not in child[start:end]]
        j = 0
        for i in range(size):
            if i < start or i >= end:
                child[i] = parent2_cities[j]
                j += 1
        return child

    def _mutate(self, tour: List[int], mutation_rate: float) -> List[int]:
        if random.random() < mutation_rate:
            i, j = random.sample(range(1, len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour

    def _select_parent(self, population: List[List[int]], costs: List[float]) -> List[int]:
        tournament_size = 3
        tournament = random.sample(range(len(population)), tournament_size)
        return population[min(tournament, key=lambda i: costs[i])]

    @time_tracked
    def solve(self, population_size: int = 50, num_generations: int = 100, mutation_rate: float = 0.1) -> Tuple[List[int], float]:
        population = self._generate_initial_population(population_size)
        for generation in range(num_generations):
            costs = [self._calculate_tour_cost(tour) for tour in population]
            min_cost_idx = min(range(len(costs)), key=lambda i: costs[i])
            if costs[min_cost_idx] < self.best_cost:
                self.best_cost = costs[min_cost_idx]
                self.best_solution = population[min_cost_idx].copy()
            new_population = []
            new_population.append(population[min_cost_idx].copy())
            while len(new_population) < population_size:
                parent1 = self._select_parent(population, costs)
                parent2 = self._select_parent(population, costs)
                child = self._order_crossover(parent1, parent2)
                child = self._mutate(child, mutation_rate)
                new_population.append(child)
            population = new_population
        return self.best_solution, self.best_cost

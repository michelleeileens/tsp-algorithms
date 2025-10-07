import numpy as np
from typing import List, Tuple, Set
import random
import math
from src.utils import time_tracked


class HillClimbing:
    def __init__(self, adj_matrix: np.ndarray):
        self.adj_matrix = adj_matrix
        self.n = len(adj_matrix)
        self.best_solution = None
        self.best_cost = float('inf')

    def _calculate_tour_cost(self, tour: List[int]) -> float:
        """Calculate tour cost efficiently."""
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.adj_matrix[tour[i]][tour[i + 1]]
        cost += self.adj_matrix[tour[-1]][tour[0]]
        return cost

    def _generate_neighbors(self, tour: List[int]) -> List[List[int]]:
        """Generate neighbors using 2-opt swaps."""
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
        """Generate a random tour starting from city 0."""
        cities = list(range(1, self.n))
        random.shuffle(cities)
        return [0] + cities

    @time_tracked
    def solve(self, num_restarts: int = 5, track_convergence: bool = False) -> Tuple[List[int], float]:
        """
        Solve TSP using Hill Climbing with multiple random restarts.
        
        Args:
            num_restarts (int): Number of random restarts
            track_convergence (bool): Whether to track convergence history
        
        Returns:
            Tuple[List[int], float]: Best tour and its cost
            If track_convergence=True, returns (best_tour, best_cost, convergence_history)
        """
        convergence_history = []
        iteration = 0
        
        for restart in range(num_restarts):
            current_tour = self._generate_random_tour()
            current_cost = self._calculate_tour_cost(current_tour)
            
            if current_cost < self.best_cost:
                self.best_solution = current_tour
                self.best_cost = current_cost
            
            if track_convergence:
                convergence_history.append((iteration, self.best_cost))
                iteration += 1
            
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
                        
                        if current_cost < self.best_cost:
                            self.best_solution = current_tour
                            self.best_cost = current_cost
                        
                        if track_convergence:
                            convergence_history.append((iteration, self.best_cost))
                            iteration += 1
                        break
        
        if track_convergence:
            return self.best_solution, self.best_cost, convergence_history
        return self.best_solution, self.best_cost
    
    def solve_with_convergence(self, num_restarts: int = 5) -> Tuple[List[int], float, List[Tuple[int, float]]]:
        """
        Solve TSP using Hill Climbing with convergence tracking (no timing decorator).
        
        Args:
            num_restarts (int): Number of random restarts
        
        Returns:
            Tuple[List[int], float, List[Tuple[int, float]]]: Best tour, cost, and convergence history
        """
        convergence_history = []
        iteration = 0
        
        for restart in range(num_restarts):
            current_tour = self._generate_random_tour()
            current_cost = self._calculate_tour_cost(current_tour)
            
            if current_cost < self.best_cost:
                self.best_solution = current_tour
                self.best_cost = current_cost
            
            convergence_history.append((iteration, self.best_cost))
            iteration += 1
            
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
                        
                        if current_cost < self.best_cost:
                            self.best_solution = current_tour
                            self.best_cost = current_cost
                        
                        convergence_history.append((iteration, self.best_cost))
                        iteration += 1
                        break
        
        return self.best_solution, self.best_cost, convergence_history


class SimulatedAnnealing:
    def __init__(self, adj_matrix: np.ndarray):
        self.adj_matrix = adj_matrix
        self.n = len(adj_matrix)
        self.best_solution = None
        self.best_cost = float('inf')

    def _calculate_tour_cost(self, tour: List[int]) -> float:
        """Calculate tour cost efficiently."""
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += self.adj_matrix[tour[i]][tour[i + 1]]
        cost += self.adj_matrix[tour[-1]][tour[0]]
        return cost

    def _calculate_2opt_delta(self, tour: List[int], i: int, j: int) -> float:
        """Calculate cost change for 2-opt move without creating new tour."""
        # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[j+1])
        # New edges after reversal: (tour[i-1], tour[j]) and (tour[i], tour[j+1])
        
        n = len(tour)
        if i > j:
            i, j = j, i

        prev_i = i - 1
        next_j = (j + 1) % n
        
        # Cost of edges to be removed
        old_cost = (self.adj_matrix[tour[prev_i]][tour[i]] + 
                   self.adj_matrix[tour[j]][tour[next_j]])
        
        # Cost of edges to be added
        new_cost = (self.adj_matrix[tour[prev_i]][tour[j]] + 
                   self.adj_matrix[tour[i]][tour[next_j]])
        
        return new_cost - old_cost

    def _apply_2opt_move(self, tour: List[int], i: int, j: int) -> None:
        """Apply 2-opt move in-place by reversing segment."""
        if i > j:
            i, j = j, i
        tour[i:j+1] = tour[i:j+1][::-1]

    def _acceptance_probability(self, delta_cost: float, temperature: float) -> float:
        """Calculate acceptance probability for a move."""
        if delta_cost < 0:
            return 1.0
        return math.exp(-delta_cost / temperature)

    def _generate_initial_tour(self) -> List[int]:
        """Generate random initial tour."""
        cities = list(range(1, self.n))
        random.shuffle(cities)
        return [0] + cities

    @time_tracked
    def solve(self, initial_temp: float = None, alpha: float = 0.99, min_temp: float = 0.01, max_iterations: int = None, track_convergence: bool = False) -> Tuple[List[int], float]:
        """
        Optimized SA solve with efficient neighbor generation and faster convergence.

        Args:
            initial_temp (float): Initial temperature for SA
            alpha (float): Cooling rate
            min_temp (float): Minimum temperature
            max_iterations (int): Maximum iterations per temperature level
            track_convergence (bool): Whether to track convergence history

        Returns:
            Tuple[List[int], float]: Best tour and its cost
            If track_convergence=True, returns (best_tour, best_cost, convergence_history)
        """
        # Set default parameters as requested
        if initial_temp is None:
            initial_temp = self.n * 50
        if max_iterations is None:
            max_iterations = self.n * 20
        
        # Initialize with random tour
        current_tour = self._generate_initial_tour()
        current_cost = self._calculate_tour_cost(current_tour)
        
        # Track best solution
        self.best_solution = current_tour[:]
        self.best_cost = current_cost
        
        convergence_history = []
        iteration = 0
        
        if track_convergence:
            convergence_history.append((iteration, self.best_cost))
        
        temperature = initial_temp
        no_improve_count = 0
        max_no_improve = self.n * 5  # Early termination
        
        # Main SA loop
        while temperature > min_temp and no_improve_count < max_no_improve:
            improved = False
            
            # Fixed number of attempts per temperature level
            for _ in range(max_iterations):
                # Generate random 2-opt move
                # For valid 2-opt: need i < j and j >= i + 2 to avoid adjacent edges
                if self.n < 4:  # Too small for 2-opt
                    break
                    
                i = random.randint(1, max(1, self.n - 3))  # Ensure valid range
                j = random.randint(i + 2, self.n - 1)
                
                # Calculate cost change efficiently
                delta = self._calculate_2opt_delta(current_tour, i, j)
                
                # Accept or reject move
                if (delta < 0 or
                    random.random() < math.exp(-delta / temperature)):

                    # Apply move in-place
                    self._apply_2opt_move(current_tour, i, j)
                    current_cost += delta
                    
                    # Update best solution if improved
                    if current_cost < self.best_cost:
                        self.best_solution = current_tour[:]
                        self.best_cost = current_cost
                        improved = True
                        no_improve_count = 0
                        
                        if track_convergence:
                            iteration += 1
                            convergence_history.append((iteration, self.best_cost))

            # Cool down temperature
            temperature *= alpha

            if not improved:
                no_improve_count += 1
        
        if track_convergence:
            return self.best_solution, self.best_cost, convergence_history
        return self.best_solution, self.best_cost
    
    def solve_with_convergence(self, initial_temp: float = None, alpha: float = 0.99, min_temp: float = 0.01, max_iterations: int = None) -> Tuple[List[int], float, List[Tuple[int, float]]]:
        """
        Solve TSP using Simulated Annealing with convergence tracking (no timing decorator).
        """
        # Set default parameters
        if initial_temp is None:
            initial_temp = self.n * 50
        if max_iterations is None:
            max_iterations = self.n * 20
        
        # Initialize with random tour
        current_tour = self._generate_initial_tour()
        current_cost = self._calculate_tour_cost(current_tour)
        
        # Track best solution
        self.best_solution = current_tour[:]
        self.best_cost = current_cost
        
        convergence_history = []
        iteration = 0
        convergence_history.append((iteration, self.best_cost))
        
        temperature = initial_temp
        no_improve_count = 0
        max_no_improve = self.n * 5
        
        # Main SA loop
        while temperature > min_temp and no_improve_count < max_no_improve:
            improved = False
            
            for _ in range(max_iterations):
                if self.n < 4:
                    break
                    
                i = random.randint(1, max(1, self.n - 3))
                j = random.randint(i + 2, self.n - 1)
                
                delta = self._calculate_2opt_delta(current_tour, i, j)
                
                if (delta < 0 or random.random() < math.exp(-delta / temperature)):
                    self._apply_2opt_move(current_tour, i, j)
                    current_cost += delta
                    
                    if current_cost < self.best_cost:
                        self.best_solution = current_tour[:]
                        self.best_cost = current_cost
                        improved = True
                        no_improve_count = 0
                        
                        iteration += 1
                        convergence_history.append((iteration, self.best_cost))

            temperature *= alpha
            if not improved:
                no_improve_count += 1
        
        return self.best_solution, self.best_cost, convergence_history
    

class GeneticAlgorithm:
    def __init__(self, adj_matrix: np.ndarray):
        self.adj_matrix = adj_matrix
        self.n = len(adj_matrix)
        self.best_solution = None
        self.best_cost = float('inf')

    def _calculate_tour_cost(self, tour: List[int]) -> float:
        """Calculate tour cost efficiently."""
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += self.adj_matrix[tour[i]][tour[i + 1]]
        cost += self.adj_matrix[tour[-1]][tour[0]]
        return cost

    def _validate_tour(self, tour: List[int]) -> bool:
        """Validate that tour is a valid TSP solution."""
        if len(tour) != self.n:
            return False
        if tour[0] != 0:
            return False
        if set(tour) != set(range(self.n)):
            return False
        return True

    def _generate_initial_population(self, population_size: int) -> List[List[int]]:
        """Generate initial population with diverse tours starting from city 0."""
        population = []
        for _ in range(population_size):
            # Create tour: [0, ...random permutation of cities 1 to n-1...]
            cities = list(range(1, self.n))
            random.shuffle(cities)
            tour = [0] + cities
            
            # Validate tour
            assert self._validate_tour(tour), f"Invalid initial tour: {tour}"
            population.append(tour)
        return population

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Correct Order Crossover (OX) implementation. Always maintains city 0 at position 0."""
        size = len(parent1)
        
        # Select crossover points in range [1, size-1] (skip position 0)
        if size <= 2:
            return parent1[:]
        
        # Select crossover segment (excluding position 0)
        start, end = sorted(random.sample(range(1, size), 2))
         # Create child with segment from parent1
        child = [0] + [-1] * (size - 1)
        child[start:end] = parent1[start:end]
        
        # Fill remaining positions with cities from parent2 in order
        used = set(child[start:end])
        used.add(0)
        
        p2_cities = [city for city in parent2[1:] if city not in used]
        
        # Fill positions before and after the segment
        idx = 0
        for pos in range(1, size):
            if child[pos] == -1:
                child[pos] = p2_cities[idx]
                idx += 1
        
        return child

    def _mutate(self, tour: List[int], mutation_rate: float) -> List[int]:
        """Apply 2-opt mutation with given probability."""
        if random.random() > mutation_rate or self.n <= 3:
            return tour[:]
                
        # Perform 2-opt swap (excluding position 0)
        i, j = sorted(random.sample(range(1, len(tour)), 2))
        tour_copy = tour[:]
        tour_copy[i:j+1] = reversed(tour_copy[i:j+1])
        return tour_copy

    def _select_parent(self, population: List[List[int]], costs: List[float]) -> List[int]:
        """Tournament selection with moderate selection pressure."""
        tournament_size = min(3, len(population))  # Smaller tournament for more diversity
        tournament = random.sample(range(len(population)), tournament_size)
        winner_idx = min(tournament, key=lambda i: costs[i])
        return population[winner_idx][:]

    @time_tracked
    def solve(self, population_size: int = 50, num_generations: int = 100, mutation_rate: float = 0.02, track_convergence: bool = False) -> Tuple[List[int], float]:
        """
        Solve TSP using Genetic Algorithm with Order Crossover.
        
        Args:
            population_size: Number of individuals in population (default: 50)
            num_generations: Number of generations to evolve (default: 100) 
            mutation_rate: Probability of mutation per individual (default: 0.02)
            track_convergence (bool): Whether to track convergence history
            
        Returns:
            Tuple[List[int], float]: Best tour and its cost
            If track_convergence=True, returns (best_tour, best_cost, convergence_history)
        """
        # Initialize population
        population = self._generate_initial_population(population_size)
        costs = [self._calculate_tour_cost(tour) for tour in population]
        
        # Track best solution
        best_idx = costs.index(min(costs))
        self.best_cost = costs[best_idx]
        self.best_solution = population[best_idx][:]
        
        convergence_history = []
        if track_convergence:
            convergence_history.append((0, self.best_cost))
        
        # Evolution loop
        for generation in range(num_generations):
            # Update costs
            costs = [self._calculate_tour_cost(tour) for tour in population]
            
            # Track best
            current_best_idx = costs.index(min(costs))
            if costs[current_best_idx] < self.best_cost:
                self.best_cost = costs[current_best_idx]
                self.best_solution = population[current_best_idx][:]
                
                if track_convergence:
                    convergence_history.append((generation + 1, self.best_cost))
            
            # Elitism: keep top 20%
            elite_count = max(1, population_size // 5)
            elite_indices = sorted(range(len(costs)), key=lambda i: costs[i])[:elite_count]
            new_population = [population[i][:] for i in elite_indices]
            
            # Generate offspring
            while len(new_population) < population_size:
                parent1 = self._select_parent(population, costs)
                parent2 = self._select_parent(population, costs)
                
                child = self._order_crossover(parent1, parent2)
                child = self._mutate(child, mutation_rate)
                
                new_population.append(child)
            
            population = new_population
        
        if track_convergence:
            return self.best_solution, self.best_cost, convergence_history
        return self.best_solution, self.best_cost
    
    def solve_with_convergence(self, population_size: int = 150, num_generations: int = 300, mutation_rate: float = 0.02) -> Tuple[List[int], float, List[Tuple[int, float]]]:
        """
        Solve TSP using Genetic Algorithm with convergence tracking (no timing decorator).
        """
        # Initialize population
        population = self._generate_initial_population(population_size)
        costs = [self._calculate_tour_cost(tour) for tour in population]
        
        # Track best solution
        best_idx = costs.index(min(costs))
        self.best_cost = costs[best_idx]
        self.best_solution = population[best_idx][:]
        
        convergence_history = []
        convergence_history.append((0, self.best_cost))
        
        # Evolution loop
        for generation in range(num_generations):
            # Update costs
            costs = [self._calculate_tour_cost(tour) for tour in population]
            
            # Track best
            current_best_idx = costs.index(min(costs))
            if costs[current_best_idx] < self.best_cost:
                self.best_cost = costs[current_best_idx]
                self.best_solution = population[current_best_idx][:]
                convergence_history.append((generation + 1, self.best_cost))
            
            # Elitism: keep top 20%
            elite_count = max(1, population_size // 5)
            elite_indices = sorted(range(len(costs)), key=lambda i: costs[i])[:elite_count]
            new_population = [population[i][:] for i in elite_indices]
            
            # Generate offspring
            while len(new_population) < population_size:
                parent1 = self._select_parent(population, costs)
                parent2 = self._select_parent(population, costs)
                
                child = self._order_crossover(parent1, parent2)
                child = self._mutate(child, mutation_rate)
                
                new_population.append(child)
            
            population = new_population
        
        return self.best_solution, self.best_cost, convergence_history

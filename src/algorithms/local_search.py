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
        prev_i = (i - 1) % n
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
    def solve(self, initial_temp: float = None, alpha: float = 0.99, min_temp: float = 0.01, max_iterations: int = None) -> Tuple[List[int], float]:
        """Optimized SA solve with efficient neighbor generation and faster convergence."""
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
        
        temperature = initial_temp
        iterations_since_improvement = 0
        max_no_improve = self.n * 5  # Early termination
        
        # Main SA loop
        while temperature > min_temp and iterations_since_improvement < max_no_improve:
            improved_this_temp = False
            
            # Fixed number of attempts per temperature level
            for _ in range(max_iterations):
                # Generate random 2-opt move
                i, j = random.sample(range(1, self.n), 2)
                if i > j:
                    i, j = j, i
                if j - i < 2:  # Skip moves that don't change the tour
                    continue
                
                # Calculate cost change efficiently
                delta_cost = self._calculate_2opt_delta(current_tour, i, j)
                
                # Accept or reject move
                if (delta_cost < 0 or 
                    random.random() < self._acceptance_probability(delta_cost, temperature)):
                    
                    # Apply move in-place
                    self._apply_2opt_move(current_tour, i, j)
                    current_cost += delta_cost
                    
                    # Update best solution if improved
                    if current_cost < self.best_cost:
                        self.best_solution = current_tour[:]
                        self.best_cost = current_cost
                        improved_this_temp = True
                        iterations_since_improvement = 0
            
            # Cool down temperature
            temperature *= alpha
            
            if not improved_this_temp:
                iterations_since_improvement += 1
        
        return self.best_solution, self.best_cost
    
# ------------------ Genetic Algorithm ------------------
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
        """
        Correct Order Crossover (OX) implementation.
        Maintains city 0 at position 0 and preserves relative order from parents.
        """
        size = len(parent1)
        
        # Ensure parents are valid
        assert self._validate_tour(parent1), f"Invalid parent1: {parent1}"
        assert self._validate_tour(parent2), f"Invalid parent2: {parent2}"
        
        # Select crossover points in range [1, size-1] (skip position 0)
        if size <= 2:
            return parent1[:]
        
        start, end = sorted(random.sample(range(1, size), 2))
        if start == end:
            end = min(end + 1, size)
        
        # Initialize child
        child = [0] * size  # Start with city 0 at position 0
        
        # Step 1: Copy segment from parent1 to child
        for i in range(start, end):
            child[i] = parent1[i]
        
        # Step 2: Create list of cities from parent2 in order, excluding those already in child
        used_cities = set(child[start:end])
        used_cities.add(0)  # City 0 is always used
        
        # Get remaining cities from parent2 in their original order
        remaining_cities = [city for city in parent2 if city not in used_cities]
        
        # Step 3: Fill remaining positions in child with cities from remaining_cities
        fill_idx = 0
        for i in range(1, size):  # Skip position 0 (always city 0)
            if i < start or i >= end:  # Position not filled by crossover segment
                if fill_idx < len(remaining_cities):
                    child[i] = remaining_cities[fill_idx]
                    fill_idx += 1
        
        # Final validation and repair if needed
        if not self._validate_tour(child):
            # Repair tour by ensuring all cities are present exactly once
            child_cities = set(child[1:])  # Exclude city 0
            all_cities = set(range(1, self.n))
            missing = all_cities - child_cities
            duplicates = []
            
            # Find duplicates (excluding city 0)
            seen = {0}
            for i in range(1, size):
                if child[i] in seen:
                    duplicates.append(i)
                else:
                    seen.add(child[i])
            
            # Replace duplicates with missing cities
            missing_list = list(missing)
            for i, pos in enumerate(duplicates):
                if i < len(missing_list):
                    child[pos] = missing_list[i]
        
        # Final validation
        assert self._validate_tour(child), f"Invalid child after crossover: {child}"
        return child

    def _mutate(self, tour: List[int], mutation_rate: float) -> List[int]:
        """
        Apply 2-opt mutation with controlled probability.
        Preserves city 0 at position 0.
        """
        if random.random() > mutation_rate or self.n <= 3:
            return tour[:]
        
        tour = tour[:]
        
        # Apply 2-opt mutation (reverse a segment)
        i, j = sorted(random.sample(range(1, len(tour)), 2))
        if j - i >= 1:  # Ensure meaningful mutation
            tour[i:j+1] = tour[i:j+1][::-1]
        
        # Validate mutated tour
        assert self._validate_tour(tour), f"Invalid tour after mutation: {tour}"
        return tour

    def _select_parent(self, population: List[List[int]], costs: List[float]) -> List[int]:
        """Tournament selection with moderate selection pressure."""
        tournament_size = min(3, len(population))  # Smaller tournament for more diversity
        tournament = random.sample(range(len(population)), tournament_size)
        winner_idx = min(tournament, key=lambda i: costs[i])
        return population[winner_idx][:]

    @time_tracked
    def solve(self, population_size: int = 150, num_generations: int = 300, mutation_rate: float = 0.02) -> Tuple[List[int], float]:
        """
        Solve TSP using Genetic Algorithm with Order Crossover.
        
        Args:
            population_size: Number of individuals in population (default: 150)
            num_generations: Number of generations to evolve (default: 300) 
            mutation_rate: Probability of mutation per individual (default: 0.02)
            
        Returns:
            Tuple[List[int], float]: Best tour and its cost
        """
        # Generate initial population
        population = self._generate_initial_population(population_size)
        
        # Initialize best solution tracking
        costs = [self._calculate_tour_cost(tour) for tour in population]
        best_idx = min(range(len(costs)), key=lambda i: costs[i])
        self.best_cost = costs[best_idx]
        self.best_solution = population[best_idx][:]
        
        # Evolution loop
        for generation in range(num_generations):
            # Calculate fitness for current population
            costs = [self._calculate_tour_cost(tour) for tour in population]
            
            # Update best solution
            current_best_idx = min(range(len(costs)), key=lambda i: costs[i])
            if costs[current_best_idx] < self.best_cost:
                self.best_cost = costs[current_best_idx]
                self.best_solution = population[current_best_idx][:]
            
            # Create new population
            # Elitism: Keep top 20% of population
            elite_count = max(1, population_size // 5)
            sorted_indices = sorted(range(len(costs)), key=lambda i: costs[i])
            new_population = [population[i][:] for i in sorted_indices[:elite_count]]
            
            # Generate offspring to fill rest of population
            while len(new_population) < population_size:
                # Select parents
                parent1 = self._select_parent(population, costs)
                parent2 = self._select_parent(population, costs)
                
                # Create offspring through crossover and mutation
                child = self._order_crossover(parent1, parent2)
                child = self._mutate(child, mutation_rate)
                
                # Validate child before adding to population
                if self._validate_tour(child):
                    new_population.append(child)
                else:
                    # Fallback: use parent1 if child is invalid
                    new_population.append(parent1[:])
            
            population = new_population
        
        # Final validation
        assert self._validate_tour(self.best_solution), f"Invalid final solution: {self.best_solution}"
        return self.best_solution, self.best_cost

import numpy as np
from typing import List, Tuple, Set
import random
from ..utils import time_tracked, calculate_path_cost

class NearestNeighbor:
    def __init__(self, adj_matrix: np.ndarray):
        """
        Initialize the Nearest Neighbor solver.
        
        Args:
            adj_matrix (np.ndarray): The adjacency matrix representing distances between cities
        """
        self.adj_matrix = adj_matrix
        self.n_cities = len(adj_matrix)
    
    @time_tracked
    def solve(self, start_city: int = 0) -> Tuple[List[int], float]:
        """
        Find a solution using the basic nearest neighbor algorithm.
        
        Args:
            start_city (int): The city to start from. Defaults to 0.
            
        Returns:
            Tuple[List[int], float]: The path and its cost
        """
        unvisited = set(range(self.n_cities))
        current = start_city
        path = [current]
        unvisited.remove(current)
        total_cost = 0.0
        
        while unvisited:
            next_city = min(unvisited, 
                          key=lambda x: self.adj_matrix[current][x])
            total_cost += self.adj_matrix[current][next_city]
            current = next_city
            path.append(current)
            unvisited.remove(current)
        
        # Return to start
        total_cost += self.adj_matrix[current][start_city]
        path.append(start_city)
        
        return path, total_cost

class NearestNeighbor2Opt:
    def __init__(self, adj_matrix: np.ndarray):
        self.adj_matrix = adj_matrix
        self.n_cities = len(adj_matrix)
        self.nn = NearestNeighbor(adj_matrix)
    
    def _swap_2opt(self, path: List[int], i: int, k: int) -> List[int]:
        """Perform a 2-opt swap by reversing the path between positions i and k."""
        return path[:i] + path[i:k+1][::-1] + path[k+1:]
    
    def _calculate_swap_delta(self, path: List[int], i: int, k: int) -> float:
        """Calculate the change in path length if we perform a 2-opt swap."""
        if k - i <= 1:  # Adjacent or same cities
            return 0.0
            
        n = len(path)
        a, b = path[i-1], path[i]
        c, d = path[k], path[(k+1) % n]
        
        old_distance = (self.adj_matrix[a][b] + 
                       self.adj_matrix[c][d])
        new_distance = (self.adj_matrix[a][c] + 
                       self.adj_matrix[b][d])
        
        return new_distance - old_distance
    
    @time_tracked
    def solve(self, start_city: int = 0) -> Tuple[List[int], float]:
        """
        Find a solution using nearest neighbor followed by 2-opt improvement.
        Args:
            start_city (int): The city to start from. Defaults to 0.
        Returns:
            Tuple[List[int], float]: The improved path and its cost
        """
        # Get initial solution from nearest neighbor
        result, _, _ = self.nn.solve(start_city)
        path, _ = result
        # Remove the duplicate start city at the end for 2-opt processing
        if len(path) > 1 and path[0] == path[-1]:
            path = path[:-1]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(path)):
                for k in range(i + 1, len(path)):
                    if k - i <= 1:  # Skip adjacent cities
                        continue
                    delta = self._calculate_swap_delta(path, i, k)
                    if delta < 0:  # Improvement found
                        path = self._swap_2opt(path, i, k)
                        improved = True
                        break
                if improved:
                    break
        
        # Add start city to complete the cycle
        path.append(path[0])
        total_cost = calculate_path_cost(path, self.adj_matrix)
        return path, total_cost

class RepeatedRandomNN:
    def __init__(self, adj_matrix: np.ndarray):
        self.adj_matrix = adj_matrix
        self.n_cities = len(adj_matrix)
    
    def _get_k_nearest(self, current: int, unvisited: Set[int], k: int) -> List[int]:
        """Get k nearest unvisited cities to the current city."""
        distances = [(city, self.adj_matrix[current][city]) 
                    for city in unvisited]
        distances.sort(key=lambda x: x[1])
        return [city for city, _ in distances[:min(k, len(distances))]]
    
    def _single_solution(self, start_city: int, k: int) -> Tuple[List[int], float]:
        """Generate a single solution using k-nearest random selection."""
        unvisited = set(range(self.n_cities))
        current = start_city
        path = [current]
        unvisited.remove(current)
        total_cost = 0.0
        
        while unvisited:
            candidates = self._get_k_nearest(current, unvisited, k)
            if len(candidates) == 1:
                next_city = candidates[0]
            else:
                next_city = random.choice(candidates)
            total_cost += self.adj_matrix[current][next_city]
            current = next_city
            path.append(current)
            unvisited.remove(current)
        
        # Return to start
        total_cost += self.adj_matrix[current][start_city]
        path.append(start_city)
        
        return path, total_cost
    
    @time_tracked
    def solve(self, k: int = 3, num_repeats: int = 50, start_city: int = 0) -> Tuple[List[int], float]:
        """
        Solve TSP using Repeated Random Nearest Neighbor.
        Args:
            k (int): Number of nearest cities to consider 
            num_repeats (int): Number of repeated runs
            start_city (int): Starting city
        Returns:
            Tuple[List[int], float]: Best tour found and its cost
        """
        
        best_path = None
        best_cost = float('inf')
        
        # Try multiple k values around the chosen one for better results
        k_values = [max(1, k-1), k, min(self.n_cities-1, k+1)]
        
        for test_k in k_values:
            for _ in range(num_repeats // len(k_values)):
                path, cost = self._single_solution(start_city, test_k)
                if cost < best_cost:
                    best_path = path
                    best_cost = cost
        
        return best_path, best_cost
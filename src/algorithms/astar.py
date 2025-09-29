import numpy as np
from typing import List, Tuple, Set, Dict
from queue import PriorityQueue
from scipy.sparse.csgraph import minimum_spanning_tree
from ..utils import time_tracked, calculate_path_cost

class State:
    def __init__(self, path: List[int], unvisited: Set[int], cost: float):
        """
        Represents a state in the A* search.
        
        Args:
            path (List[int]): Current path of visited cities
            unvisited (Set[int]): Set of unvisited cities
            cost (float): Current path cost (g-value)
        """
        self.path = path
        self.unvisited = unvisited
        self.cost = cost  # g(n)
        self.h_cost = 0   # h(n)
        self.f_cost = 0   # f(n) = g(n) + h(n)
        self._hash = hash(tuple(path))
    
    def __lt__(self, other):
        if self.f_cost == other.f_cost:
            return len(self.path) > len(other.path)
        return self.f_cost < other.f_cost
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        return self.path == other.path

class AStarTSP:
    def __init__(self, adj_matrix: np.ndarray):
        """
        Initialize the A* TSP solver.
        
        Args:
            adj_matrix (np.ndarray): The adjacency matrix representing distances between cities
        """
        self.adj_matrix = adj_matrix
        self.n_cities = len(adj_matrix)
        self.nodes_expanded = 0
        self._mst_cache: Dict[Tuple[int, ...], float] = {}
        self._nearest_neighbor_cache: Dict[int, List[int]] = {}
        self._best_known_cost = float('inf')
        
        # Precompute nearest neighbors for each city
        self._precompute_nearest_neighbors()
    
    def _precompute_nearest_neighbors(self):
        """Precompute and cache sorted nearest neighbors for each city."""
        for city in range(self.n_cities):
            distances = [(j, self.adj_matrix[city][j]) 
                        for j in range(self.n_cities) if j != city]
            distances.sort(key=lambda x: x[1])
            self._nearest_neighbor_cache[city] = [c for c, _ in distances]
    
    def _calculate_mst_cost(self, vertices: Set[int]) -> float:
        """Calculate MST cost with caching and optimizations."""
        if len(vertices) <= 1:
            return 0.0
        
        cache_key = tuple(sorted(vertices))
        if cache_key in self._mst_cache:
            return self._mst_cache[cache_key]
        
        # For small sets, use direct calculation
        if len(vertices) <= 3:
            if len(vertices) == 2:
                v1, v2 = tuple(vertices)
                cost = self.adj_matrix[v1][v2]
            else:  # len == 3
                v1, v2, v3 = tuple(vertices)
                cost = min(
                    self.adj_matrix[v1][v2] + self.adj_matrix[v2][v3],
                    self.adj_matrix[v1][v3] + self.adj_matrix[v2][v3],
                    self.adj_matrix[v1][v2] + self.adj_matrix[v1][v3]
                )
            self._mst_cache[cache_key] = cost
            return cost
        
        # Use scipy for larger sets
        vertices_list = list(vertices)
        n = len(vertices_list)
        submatrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                submatrix[i][j] = self.adj_matrix[vertices_list[i]][vertices_list[j]]
        
        mst = minimum_spanning_tree(submatrix)
        cost = float(mst.sum())
        self._mst_cache[cache_key] = cost
        return cost
    
    def _calculate_heuristic(self, state: State) -> float:
        """Calculate h(n) using MST heuristic."""
        if not state.unvisited:
            return self.adj_matrix[state.path[-1]][state.path[0]]
        
        # Quick lower bound check for last city
        if len(state.unvisited) == 1:
            last_city = next(iter(state.unvisited))
            return (self.adj_matrix[state.path[-1]][last_city] + 
                   self.adj_matrix[last_city][state.path[0]])
        
        # Calculate MST cost of unvisited cities
        mst_cost = self._calculate_mst_cost(state.unvisited)
        
        # Find minimum costs to connect MST to path
        current = state.path[-1]
        start = state.path[0]
        
        # Use cached nearest neighbors for efficiency
        min_to_unvisited = float('inf')
        min_to_start = float('inf')
        
        for city in state.unvisited:
            dist_to_current = self.adj_matrix[current][city]
            if dist_to_current < min_to_unvisited:
                min_to_unvisited = dist_to_current
            
            dist_to_start = self.adj_matrix[city][start]
            if dist_to_start < min_to_start:
                min_to_start = dist_to_start
        
        return mst_cost + min_to_unvisited + min_to_start
    
    def _is_promising(self, state: State) -> bool:
        """Check if a state could possibly lead to a better solution."""
        if state.f_cost >= self._best_known_cost:
            return False
            
        if len(state.unvisited) == 1:
            last_city = next(iter(state.unvisited))
            complete_cost = (state.cost + 
                           self.adj_matrix[state.path[-1]][last_city] + 
                           self.adj_matrix[last_city][state.path[0]])
            if complete_cost >= self._best_known_cost:
                return False
        
        return True
    
    @time_tracked
    def solve(self, start_city: int = 0) -> Tuple[List[int], float, int]:
        """Find optimal solution using A* search."""
        self.nodes_expanded = 0
        self._mst_cache.clear()
        self._best_known_cost = float('inf')
        
        # Initialize starting state
        initial_state = State([start_city], 
                            set(range(self.n_cities)) - {start_city}, 
                            0.0)
        initial_state.h_cost = self._calculate_heuristic(initial_state)
        initial_state.f_cost = initial_state.h_cost
        
        frontier = PriorityQueue()
        frontier.put(initial_state)
        visited = dict()
        while not frontier.empty():
            current = frontier.get()
            state_key = (tuple(current.path), tuple(sorted(current.unvisited)))
            # Only skip if we've already found a better or equal cost to this state
            if state_key in visited and visited[state_key] <= current.cost:
                continue
            visited[state_key] = current.cost
            if not self._is_promising(current):
                continue
            self.nodes_expanded += 1
            # Check if we've found a solution
            if not current.unvisited:
                final_cost = current.cost + self.adj_matrix[current.path[-1]][start_city]
                if final_cost < self._best_known_cost:
                    self._best_known_cost = final_cost
                return current.path + [start_city], final_cost, self.nodes_expanded
            # Generate successors in order of promising cities
            current_city = current.path[-1]
            for next_city in self._nearest_neighbor_cache[current_city]:
                if next_city not in current.unvisited:
                    continue
                
                new_cost = current.cost + self.adj_matrix[current_city][next_city]
                if new_cost >= self._best_known_cost:
                    continue
                
                successor = State(
                    current.path + [next_city],
                    current.unvisited - {next_city},
                    new_cost
                )
                successor.h_cost = self._calculate_heuristic(successor)
                successor.f_cost = successor.cost + successor.h_cost
                
                if self._is_promising(successor):
                    frontier.put(successor)
        
        raise ValueError("No solution found")
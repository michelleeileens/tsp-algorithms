# Project Design Document:### Phase 1: Core Infrastructure (Days 1-2) [✓]
1. Setup project structure and environment [✓]
2. Implement data loading utilities for adjacency matrices [✓]
3. Create basic visualization framework for results
4. Setup testing infrastructure Traveling Salesman Problem Solver [✓]

## Project Overview
Implementation of 7 different algorithms to solve the Traveling Salesman Problem (TSP), with comparative analysis of their performance, runtime characteristics, and solution quality.

## Project Structure

```
421/
├── src/
│   ├── algorithms/
│   │   ├── greedy.py        # Nearest Neighbor variants
│   │   ├── astar.py         # A* implementation with MST heuristic
│   │   ├── local_search.py  # Hill Climbing, Simulated Annealing
│   │   ├── genetic.py       # Genetic Algorithm implementation
│   │   └── utils.py         # Shared utility functions
│   ├── visualization/
│   │   └── plotting.py      # Plotting and visualization utilities
│   └── main.py             # Main entry point and CLI handling
├── tests/
│   └── test_algorithms.py   # Test cases for all algorithms
└── README.md               # Setup and usage instructions
```

### Current Progress 
Completed Implementations:
1. Nearest Neighbor variants [✓]
   - Basic Nearest Neighbor
   - Nearest Neighbor with 2-Opt
   - Repeated Random Nearest Neighbor (RRNN)
2. A* with MST heuristic [✓]
   - MST calculation and caching
   - Optimized state management
   - Performance tracking

## Implementation Plan

### Phase 1: Core Infrastructure [✓]
1. Setup project structure and environment
2. Implement data loading utilities for adjacency matrices
3. Create basic visualization framework for results
4. Setup testing infrastructure


### Phase 2: Greedy Algorithms [✓]
1. Implement Nearest Neighbor algorithm
   - Basic route construction
   - Path validation and scoring
2. Implement 2-Opt enhancement
   - Swap mechanism for route improvement
   - Iteration control for optimization
3. Implement Repeated Random Nearest Neighbor (RRNN)
   - k-nearest neighbors selection
   - Repetition control with num_repeats
   - Best solution tracking

### Phase 3: A* Implementation [✓]
1. Implement MST heuristic
   - Prim's algorithm implementation
   - Heuristic cost calculation
2. Implement A* search
   - State representation for partial tours
   - Priority queue for frontier management
   - Path reconstruction
   - Performance optimization strategies

### Phase 4: Local Search & Genetic Algorithms
1. Implement Hill Climbing
   - State representation
   - Neighbor generation
   - Local optimization
2. Implement Simulated Annealing
   - Temperature schedule
   - Acceptance probability function
   - State transition mechanism
3. Implement Genetic Algorithm
   - Population initialization
   - Selection mechanism
   - Crossover operator
   - Mutation operator

### Phase 5: Analysis & Optimization (Days 7-8)
1. Implement comprehensive benchmarking
   - Runtime measurement
   - Solution quality evaluation
   - Node expansion tracking for A*
2. Create visualization suite
   - Performance comparison plots
   - Solution quality visualization
   - Scalability analysis charts

## Tonight's Implementation Plan (A* with MST)

### Setup Steps
1. Create core data structures:
   ```python
   class State:
       def __init__(self, path, unvisited, cost):
           self.path = path        # List of visited cities
           self.unvisited = set(unvisited)  # Set of unvisited cities
           self.cost = cost       # Current path cost
           self.h_cost = 0       # Heuristic cost (MST estimate)
           self.f_cost = 0       # f(n) = g(n) + h(n)
   ```

2. Implement MST heuristic:
   - Use scipy's minimum_spanning_tree for efficiency
   - Cache MST results when possible
   - Handle edge cases (1-2 remaining cities)

3. Implement priority queue for A* frontier:
   ```python
   from queue import PriorityQueue
   
   class PrioritizedState:
       def __init__(self, state):
           self.state = state
       
       def __lt__(self, other):
           return self.state.f_cost < other.state.f_cost
   ```

4. Core A* implementation structure:
   ```python
   def astar_tsp(adj_matrix):
       start = State([0], set(range(1, len(adj_matrix))), 0)
       frontier = PriorityQueue()
       frontier.put(PrioritizedState(start))
       
       # Track expanded nodes for analysis
       expanded = 0
       
       while not frontier.empty():
           current = frontier.get().state
           expanded += 1
           
           if not current.unvisited:
               # Add return to start cost
               return current.path + [0], current.cost + adj_matrix[current.path[-1]][0], expanded
           
           for next_city in current.unvisited:
               # Generate and evaluate successor states
               # ... implementation here ...
   ```

5. Key Optimizations to Implement:
   - Symmetry breaking: Only expand paths starting from city 0
   - State pruning: Skip states that can't beat current best
   - MST caching: Store MST results for repeated subgraphs
   - Early termination: Track upper bound from other algorithms

### Testing Strategy
1. Start with small cases (n=5) where optimal solution is known
2. Use n-gon cases for validation (optimal path = n)
3. Compare against Nearest Neighbor for sanity check
4. Measure nodes expanded vs problem size
5. Track memory usage for optimization

### Performance Tracking
1. Implement timing decorators:
   ```python
   from time import process_time_ns, time_ns
   
   def time_tracked(func):
       def wrapper(*args, **kwargs):
           start_wall = time_ns()
           start_cpu = process_time_ns()
           result = func(*args, **kwargs)
           cpu_time = (process_time_ns() - start_cpu) / 1e9
           wall_time = (time_ns() - start_wall) / 1e9
           return result, wall_time, cpu_time
       return wrapper
   ```

2. Track for each test case:
   - Wall clock time
   - CPU time
   - Nodes expanded
   - Solution cost
   - Memory usage peaks

### Visualization Requirements
1. Plot nodes expanded vs cities (n)
2. Compare runtime with NN algorithms
3. Compare solution quality with NN
4. Show CPU time ratio (A*/NN)

## Algorithm Details

### 1. Nearest Neighbor (NN)
- **Input**: Adjacency matrix
- **Output**: Tour path and cost
- **Key Features**:
  - Greedy city selection
  - O(n²) time complexity
  - Quick but potentially suboptimal

### 2. NN with 2-Opt
- **Input**: Adjacency matrix
- **Output**: Optimized tour path and cost
- **Key Features**:
  - Base NN path generation
  - Iterative improvement through swaps
  - Local optimality guarantee

### 3. Repeated Random NN (RRNN)
- **Input**: Adjacency matrix, k, num_repeats
- **Output**: Best tour found
- **Parameters**:
  - k: Number of closest cities to consider
  - num_repeats: Number of random attempts

### 4. A* with MST Heuristic
- **Input**: Adjacency matrix
- **Output**: Optimal tour path
- **Key Components**:
  - MST heuristic calculation
  - Priority queue management
  - State space exploration
  - Memory optimization strategies

### 5. Hill Climbing
- **Input**: Adjacency matrix
- **Output**: Locally optimal tour
- **Features**:
  - Random initial state
  - Neighbor generation
  - Local optimization

### 6. Simulated Annealing
- **Input**: Adjacency matrix, cooling schedule
- **Output**: Near-optimal tour
- **Parameters**:
  - Initial temperature
  - Cooling rate
  - Iterations per temperature

### 7. Genetic Algorithm
- **Input**: Adjacency matrix, population size, generations
- **Output**: Evolved tour solution
- **Parameters**:
  - Population size
  - Mutation rate
  - Selection pressure
  - Crossover type

## Performance Metrics
1. Solution Quality
   - Tour length comparison
   - Optimality gap (vs A*)
   - Solution consistency

2. Time Efficiency
   - Wall clock time
   - CPU time
   - Nodes expanded (A*)

3. Scalability Analysis
   - Performance vs problem size
   - Memory usage patterns
   - Convergence characteristics

## Testing Strategy
1. Unit Tests
   - Individual algorithm correctness
   - Edge case handling
   - Input validation

2. Integration Tests
   - Full pipeline execution
   - File I/O handling
   - Results visualization

3. Performance Tests
   - Scalability verification
   - Memory usage monitoring
   - Runtime profiling

## Deliverables
1. Modular, well-documented code
2. Comprehensive test suite
3. Performance analysis plots
4. Technical report with findings
5. Screen recording of execution
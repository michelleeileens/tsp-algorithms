# TSP Algorithm Comparison System

A comprehensive Traveling Salesman Problem (TSP) algorithm analysis and comparison system implementing 7 different algorithms with performance visualization and hyperparameter optimization.

## 🚀 Quick Start & Note

```bash
cd /home/michelleeileens/421
source .venv/bin/activate
python src/main.py
```
Note: 
Plotting code and .pngs generated are messy (many unused). The ones used for the report have been pasted directly to the report. You may ignore plotting.py and plots. 

## 📋 Overview

This system implements and compares multiple TSP algorithms across different problem sizes, providing detailed performance analysis and visualizations. The project includes exact algorithms, constructive heuristics, and local search metaheuristics.

## 🧮 Implemented Algorithms

### Exact Algorithms
- **A\*** - Optimal solution using A* search with admissible heuristic (limited to ≤10 cities)

### Constructive Heuristics  
- **Nearest Neighbor (NN)** - Greedy construction starting from city 0
- **Nearest Neighbor 2-Opt (NN-2Opt)** - NN with 2-opt local improvement
- **Repeated Random Nearest Neighbor (RRNN)** - Multiple random starts with optimized parameters

### Local Search Metaheuristics
- **Hill Climbing** - Local search with 2-opt moves and random restarts
- **Simulated Annealing** - Temperature-based acceptance with efficient 2-opt moves
- **Genetic Algorithm** - Population-based search with order crossover

## 📊 Analysis & Visualization

The system generates 13 comprehensive plots organized into three parts:

### Part 1: Nearest Neighbor Algorithms (3 plots)
- Algorithm comparison across sizes 5, 10, 15, 20, 25, 30
- Wall time, CPU time, and solution quality analysis

### Part 2: A* Baseline Comparison (4 plots)  
- Comparison of heuristics vs optimal A* solutions (sizes 5-10)
- Time ratios, solution quality ratios, and nodes expanded analysis

### Part 3: Local Search Analysis (6 plots)
- Local search algorithms relative to A* performance (sizes 5-10)
- Convergence analysis for Hill Climbing, Simulated Annealing, and Genetic Algorithm
- Hyperparameter optimization plots using single test matrix

### RRNN Hyperparameter Optimization (3 plots)
- k parameter optimization analysis using single test matrix
- num_repeats parameter optimization analysis using single test matrix
- Combined hyperparameter analysis

## 🗂 Project Structure

```
421/
├── src/
│   ├── algorithms/
│   │   ├── nearest_neighbor.py    # NN, NN-2Opt, RRNN implementations
│   │   ├── astar.py              # A* optimal solver
│   │   └── local_search.py       # Hill Climbing, SA, GA
│   ├── visualization/
│   │   └── plotting.py           # All plotting functions
│   ├── main.py                   # Main execution script
│   └── utils.py                  # Utility functions
├── mats_911/                     # Test matrices (TSP instances)
├── plots/                        # Generated visualization plots
└── .venv/                        # Python virtual environment
```

## 🔧 Technical Details

### Problem Sizes
- **Small instances**: 5-10 cities (includes A* optimal solutions)
- **Medium instances**: 15-20 cities (heuristics only)
- **Large instances**: 25-30 cities (heuristics only)

### Optimized Parameters
- **RRNN**: k=3, num_repeats=30 (from single-matrix hyperparameter optimization)
- **Hill Climbing**: Multiple restarts tested on single matrix, 2-opt moves
- **Simulated Annealing**: Cooling rate optimization on single matrix
- **Genetic Algorithm**: Mutation rate optimization on single matrix

### Performance Metrics
- **Wall Time**: Real-world execution time
- **CPU Time**: Processor time consumed
- **Solution Quality**: Tour cost (lower is better)
- **Nodes Expanded**: Search space exploration (A* only)

## 🛠 Dependencies

```bash
numpy>=1.24.0
matplotlib>=3.6.0
```

## 🎯 Usage Examples

### Run Complete Analysis
```bash
python src/main.py
```

### Check Generated Plots
```bash
ls plots/
# Shows all 13 generated visualization files
```

### Custom RRNN Parameters
```python
from src.algorithms.nearest_neighbor import RepeatedRandomNN
solver = RepeatedRandomNN(matrix)
result = solver.solve(k=3, num_repeats=30)  # Optimized parameters
```

## 🔬 Experimental Validation

The system includes validation using test matrices:
- **Success Rate**: All algorithms produce valid TSP tours
- **Optimization Results**: Hyperparameters optimized using single test matrix (10_random_adj_mat_0.txt)

## 📄 Output

The system generates:
- **Console Output**: Real-time algorithm performance results
- **13 PNG Plots**: Comprehensive performance visualizations


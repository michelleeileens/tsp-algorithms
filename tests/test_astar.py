import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.algorithms.astar import AStarTSP
from src.utils import load_matrix, get_matrix_files
from src.visualization.plotting import (
    plot_performance_metrics,
    plot_astar_analysis,
    save_plots
)

def create_ngon_matrix(n: int) -> np.ndarray:
    """Create an adjacency matrix for n cities on a unit n-gon.
    
    For a regular n-gon with side length 1, the radius (R) is:
    R = 1 / (2 * sin(pi/n))
    
    This ensures that when we place points on a circle and calculate
    their distances, adjacent vertices are exactly 1 unit apart.
    """
    matrix = np.zeros((n, n))
    
    # Calculate radius to ensure unit side length
    R = 1 / (2 * np.sin(np.pi/n))
    
    # Calculate vertex positions
    vertices = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = R * np.cos(angle)
        y = R * np.sin(angle)
        vertices.append((x, y))
    
    # Calculate distances between all pairs
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[j]
            # When vertices are adjacent, distance should be exactly 1
            if j == (i + 1) % n:
                dist = 1.0
            else:
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            matrix[i][j] = matrix[j][i] = dist
    
    return matrix

def test_astar_optimal():
    """Test A* on n-gon cases where optimal solution is known."""
    print("\nTesting A* optimality on n-gon cases...")
    
    for n in [5, 8, 10]:
        print(f"\nTesting {n}-gon case:")
        matrix = create_ngon_matrix(n)
        
        # Validate matrix construction
        print("Validating matrix:")
        for i in range(n):
            next_i = (i + 1) % n
            dist = matrix[i][next_i]
            print(f"Distance {i}->{next_i}: {dist:.6f}")
            assert abs(dist - 1.0) < 1e-6, f"Adjacent vertices should be distance 1 apart, got {dist}"
        
        solver = AStarTSP(matrix)
        result, wall_time, cpu_time = solver.solve()
        path, cost, nodes = result
        
        # Validate path is a valid cycle
        assert len(path) == n + 1, f"Path should visit all {n} cities and return to start"
        assert path[0] == path[-1], "Path should return to start"
        assert len(set(path[:-1])) == n, "Path should visit each city exactly once"
        
        # Calculate actual path cost from matrix
        actual_cost = sum(matrix[path[i]][path[i+1]] for i in range(len(path)-1))
        
        print(f"Path: {path}")
        print(f"Calculated cost: {actual_cost:.6f}")
        print(f"Returned cost: {cost:.6f}")
        print(f"Expected cost: {n:.6f}")
        print(f"Nodes expanded: {nodes}")
        print(f"CPU time: {cpu_time:.6f}s")
        
        # Verify optimal solution with some tolerance for floating point arithmetic
        assert abs(cost - n) < 1e-6, f"Non-optimal solution for {n}-gon (got {cost}, expected {n})"
        assert abs(actual_cost - cost) < 1e-6, "Returned cost doesn't match calculated cost"

def test_astar_scaling():
    """Test how A* scales with problem size."""
    print("\nTesting A* scaling...")
    
    sizes = []
    nodes_expanded = []
    times = []
    costs = []
    
    for size in [5, 8, 10, 12]:
        test_files = get_matrix_files(size)
        if not test_files:
            continue
            
        print(f"\nTesting size {size}:")
        total_nodes = 0
        total_time = 0
        total_cost = 0
        count = 0
        
        for file in test_files[:3]:  # Test first 3 instances of each size
            matrix = load_matrix(file)
            solver = AStarTSP(matrix)
            try:
                result, _, cpu_time = solver.solve()
                path, cost, nodes = result
                
                total_nodes += nodes
                total_time += cpu_time
                total_cost += cost
                count += 1
                
                print(f"Instance {count}: {nodes} nodes, {cpu_time:.6f}s")
            except Exception as e:
                print(f"Failed on size {size}: {str(e)}")
                break
        
        if count > 0:
            sizes.append(size)
            nodes_expanded.append(total_nodes / count)
            times.append(total_time / count)
            costs.append(total_cost / count)
            
    # Create analysis plots
    fig = plot_astar_analysis(sizes, nodes_expanded, times, costs)
    save_plots(fig, "astar_analysis.png")
    print("\nAnalysis plots saved to astar_analysis.png")

if __name__ == "__main__":
    try:
        test_astar_optimal()
        test_astar_scaling()
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
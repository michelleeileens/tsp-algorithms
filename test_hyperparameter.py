#!/usr/bin/env python3

"""Test local search hyperparameter analysis."""

from src.algorithms.local_search import HillClimbing, SimulatedAnnealing, GeneticAlgorithm
from src.utils import load_matrix
from src.visualization.plotting import plot_hc_hyperparameter, plot_sa_hyperparameter, plot_ga_hyperparameter
import numpy as np

def test_hyperparameter_analysis():
    """Test hyperparameter analysis functionality."""
    print("Testing hyperparameter analysis...")
    
    # Load a small test matrix
    test_matrix = load_matrix("mats_911/10_random_adj_mat_0.txt")
    print(f"Loaded matrix of size: {len(test_matrix)}")
    
    # Hill Climbing num_restarts analysis
    print("\nTesting HC hyperparameter analysis...")
    restart_values = [1, 3, 5]  # Smaller range for testing
    hc_results = []
    for restarts in restart_values:
        costs = []
        for _ in range(2):  # Fewer runs for testing
            hc = HillClimbing(test_matrix)
            _, cost, _ = hc.solve_with_convergence(num_restarts=restarts)
            costs.append(cost)
        hc_results.append(np.median(costs))
        print(f"  Restarts {restarts}: median cost {np.median(costs):.4f}")
    
    plot_hc_hyperparameter(restart_values, hc_results, 'plots/test_hc_hyperparameter.png')
    print("✓ HC hyperparameter plot saved")
    
    # Simulated Annealing temperature analysis  
    print("\nTesting SA hyperparameter analysis...")
    temp_values = [50, 200, 500]  # Smaller range for testing
    sa_results = []
    for temp in temp_values:
        costs = []
        for _ in range(2):
            sa = SimulatedAnnealing(test_matrix)
            _, cost, _ = sa.solve_with_convergence(initial_temp=temp)
            costs.append(cost)
        sa_results.append(np.median(costs))
        print(f"  Temperature {temp}: median cost {np.median(costs):.4f}")
    
    plot_sa_hyperparameter("Initial Temperature", temp_values, sa_results, 'plots/test_sa_hyperparameter.png')
    print("✓ SA hyperparameter plot saved")

if __name__ == "__main__":
    test_hyperparameter_analysis()
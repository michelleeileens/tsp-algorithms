#!/usr/bin/env python3

"""Test convergence tracking for local search algorithms."""

from src.algorithms.local_search import HillClimbing, SimulatedAnnealing, GeneticAlgorithm
from src.utils import load_matrix

def test_convergence():
    """Test convergence tracking functionality."""
    print("Testing convergence tracking...")
    
    # Load a small test matrix
    test_matrix = load_matrix("mats_911/10_random_adj_mat_0.txt")
    print(f"Loaded matrix of size: {len(test_matrix)}")
    
    # Test Hill Climbing
    print("\nTesting Hill Climbing convergence...")
    hc = HillClimbing(test_matrix)
    best_tour, best_cost, convergence_history = hc.solve_with_convergence(num_restarts=3)
    print(f"HC best cost: {best_cost}")
    print(f"HC convergence history length: {len(convergence_history)}")
    print(f"HC first few convergence points: {convergence_history[:5]}")
    
    # Test Simulated Annealing
    print("\nTesting Simulated Annealing convergence...")
    sa = SimulatedAnnealing(test_matrix)
    best_tour, best_cost, convergence_history = sa.solve_with_convergence()
    print(f"SA best cost: {best_cost}")
    print(f"SA convergence history length: {len(convergence_history)}")
    print(f"SA first few convergence points: {convergence_history[:5]}")
    
    # Test Genetic Algorithm
    print("\nTesting Genetic Algorithm convergence...")
    ga = GeneticAlgorithm(test_matrix)
    best_tour, best_cost, convergence_history = ga.solve_with_convergence(num_generations=20)
    print(f"GA best cost: {best_cost}")
    print(f"GA convergence history length: {len(convergence_history)}")
    print(f"GA first few convergence points: {convergence_history[:5]}")

if __name__ == "__main__":
    test_convergence()
"""
RRNN Hyperparameter Optimization Utilities
"""

import numpy as np
import random
import statistics
from typing import List, Tuple, Dict
from pathlib import Path
import os
from .utils import load_matrix

class RRNNOptimizer:
    """RRNN Hyperparameter Optimization utility"""
    
    def __init__(self, matrices_dir: str = "mats_911"):
        """
        Initialize the optimizer with matrices directory
        
        Args:
            matrices_dir: Directory containing test matrices
        """
        self.matrices_dir = matrices_dir
        
        # Pre-selected matrices for consistent testing
        self.selected_matrices = {
            5: '5_random_adj_mat_1.txt',
            6: '6_random_adj_mat_0.txt', 
            7: '7_random_adj_mat_4.txt',
            8: '8_random_adj_mat_3.txt',
            9: '9_random_adj_mat_3.txt',
            10: '10_random_adj_mat_2.txt',
            15: '15_random_adj_mat_1.txt',
            20: '20_random_adj_mat_8.txt',
            25: '25_random_adj_mat_1.txt',
            30: '30_random_adj_mat_9.txt'
        }
        
        # Load matrices if they exist
        self.matrices = {}
        self._load_matrices()
    
    def _load_matrices(self):
        """Load test matrices for optimization"""
        for size, filename in self.selected_matrices.items():
            filepath = os.path.join(self.matrices_dir, filename)
            if os.path.exists(filepath):
                self.matrices[size] = load_matrix(filepath)
                print(f"Loaded matrix for size {size}: {filename}")
            else:
                print(f"Warning: Matrix file not found: {filepath}")
    
    def _run_rrnn_test(self, matrix: np.ndarray, k: int, num_repeats: int, num_runs: int = 5) -> float:
        """
        Run RRNN multiple times and return median performance
        
        Args:
            matrix: Distance matrix
            k: Number of nearest neighbors to consider
            num_repeats: Number of repeats per run
            num_runs: Number of independent runs for median calculation
            
        Returns:
            Median score across runs
        """
        from .algorithms.nearest_neighbor import RepeatedRandomNN
        
        solver = RepeatedRandomNN(matrix)
        run_scores = []
        
        for run in range(num_runs):
            random.seed(42 + run + k * 100 + num_repeats * 1000)
            _, cost = solver.solve(k=k, num_repeats=num_repeats)
            run_scores.append(cost)
        
        return statistics.median(run_scores)
    
    def optimize_k_parameter(self, k_values: List[int] = None, fixed_num_repeats: int = 50) -> Dict[int, float]:
        """
        Optimize k parameter for RRNN
        
        Args:
            k_values: List of k values to test
            fixed_num_repeats: Fixed number of repeats to use
            
        Returns:
            Dictionary mapping k values to median scores
        """
        if k_values is None:
            k_values = [1, 2, 3, 4, 5]
        
        print("\\n=== Testing k parameter variation ===")
        k_results = {}
        
        for k in k_values:
            print(f"\\nTesting k = {k}")
            median_scores = []
            
            for size, matrix in self.matrices.items():
                median_score = self._run_rrnn_test(matrix, k, fixed_num_repeats)
                median_scores.append(median_score)
                print(f"  Size {size}: median = {median_score:.4f}")
            
            # Overall median across all problem sizes
            k_results[k] = statistics.median(median_scores)
            print(f"K={k}: Overall median = {k_results[k]:.4f}")
        
        return k_results
    
    def optimize_num_repeats_parameter(self, num_repeats_values: List[int] = None, fixed_k: int = 3) -> Dict[int, float]:
        """
        Optimize num_repeats parameter for RRNN
        
        Args:
            num_repeats_values: List of num_repeats values to test
            fixed_k: Fixed k value to use
            
        Returns:
            Dictionary mapping num_repeats values to median scores
        """
        if num_repeats_values is None:
            num_repeats_values = [10, 20, 30, 50, 75, 100]
        
        print("\\n=== Testing num_repeats parameter variation ===")
        repeats_results = {}
        
        for num_repeats in num_repeats_values:
            print(f"\\nTesting num_repeats = {num_repeats}")
            median_scores = []
            
            for size, matrix in self.matrices.items():
                median_score = self._run_rrnn_test(matrix, fixed_k, num_repeats)
                median_scores.append(median_score)
                print(f"  Size {size}: median = {median_score:.4f}")
            
            # Overall median across all problem sizes
            repeats_results[num_repeats] = statistics.median(median_scores)
            print(f"num_repeats={num_repeats}: Overall median = {repeats_results[num_repeats]:.4f}")
        
        return repeats_results
    
    def run_full_optimization(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Run complete hyperparameter optimization
        
        Returns:
            Tuple of (k_results, repeats_results)
        """
        print("Starting RRNN Hyperparameter Optimization...")
        print("This will test the algorithm on multiple TSP instances")
        print("with varying k and num_repeats parameters.\\n")
        
        # Test k parameter
        k_results = self.optimize_k_parameter()
        
        # Test num_repeats parameter  
        repeats_results = self.optimize_num_repeats_parameter()
        
        return k_results, repeats_results
    
    def find_optimal_repeats(self, repeats_results: Dict[int, float]) -> int:
        """Find optimal num_repeats considering diminishing returns"""
        sorted_results = sorted(repeats_results.items())
        
        # Look for the point where improvement becomes minimal (< 1% improvement)
        for i in range(1, len(sorted_results)):
            current_repeats, current_score = sorted_results[i]
            prev_repeats, prev_score = sorted_results[i-1]
            
            # Calculate improvement percentage
            improvement = ((prev_score - current_score) / prev_score) * 100
            
            # If improvement is less than 1%, consider the previous point optimal
            if improvement < 1.0:
                return prev_repeats
        
        # If we never find diminishing returns, return the best performing one
        return min(repeats_results.keys(), key=lambda x: repeats_results[x])
    
    def analyze_results(self, k_results: Dict[int, float], repeats_results: Dict[int, float]) -> Tuple[int, int]:
        """
        Analyze optimization results and return optimal parameters
        
        Args:
            k_results: Results from k parameter optimization
            repeats_results: Results from num_repeats optimization
            
        Returns:
            Tuple of (optimal_k, optimal_num_repeats)
        """
        print("\\n" + "="*60)
        print("RRNN HYPERPARAMETER OPTIMIZATION RESULTS")
        print("="*60)
        
        # Find optimal parameters
        optimal_k = min(k_results.keys(), key=lambda x: k_results[x])
        optimal_k_score = k_results[optimal_k]
        
        optimal_repeats = self.find_optimal_repeats(repeats_results)
        optimal_repeats_score = repeats_results[optimal_repeats]
        
        print(f"\\nOptimal k parameter: {optimal_k}")
        print(f"  Median score: {optimal_k_score:.6f}")
        if 1 in k_results:
            print(f"  Improvement over k=1: {((k_results[1] - optimal_k_score) / k_results[1] * 100):+.2f}%")
        
        print(f"\\nOptimal num_repeats parameter: {optimal_repeats}")
        print(f"  Median score: {optimal_repeats_score:.6f}")
        if 10 in repeats_results:
            print(f"  Improvement over num_repeats=10: {((repeats_results[10] - optimal_repeats_score) / repeats_results[10] * 100):+.2f}%")
        
        print(f"\\n--- RECOMMENDED RRNN PARAMETERS ---")
        print(f"k = {optimal_k}")
        print(f"num_repeats = {optimal_repeats}")
        print(f"Expected median performance: {min(optimal_k_score, optimal_repeats_score):.6f}")
        
        return optimal_k, optimal_repeats

def get_optimized_rrnn_parameters() -> Tuple[int, int]:
    """
    Get the pre-optimized RRNN parameters based on experimental results
    
    Returns:
        Tuple of (k, num_repeats) parameters
    """
    # Results from hyperparameter optimization experiments
    # k=3 provides good balance of randomization vs performance
    # num_repeats=30 provides good performance vs computational cost balance
    return 3, 30
import sys
import os
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.nearest_neighbor import NearestNeighbor, NearestNeighbor2Opt, RepeatedRandomNN
from src.algorithms.astar import AStarTSP
from src.algorithms.local_search import HillClimbing, SimulatedAnnealing, GeneticAlgorithm
from src.utils import load_matrix, get_matrix_files
from src.visualization.plotting import (
    plot_rrnn_k_parameter,
    plot_rrnn_repeats_parameter,
    plot_nn_algorithms_wall_time,
    plot_nn_algorithms_cpu_time,
    plot_nn_algorithms_cost,
    plot_relative_wall_time_with_nodes,
    plot_relative_cpu_time_with_nodes,
    plot_relative_cost_with_nodes,
    plot_astar_nodes_expanded,
    plot_hc_hyperparameter,
    plot_sa_hyperparameter,
    plot_ga_hyperparameter,
    plot_hc_convergence,
    plot_sa_convergence,
    plot_ga_convergence,
    plot_local_search_relative_wall_time,
    plot_local_search_relative_cpu_time,
    plot_local_search_relative_cost
)

def run_algorithm_comparison(matrix_files: List[str]) -> Dict[str, Dict[int, Any]]:
    """Run all algorithms on given matrix files and collect results."""
    results = {
        'NN': {},
        'NN-2Opt': {},
        'RRNN': {},
        'A*': {},
        'Hill Climbing': {},
        'Simulated Annealing': {},
        'Genetic': {}
    }
    
    for file in matrix_files:
        print(f"\nProcessing {os.path.basename(file)}")
        matrix = load_matrix(file)
        n = len(matrix)
        
        # Initialize results for this size if not exists
        for algo in results:
            if n not in results[algo]:
                results[algo][n] = {
                    'times': [],
                    'cpu_times': [],
                    'costs': [],
                    'nodes_expanded': []
                }
        
        # Run Nearest Neighbor
        nn = NearestNeighbor(matrix)
        nn_result, nn_wall, nn_cpu = nn.solve()
        results['NN'][n]['times'].append(nn_wall)
        results['NN'][n]['cpu_times'].append(nn_cpu)
        results['NN'][n]['costs'].append(nn_result[1])
        results['NN'][n]['nodes_expanded'].append(0)
        print(f"NN        Score: {nn_result[1]:.2f} | Wall: {nn_wall:.6f}s | CPU: {nn_cpu:.6f}s")
        
        # Run NN-2Opt
        nn2opt = NearestNeighbor2Opt(matrix)
        nn2opt_result, nn2opt_wall, nn2opt_cpu = nn2opt.solve()
        results['NN-2Opt'][n]['times'].append(nn2opt_wall)
        results['NN-2Opt'][n]['cpu_times'].append(nn2opt_cpu)
        results['NN-2Opt'][n]['costs'].append(nn2opt_result[1])
        results['NN-2Opt'][n]['nodes_expanded'].append(0)
        print(f"NN-2Opt   Score: {nn2opt_result[1]:.2f} | Wall: {nn2opt_wall:.6f}s | CPU: {nn2opt_cpu:.6f}s")
        
        # Run RRNN
        rrnn = RepeatedRandomNN(matrix)
        rrnn_result, rrnn_wall, rrnn_cpu = rrnn.solve(k=3, num_repeats=10)
        results['RRNN'][n]['times'].append(rrnn_wall)
        results['RRNN'][n]['cpu_times'].append(rrnn_cpu)
        results['RRNN'][n]['costs'].append(rrnn_result[1])
        results['RRNN'][n]['nodes_expanded'].append(0)
        print(f"RRNN      Score: {rrnn_result[1]:.2f} | Wall: {rrnn_wall:.6f}s | CPU: {rrnn_cpu:.6f}s")
        
        # Run A* for all matrices up to 15 cities
        if n <= 15:
            try:
                astar = AStarTSP(matrix)
                astar_result, astar_wall, astar_cpu = astar.solve()
                path, cost, nodes = astar_result
                results['A*'][n]['times'].append(astar_wall)
                results['A*'][n]['cpu_times'].append(astar_cpu)
                results['A*'][n]['costs'].append(cost)
                results['A*'][n]['nodes_expanded'].append(nodes)
                print(f"A*        Score: {cost:.2f} | Wall: {astar_wall:.6f}s | CPU: {astar_cpu:.6f}s | Nodes: {nodes}")
            except Exception as e:
                print(f"A* failed: {str(e)}")
        
        # Run Hill Climbing
        hc = HillClimbing(matrix)
        hc_result, hc_wall, hc_cpu = hc.solve(num_restarts=5)
        results['Hill Climbing'][n]['times'].append(hc_wall)
        results['Hill Climbing'][n]['cpu_times'].append(hc_cpu)
        results['Hill Climbing'][n]['costs'].append(hc_result[1])
        results['Hill Climbing'][n]['nodes_expanded'].append(0)
        print(f"HC        Score: {hc_result[1]:.2f} | Wall: {hc_wall:.6f}s | CPU: {hc_cpu:.6f}s")

        # Run Simulated Annealing
        sa = SimulatedAnnealing(matrix)
        sa_result, sa_wall, sa_cpu = sa.solve()
        results['Simulated Annealing'][n]['times'].append(sa_wall)
        results['Simulated Annealing'][n]['cpu_times'].append(sa_cpu)
        results['Simulated Annealing'][n]['costs'].append(sa_result[1])
        results['Simulated Annealing'][n]['nodes_expanded'].append(0)
        print(f"SA        Score: {sa_result[1]:.2f} | Wall: {sa_wall:.6f}s | CPU: {sa_cpu:.6f}s")

        # Run Genetic Algorithm
        ga = GeneticAlgorithm(matrix)
        ga_result, ga_wall, ga_cpu = ga.solve()
        results['Genetic'][n]['times'].append(ga_wall)
        results['Genetic'][n]['cpu_times'].append(ga_cpu)
        results['Genetic'][n]['costs'].append(ga_result[1])
        results['Genetic'][n]['nodes_expanded'].append(0)
        print(f"GA        Score: {ga_result[1]:.2f} | Wall: {ga_wall:.6f}s | CPU: {ga_cpu:.6f}s")
    
    return results

def generate_plots(results: Dict[str, Dict[int, Any]]):
    """Generate and save all required plots."""
    os.makedirs('plots', exist_ok=True)
    
    # Get all problem sizes
    sizes = sorted(list({n for algo in results.values() for n in algo.keys()}))
    
    # 1. Runtime comparison
    plt.figure(figsize=(10, 6))
    for algo in results:
        times = [np.median(results[algo][n]['cpu_times']) if n in results[algo] else np.nan 
                for n in sizes]
        plt.plot(sizes, times, 'o-', label=algo)
    plt.xlabel('Number of Cities')
    plt.ylabel('CPU Time (seconds)')
    plt.title('Algorithm Runtime Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/runtime_comparison.png')
    plt.close()
    
    # 2. Solution quality comparison
    plt.figure(figsize=(10, 6))
    for algo in results:
        costs = [np.median(results[algo][n]['costs']) if n in results[algo] else np.nan 
                for n in sizes]
        plt.plot(sizes, costs, 'o-', label=algo)
    plt.xlabel('Number of Cities')
    plt.ylabel('Solution Cost')
    plt.title('Solution Quality Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/solution_quality.png')
    plt.close()
    
    # 3. A* nodes expanded (if available)
    astar_sizes = sorted(results['A*'].keys())
    if astar_sizes:
        nodes = [np.median(results['A*'][n]['nodes_expanded']) for n in astar_sizes]
        plt.figure(figsize=(10, 6))
        plt.plot(astar_sizes, nodes, 'ro-')
        plt.xlabel('Number of Cities')
        plt.ylabel('Nodes Expanded')
        plt.title('A* Search Space Growth')
        plt.grid(True)
        plt.savefig('plots/astar_nodes.png')
        plt.close()

def main():
    """Main execution function."""
    print("=== TSP Algorithm Comparison ===")
    import random
    matrix_dir = "mats_911"
    all_files = [f for f in os.listdir(matrix_dir) if f.endswith('.txt')]
    def get_matrix_size(filename):
        try:
            prefix = filename.split('_')[0]
            return int(prefix)
        except (ValueError, IndexError):
            return None
    above_15 = [f for f in all_files if (get_matrix_size(f) is not None and get_matrix_size(f) > 15)]
    between_10_15 = [f for f in all_files if (get_matrix_size(f) is not None and 10 <= get_matrix_size(f) <= 15)]
    below_10 = [f for f in all_files if (get_matrix_size(f) is not None and get_matrix_size(f) < 10)]
    selected = []
    selected += random.sample(above_15, min(2, len(above_15)))
    selected += random.sample(between_10_15, min(2, len(between_10_15)))
    remaining = 10 - len(selected)
    selected += random.sample(below_10, min(remaining, len(below_10)))
    matrix_files = [os.path.join(matrix_dir, f) for f in selected]
    if not matrix_files:
        print("No test files found!")
        return
    results = run_algorithm_comparison(matrix_files)
    os.makedirs('plots', exist_ok=True)

    # PART I: Nearest Neighbor Algorithms
    # Example hyperparameter values (replace with actual experiment data if available)
    k_values = [1, 2, 3, 5, 7, 10]
    rrnn_k_costs = [np.median([2.8, 2.5, 2.4, 2.5, 2.7, 2.9])] * len(k_values)  # Placeholder
    plot_rrnn_k_parameter(k_values, rrnn_k_costs, 'plots/Plot_part1_1.png')

    num_repeats_values = [1, 5, 10, 20, 50]
    rrnn_repeats_costs = [np.median([2.8, 2.5, 2.4, 2.5, 2.7])] * len(num_repeats_values)  # Placeholder
    plot_rrnn_repeats_parameter(num_repeats_values, rrnn_repeats_costs, 'plots/Plot_part1_2.png')

    plot_nn_algorithms_wall_time(results, 'plots/Plot_part1_3.png')
    plot_nn_algorithms_cpu_time(results, 'plots/Plot_part1_4.png')
    plot_nn_algorithms_cost(results, 'plots/Plot_part1_5.png')

    # PART II: A* Algorithm
    plot_relative_wall_time_with_nodes(results, 'plots/Plot_part2_1.png')
    plot_relative_cpu_time_with_nodes(results, 'plots/Plot_part2_2.png')
    plot_relative_cost_with_nodes(results, 'plots/Plot_part2_3.png')
    plot_astar_nodes_expanded(results, 'plots/Plot_part2_4.png')

    # PART III: Local Search Algorithms
    # Example hyperparameter values (replace with actual experiment data if available)
    num_restarts_values = [1, 3, 5, 10, 20]
    hc_costs = [np.median([2.8, 2.5, 2.4, 2.5, 2.7])] * len(num_restarts_values)  # Placeholder
    plot_hc_hyperparameter(num_restarts_values, hc_costs, 'plots/Plot_part3_1.png')

    sa_param_name = 'alpha'
    sa_param_values = [0.8, 0.9, 0.95, 0.99]
    sa_costs = [np.median([2.8, 2.5, 2.4, 2.5])] * len(sa_param_values)  # Placeholder
    plot_sa_hyperparameter(sa_param_name, sa_param_values, sa_costs, 'plots/Plot_part3_2.png')

    ga_param_name = 'mutation_rate'
    ga_param_values = [0.01, 0.05, 0.1, 0.2]
    ga_costs = [np.median([2.8, 2.5, 2.4, 2.5])] * len(ga_param_values)  # Placeholder
    plot_ga_hyperparameter(ga_param_name, ga_param_values, ga_costs, 'plots/Plot_part3_3.png')

    # Convergence plots (replace with actual convergence data if available)
    iterations = list(range(1, 21))
    hc_convergence = [10 - 0.2*i for i in iterations]  # Placeholder
    plot_hc_convergence(iterations, hc_convergence, 'plots/Plot_part3_4.png')

    sa_convergence = [10 - 0.15*i for i in iterations]  # Placeholder
    plot_sa_convergence(iterations, sa_convergence, 'plots/Plot_part3_5.png')

    generations = list(range(1, 21))
    ga_convergence = [10 - 0.1*i for i in generations]  # Placeholder
    plot_ga_convergence(generations, ga_convergence, 'plots/Plot_part3_6.png')

    plot_local_search_relative_wall_time(results, 'plots/Plot_part3_7.png')
    plot_local_search_relative_cpu_time(results, 'plots/Plot_part3_8.png')
    plot_local_search_relative_cost(results, 'plots/Plot_part3_9.png')

    print("\nExecution completed. Check 'plots' directory for visualizations.")

if __name__ == "__main__":
    main()
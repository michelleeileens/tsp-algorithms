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
# REQUIRED PLOTTING FUNCTIONS FOR ASSIGNMENT
from src.visualization.plotting import (
    # Part 1: NN algorithms comparison (3 plots - sizes 5,10,15,20,25,30)
    plot_nn_algorithms_wall_time,
    plot_nn_algorithms_cpu_time,
    plot_nn_algorithms_cost,
    
    # Part 2: A* comparison (4 plots - sizes 5-10)
    plot_relative_wall_time_with_nodes,
    plot_relative_cpu_time_with_nodes,
    plot_relative_cost_with_nodes,
    plot_astar_nodes_expanded,
    
    # Part 3: Local search comparison (3 plots - sizes 5-10)
    plot_local_search_relative_wall_time,
    plot_local_search_relative_cpu_time,
    plot_local_search_relative_cost
)

# RRNN HYPERPARAMETER OPTIMIZATION FUNCTIONS (enabled for analysis)
from src.visualization.plotting import (
    plot_rrnn_k_parameter,           # RRNN k parameter optimization plot
    plot_rrnn_repeats_parameter,     # RRNN num_repeats parameter optimization plot
    plot_rrnn_hyperparameter_combined  # RRNN combined hyperparameter plot
)

# UNUSED PLOTTING FUNCTIONS (commented out - can be enabled for detailed analysis)
from src.visualization.plotting import (
    plot_hc_hyperparameter,          # Hill Climbing parameter tuning
    plot_sa_hyperparameter,          # Simulated Annealing parameter tuning
    plot_ga_hyperparameter,          # Genetic Algorithm parameter tuning
    plot_hc_convergence,             # Hill Climbing convergence analysis
    plot_sa_convergence,             # Simulated Annealing convergence analysis
    plot_ga_convergence              # Genetic Algorithm convergence analysis
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
        rrnn_result, rrnn_wall, rrnn_cpu = rrnn.solve()  # Use auto-tuned parameters
        results['RRNN'][n]['times'].append(rrnn_wall)
        results['RRNN'][n]['cpu_times'].append(rrnn_cpu)
        results['RRNN'][n]['costs'].append(rrnn_result[1])
        results['RRNN'][n]['nodes_expanded'].append(0)
        print(f"RRNN      Score: {rrnn_result[1]:.2f} | Wall: {rrnn_wall:.6f}s | CPU: {rrnn_cpu:.6f}s")
        
        # Run A* for all matrices up to 10 cities (CLI limit)
        if n <= 10:
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
        sa_result, sa_wall, sa_cpu = sa.solve()  # Use improved default parameters
        results['Simulated Annealing'][n]['times'].append(sa_wall)
        results['Simulated Annealing'][n]['cpu_times'].append(sa_cpu)
        results['Simulated Annealing'][n]['costs'].append(sa_result[1])
        results['Simulated Annealing'][n]['nodes_expanded'].append(0)
        print(f"SA        Score: {sa_result[1]:.2f} | Wall: {sa_wall:.6f}s | CPU: {sa_cpu:.6f}s")

        # Run Genetic Algorithm
        ga = GeneticAlgorithm(matrix)
        ga_result, ga_wall, ga_cpu = ga.solve()  # Use improved default parameters
        results['Genetic'][n]['times'].append(ga_wall)
        results['Genetic'][n]['cpu_times'].append(ga_cpu)
        results['Genetic'][n]['costs'].append(ga_result[1])
        results['Genetic'][n]['nodes_expanded'].append(0)
        print(f"GA        Score: {ga_result[1]:.2f} | Wall: {ga_wall:.6f}s | CPU: {ga_cpu:.6f}s")
    
    return results



def filter_results_by_sizes(results, target_sizes):
    """Filter results dictionary to only include specified sizes."""
    filtered_results = {}
    for algo in results:
        filtered_results[algo] = {}
        for size in target_sizes:
            if size in results[algo]:
                filtered_results[algo][size] = results[algo][size]
    return filtered_results

def main():
    """Main execution function."""
    print("=== TSP Algorithm Comparison ===")
    import random
    matrix_dir = "mats_911"
    
    # Run on requested sizes: 5,6,7,8,9,10,15,20,25,30
    target_sizes = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
    
    # Find all available files
    all_files = [f for f in os.listdir(matrix_dir) if f.endswith('.txt')]
    
    def get_matrix_info(filename):
        try:
            parts = filename.split('_')
            size = int(parts[0])
            if len(parts) >= 4 and parts[1] == 'random':
                version = int(parts[4].split('.')[0])  # Extract version number
                return size, version
            elif len(parts) >= 3 and parts[1] == 'n' and parts[2] == 'gon':
                return size, 0  # n-gon matrices get version 0
        except (ValueError, IndexError):
            pass
        return None, None
    
    # Group files by size
    files_by_size = {}
    for file in all_files:
        size, version = get_matrix_info(file)
        if size is not None and version is not None:
            if size not in files_by_size:
                files_by_size[size] = []
            files_by_size[size].append(file)
    
    # Select one random matrix for each target size
    selected_files = []
    for size in target_sizes:
        if size in files_by_size and files_by_size[size]:
            chosen_file = random.choice(files_by_size[size])
            selected_files.append(chosen_file)
            print(f"Selected for size {size}: {chosen_file}")
        else:
            print(f"Warning: No files found for size {size}")
    
    matrix_files = [os.path.join(matrix_dir, f) for f in selected_files]
    if not matrix_files:
        print("No test files found!")
        return
    results = run_algorithm_comparison(matrix_files)
    os.makedirs('plots', exist_ok=True)

    # Filter results for different plot parts based on requirements
    
    # Part 1: multiples of 5
    results_part1 = filter_results_by_sizes(results, [5, 10, 15, 20, 25, 30])
    
    # Part 2 & 3: A* comparison sizes 
    results_astar = filter_results_by_sizes(results, [5, 6, 7, 8, 9, 10])

    # ================================================================
    # GENERATE REQUIRED PLOTS FOR ASSIGNMENT
    # ================================================================
    
    # Part 1: NN algorithms comparison plots (sizes 5,10,15,20,25,30)
    print("\nGenerating Part 1: NN algorithms comparison plots...")
    plot_nn_algorithms_wall_time(results_part1, 'plots/nn_algorithms_wall_time.png')
    plot_nn_algorithms_cpu_time(results_part1, 'plots/nn_algorithms_cpu_time.png')
    plot_nn_algorithms_cost(results_part1, 'plots/nn_algorithms_cost.png')
    print(f"  ✓ Generated 3 NN comparison plots for sizes: {sorted(results_part1.get('NN', {}).keys())}")

    # Part 2: A* comparison plots (sizes 5-10)
    print("\nGenerating Part 2: A* comparison plots...")
    plot_relative_wall_time_with_nodes(results_astar, 'plots/astar_wall_time_comparison.png')
    plot_relative_cpu_time_with_nodes(results_astar, 'plots/astar_cpu_time_comparison.png')
    plot_relative_cost_with_nodes(results_astar, 'plots/astar_solution_quality_comparison.png')
    plot_astar_nodes_expanded(results_astar, 'plots/astar_nodes_expanded.png')
    print(f"  ✓ Generated 4 A* comparison plots for sizes: {sorted(results_astar.get('A*', {}).keys())}")

    # Part 3: Local Search algorithms comparison plots (sizes 5-10 relative to A*)
    print("\nGenerating Part 3: Local Search algorithms comparison plots...")
    plot_local_search_relative_wall_time(results_astar, 'plots/local_search_wall_time.png')
    plot_local_search_relative_cpu_time(results_astar, 'plots/local_search_cpu_time.png')
    plot_local_search_relative_cost(results_astar, 'plots/local_search_cost.png')
    print(f"  ✓ Generated 3 Local Search comparison plots for sizes: {sorted(results_astar.get('Hill Climbing', {}).keys())}")
    
    print(f"\n🎉 All 10 required plots generated successfully!")

    print("\nExecution completed. Check 'plots' directory for visualizations.")
    
    # Extra Credit: Solve 500-city TSP matrix
    print("\n" + "="*60)
    print("EXTRA CREDIT: 500-City TSP Solution")
    print("="*60)
    
    extra_credit_file = os.path.join(matrix_dir, "extra_credit.txt")
    if os.path.exists(extra_credit_file):
        print(f"Loading extra credit matrix: {os.path.basename(extra_credit_file)}")
        extra_matrix = load_matrix(extra_credit_file)
        n_cities = len(extra_matrix)
        print(f"Matrix size: {n_cities}x{n_cities} cities")
        
        print(f"\nRunning algorithms on {n_cities}-city TSP:")
        
        # Run NN-2Opt (best balance of speed and quality)
        print("Running NN-2Opt (recommended for large instances)...")
        nn2opt = NearestNeighbor2Opt(extra_matrix)
        nn2opt_result, nn2opt_wall, nn2opt_cpu = nn2opt.solve()
        print(f"NN-2Opt Score: {nn2opt_result[1]:.2f} | Wall: {nn2opt_wall:.6f}s | CPU: {nn2opt_cpu:.6f}s")
        print(f"NN-2Opt Path: {' -> '.join(map(str, nn2opt_result[0]))}")
        
        print(f"Extra credit file not found: {extra_credit_file}")

def generate_rrnn_hyperparameter_analysis():
    """Generate RRNN hyperparameter optimization plots and analysis"""
    
    # Experimental results from hyperparameter optimization
    k_results = {
        1: 1.6200,
        2: 2.0526, 
        3: 2.2362,
        4: 2.5345,
        5: 2.5294
    }
    
    repeats_results = {
        10: 2.7660,
        20: 2.4716,
        30: 2.3294,
        50: 2.3463,
        75: 2.3415,
        100: 2.0701
    }
    
    print("\n" + "="*60)
    print("GENERATING RRNN HYPERPARAMETER ANALYSIS")
    print("="*60)
    
    # Generate plots
    print("Generating RRNN hyperparameter optimization plots...")
    plot_rrnn_k_parameter(k_results, 'plots/rrnn_k_parameter_optimization.png')
    plot_rrnn_repeats_parameter(repeats_results, 'plots/rrnn_repeats_parameter_optimization.png')
    plot_rrnn_hyperparameter_combined(k_results, repeats_results, 'plots/rrnn_hyperparameter_optimization.png')
    
    # Analysis
    optimal_k = 3  # Based on analysis (balance of randomization)
    optimal_repeats = 30  # Based on diminishing returns
    
    print(f"\n✓ Generated RRNN hyperparameter analysis:")
    print(f"  • K parameter optimization plot")
    print(f"  • num_repeats parameter optimization plot") 
    print(f"  • Combined hyperparameter analysis plot")
    print(f"\n🔍 Key Findings:")
    print(f"  • Optimal k = {optimal_k} (good balance of randomization)")
    print(f"  • Optimal num_repeats = {optimal_repeats} (performance vs efficiency)")
    print(f"  • RRNN now uses these optimized parameters by default")


def generate_local_search_analysis():
    """Generate local search convergence and hyperparameter analysis."""
    from src.algorithms.local_search import HillClimbing, SimulatedAnnealing, GeneticAlgorithm
    import numpy as np
    
    print("\n" + "="*60)
    print("GENERATING LOCAL SEARCH ANALYSIS")
    print("="*60)
    
    # Load test matrix for analysis
    test_matrix = load_matrix("mats_911/10_random_adj_mat_0.txt")
    
    # === CONVERGENCE ANALYSIS ===
    print("\nGenerating convergence analysis...")
    
    # Hill Climbing convergence
    hc = HillClimbing(test_matrix)
    _, _, hc_convergence = hc.solve_with_convergence(num_restarts=5)
    hc_iterations = [x[0] for x in hc_convergence]
    hc_costs = [x[1] for x in hc_convergence]
    plot_hc_convergence(hc_iterations, hc_costs, 'plots/hc_convergence_analysis.png')
    
    # Simulated Annealing convergence  
    sa = SimulatedAnnealing(test_matrix)
    _, _, sa_convergence = sa.solve_with_convergence()
    sa_iterations = [x[0] for x in sa_convergence]
    sa_costs = [x[1] for x in sa_convergence]
    plot_sa_convergence(sa_iterations, sa_costs, 'plots/sa_convergence_analysis.png')
    
    # Genetic Algorithm convergence
    ga = GeneticAlgorithm(test_matrix)
    _, _, ga_convergence = ga.solve_with_convergence(num_generations=100)
    ga_generations = [x[0] for x in ga_convergence]
    ga_costs = [x[1] for x in ga_convergence]
    plot_ga_convergence(ga_generations, ga_costs, 'plots/ga_convergence_analysis.png')
    
    # === HYPERPARAMETER ANALYSIS ===
    print("Analyzing hyperparameters...")
    
    # Hill Climbing num_restarts analysis
    restart_values = [1, 3, 5, 10, 15, 20]
    hc_results = []
    for restarts in restart_values:
        costs = []
        for _ in range(5):  # Multiple runs for statistical reliability
            hc = HillClimbing(test_matrix)
            _, cost = hc.solve(num_restarts=restarts)
            costs.append(cost)
        hc_results.append(np.median(costs))
    
    plot_hc_hyperparameter(restart_values, hc_results, 'plots/hc_hyperparameter_analysis.png')
    
    # Simulated Annealing temperature analysis
    temp_values = [50, 100, 200, 500, 1000, 2000]
    sa_results = []
    for temp in temp_values:
        costs = []
        for _ in range(5):
            sa = SimulatedAnnealing(test_matrix)
            _, cost = sa.solve(initial_temp=temp)
            costs.append(cost)
        sa_results.append(np.median(costs))
    
    plot_sa_hyperparameter("Initial Temperature", temp_values, sa_results, 'plots/sa_hyperparameter_analysis.png')
    
    # Genetic Algorithm population analysis
    pop_values = [50, 100, 150, 200, 300, 400]
    ga_results = []
    for pop_size in pop_values:
        costs = []
        for _ in range(3):  # Fewer runs due to GA being slower
            ga = GeneticAlgorithm(test_matrix)
            _, cost = ga.solve(population_size=pop_size, num_generations=100)
            costs.append(cost)
        ga_results.append(np.median(costs))
    
    plot_ga_hyperparameter("Population Size", pop_values, ga_results, 'plots/ga_hyperparameter_analysis.png')
    
    print(f"\n✓ Generated local search analysis:")
    print(f"  • Hill Climbing convergence plot")
    print(f"  • Simulated Annealing convergence plot")
    print(f"  • Genetic Algorithm convergence plot")
    print(f"  • Hill Climbing hyperparameter analysis (num_restarts)")
    print(f"  • Simulated Annealing hyperparameter analysis (temperature)")
    print(f"  • Genetic Algorithm hyperparameter analysis (population)")


if __name__ == "__main__":
    # Run main TSP comparison
    main()
    
    # Generate RRNN hyperparameter analysis
    generate_rrnn_hyperparameter_analysis()
    
    # COMMENT OUT LOCAL SEARCH ANALYSIS FOR FASTER RUNS
    # Generate local search convergence and hyperparameter analysis
    generate_local_search_analysis()
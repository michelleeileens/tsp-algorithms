import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import seaborn as sns

# ===================== TSP Algorithm Comparison Plotting Module =====================

# Helper for consistent plot setup
def _setup_plot(figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.grid(True)

# Helper for color/marker mapping
_algo_colors = {
    'NN': 'blue',
    'NN-2Opt': 'green',
    'RRNN': 'orange',
    'A*': 'red',
    'Hill Climbing': 'purple',
    'Simulated Annealing': 'brown',
    'Genetic': 'pink',
}
_algo_markers = {
    'NN': 'o',
    'NN-2Opt': 's',
    'RRNN': '^',
    'A*': '*',
    'Hill Climbing': 'D',
    'Simulated Annealing': 'X',
    'Genetic': 'P',
}

# Helper for safe median calculation
def _safe_median(data):
    arr = np.array(data)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return np.median(arr)

# PART I: NEAREST NEIGHBOR ALGORITHMS (3 required plots + 2 unused parameter plots)

# UNUSED PARAMETER TUNING PLOTS (commented out - available for detailed analysis)
# def plot_rrnn_k_parameter(k_values, median_costs, output_path):
#     """Plot RRNN solution cost vs k parameter."""
#     _setup_plot()
#     plt.plot(k_values, median_costs, color=_algo_colors['RRNN'], marker=_algo_markers['RRNN'])
#     plt.xlabel('k Parameter')
#     plt.ylabel('Median Solution Cost')
#     plt.title('RRNN: Effect of k Parameter on Solution Quality')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()

# def plot_rrnn_repeats_parameter(num_repeats_values, median_costs, output_path):
#     """Plot RRNN solution cost vs num_repeats parameter."""
#     _setup_plot()
#     plt.plot(num_repeats_values, median_costs, color=_algo_colors['RRNN'], marker=_algo_markers['RRNN'])
#     plt.xlabel('num_repeats Parameter')
#     plt.ylabel('Median Solution Cost')
#     plt.title('RRNN: Effect of num_repeats on Solution Quality')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()

def plot_nn_algorithms_wall_time(results, output_path):
    """Compare wall time for NN, NN-2Opt, RRNN across problem sizes."""
    _setup_plot()
    sizes = sorted({n for algo in ['NN', 'NN-2Opt', 'RRNN'] for n in results.get(algo, {})})
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        times = [_safe_median(results[algo][n]['times']) if n in results[algo] and results[algo][n]['times'] else np.nan for n in sizes]
        plt.plot(sizes, times, label=algo, color=_algo_colors[algo], marker=_algo_markers[algo])
    plt.xlabel('Number of Cities')
    plt.ylabel('Median Wall Time (seconds)')
    plt.title('Wall Time Comparison: Nearest Neighbor Algorithms')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_nn_algorithms_cpu_time(results, output_path):
    """Compare CPU time for NN, NN-2Opt, RRNN across problem sizes."""
    _setup_plot()
    sizes = sorted({n for algo in ['NN', 'NN-2Opt', 'RRNN'] for n in results.get(algo, {})})
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        times = [_safe_median(results[algo][n]['cpu_times']) if n in results[algo] and results[algo][n]['cpu_times'] else np.nan for n in sizes]
        plt.plot(sizes, times, label=algo, color=_algo_colors[algo], marker=_algo_markers[algo])
    plt.xlabel('Number of Cities')
    plt.ylabel('Median CPU Time (seconds)')
    plt.title('CPU Time Comparison: Nearest Neighbor Algorithms')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_nn_algorithms_cost(results, output_path):
    """Compare solution cost for NN, NN-2Opt, RRNN across problem sizes."""
    _setup_plot()
    sizes = sorted({n for algo in ['NN', 'NN-2Opt', 'RRNN'] for n in results.get(algo, {})})
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        costs = [_safe_median(results[algo][n]['costs']) if n in results[algo] and results[algo][n]['costs'] else np.nan for n in sizes]
        plt.plot(sizes, costs, label=algo, color=_algo_colors[algo], marker=_algo_markers[algo])
    plt.xlabel('Number of Cities')
    plt.ylabel('Median Solution Cost')
    plt.title('Solution Quality Comparison: Nearest Neighbor Algorithms')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# PART II: A* ALGORITHM (4 plots)
def _get_astar_sizes(results):
    return sorted(results.get('A*', {}).keys())

def plot_relative_wall_time_with_nodes(results, output_path):
    """Plot wall time as multiple of A* wall time with A* nodes expanded on second Y-axis."""
    sizes = _get_astar_sizes(results)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Map algorithm names to match the example plot style
    algo_labels = {'NN': 'NN', 'NN-2Opt': 'NN2O', 'RRNN': 'RNN'}
    
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        multiples = []
        for n in sizes:
            try:
                algo_time = _safe_median(results[algo][n]['times'])
                astar_time = _safe_median(results['A*'][n]['times'])
                # Calculate multiple: Algorithm Time / A* Time
                if astar_time > 0:
                    multiples.append(algo_time / astar_time)
                else:
                    multiples.append(np.nan)
            except Exception:
                multiples.append(np.nan)
        
        # Plot with error bars
        ax1.errorbar(sizes, multiples, yerr=[0.01] * len(sizes), 
                    label=algo_labels[algo], color=_algo_colors[algo], 
                    marker=_algo_markers[algo], capsize=5, capthick=2)
    
    ax1.set_xlabel('Graph Size')
    ax1.set_ylabel('Wall Time (multiple of A* Wall Time)')
    ax1.set_title('Wall Time Comparison: Multiple of A* Wall Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Add second Y-axis for A* nodes expanded
    ax2 = ax1.twinx()
    nodes = [_safe_median(results['A*'][n]['nodes_expanded']) for n in sizes]
    ax2.plot(sizes, nodes, 'r*--', label='A* Nodes Expanded', alpha=0.7)
    ax2.set_ylabel('A* Nodes Expanded')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_relative_cpu_time_with_nodes(results, output_path):
    """Plot CPU time as multiple of A* CPU time with A* nodes expanded on second Y-axis."""
    sizes = _get_astar_sizes(results)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Map algorithm names to match the example plot style
    algo_labels = {'NN': 'NN', 'NN-2Opt': 'NN2O', 'RRNN': 'RNN'}
    
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        multiples = []
        for n in sizes:
            try:
                algo_cpu = _safe_median(results[algo][n]['cpu_times'])
                astar_cpu = _safe_median(results['A*'][n]['cpu_times'])
                # Calculate multiple: Algorithm CPU Time / A* CPU Time
                if astar_cpu > 0:
                    multiples.append(algo_cpu / astar_cpu)
                else:
                    multiples.append(np.nan)
            except Exception:
                multiples.append(np.nan)
        
        # Plot with error bars
        ax1.errorbar(sizes, multiples, yerr=[0.01] * len(sizes), 
                    label=algo_labels[algo], color=_algo_colors[algo], 
                    marker=_algo_markers[algo], capsize=5, capthick=2)
    
    ax1.set_xlabel('Graph Size')
    ax1.set_ylabel('CPU Time (multiple of A* CPU Time)')
    ax1.set_title('CPU Time Comparison: Multiple of A* CPU Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Add second Y-axis for A* nodes expanded
    ax2 = ax1.twinx()
    nodes = [_safe_median(results['A*'][n]['nodes_expanded']) for n in sizes]
    ax2.plot(sizes, nodes, 'r*--', label='A* Nodes Expanded', alpha=0.7)
    ax2.set_ylabel('A* Nodes Expanded')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_relative_cost_with_nodes(results, output_path):
    """Plot solution cost as multiple of optimal (A* cost)."""
    sizes = _get_astar_sizes(results)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Map algorithm names to match the example plot
    algo_labels = {'NN': 'NN', 'NN-2Opt': 'NN2O', 'RRNN': 'RNN'}
    
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        multiples = []
        for n in sizes:
            try:
                algo_cost = _safe_median(results[algo][n]['costs'])
                astar_cost = _safe_median(results['A*'][n]['costs'])
                # Calculate multiple: Algorithm Cost / A* Cost (multiple of optimal)
                if astar_cost > 0:
                    multiples.append(algo_cost / astar_cost)
                else:
                    multiples.append(np.nan)
            except Exception:
                multiples.append(np.nan)
        
        # Plot with error bars
        ax1.errorbar(sizes, multiples, yerr=[0.01] * len(sizes), 
                    label=algo_labels[algo], color=_algo_colors[algo], 
                    marker=_algo_markers[algo], capsize=5, capthick=2)
    
    ax1.set_xlabel('Graph Size')
    ax1.set_ylabel('Solution Cost (multiple of optimal)')
    ax1.set_title('Solution Quality Comparison: Multiple of Optimal (A*) Cost')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0.8)  # Start Y-axis at 0.8 to show performance relative to optimal
    ax1.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_astar_nodes_expanded(results, output_path):
    """Plot nodes expanded by A* vs problem size."""
    _setup_plot()
    sizes = _get_astar_sizes(results)
    nodes = [_safe_median(results['A*'][n]['nodes_expanded']) for n in sizes]
    plt.plot(sizes, nodes, 'r*-', label='A* Nodes Expanded')
    plt.xlabel('Number of Cities')
    plt.ylabel('Nodes Expanded')
    plt.title('A* Search Space Growth (Nodes Expanded)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# PART III: LOCAL SEARCH ALGORITHMS (9 plots)
# PART III: LOCAL SEARCH ALGORITHMS (3 required plots + 6 unused analysis plots)

# UNUSED HYPERPARAMETER TUNING AND CONVERGENCE PLOTS (commented out - available for detailed analysis)
# def plot_hc_hyperparameter(num_restarts_values, median_costs, output_path):
#     """Plot Hill Climbing solution cost vs num_restarts parameter."""
#     _setup_plot()
#     plt.plot(num_restarts_values, median_costs, color=_algo_colors['Hill Climbing'], marker=_algo_markers['Hill Climbing'])
#     plt.xlabel('num_restarts Parameter')
#     plt.ylabel('Median Solution Cost')
#     plt.title('Hill Climbing: Effect of num_restarts on Solution Quality')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()

# def plot_sa_hyperparameter(param_name, param_values, median_costs, output_path):
#     """Plot Simulated Annealing solution cost vs hyperparameter."""
#     _setup_plot()
#     plt.plot(param_values, median_costs, color=_algo_colors['Simulated Annealing'], marker=_algo_markers['Simulated Annealing'])
#     plt.xlabel(f'{param_name} Parameter')
#     plt.ylabel('Median Solution Cost')
#     plt.title(f'Simulated Annealing: Effect of {param_name} on Solution Quality')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()

# def plot_ga_hyperparameter(param_name, param_values, median_costs, output_path):
#     """Plot Genetic Algorithm solution cost vs hyperparameter."""
#     _setup_plot()
#     plt.plot(param_values, median_costs, color=_algo_colors['Genetic'], marker=_algo_markers['Genetic'])
#     plt.xlabel(f'{param_name} Parameter')
#     plt.ylabel('Median Solution Cost')
#     plt.title(f'Genetic Algorithm: Effect of {param_name} on Solution Quality')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()

# def plot_hc_convergence(iterations, best_costs_over_time, output_path):
#     """Plot Hill Climbing convergence over iterations."""
#     _setup_plot()
#     plt.plot(iterations, best_costs_over_time, color=_algo_colors['Hill Climbing'])
#     plt.xlabel('Iterations')
#     plt.ylabel('Best Solution Cost')
#     plt.title('Hill Climbing: Convergence Over Time')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()

# def plot_sa_convergence(iterations, best_costs_over_time, output_path):
#     """Plot Simulated Annealing convergence over iterations."""
#     _setup_plot()
#     plt.plot(iterations, best_costs_over_time, color=_algo_colors['Simulated Annealing'])
#     plt.xlabel('Iterations')
#     plt.ylabel('Best Solution Cost')
#     plt.title('Simulated Annealing: Convergence Over Time')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()

# def plot_ga_convergence(generations, best_costs_over_time, output_path):
#     """Plot Genetic Algorithm convergence over generations."""
#     _setup_plot()
#     plt.plot(generations, best_costs_over_time, color=_algo_colors['Genetic'])
#     plt.xlabel('Generations')
#     plt.ylabel('Best Solution Cost')
#     plt.title('Genetic Algorithm: Convergence Over Time')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()

def plot_sa_hyperparameter(param_name, param_values, median_costs, output_path):
    """Plot Simulated Annealing solution cost vs hyperparameter."""
    _setup_plot()
    plt.plot(param_values, median_costs, color=_algo_colors['Simulated Annealing'], marker=_algo_markers['Simulated Annealing'])
    plt.xlabel(f'{param_name} Parameter')
    plt.ylabel('Median Solution Cost')
    plt.title(f'Simulated Annealing: Effect of {param_name} on Solution Quality')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_ga_hyperparameter(param_name, param_values, median_costs, output_path):
    """Plot Genetic Algorithm solution cost vs hyperparameter."""
    _setup_plot()
    plt.plot(param_values, median_costs, color=_algo_colors['Genetic'], marker=_algo_markers['Genetic'])
    plt.xlabel(f'{param_name} Parameter')
    plt.ylabel('Median Solution Cost')
    plt.title(f'Genetic Algorithm: Effect of {param_name} on Solution Quality')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_hc_convergence(iterations, best_costs_over_time, output_path):
    """Plot Hill Climbing solution convergence over iterations."""
    _setup_plot()
    plt.plot(iterations, best_costs_over_time, color=_algo_colors['Hill Climbing'], marker=_algo_markers['Hill Climbing'])
    plt.xlabel('Iteration Number')
    plt.ylabel('Best Cost So Far (Median)')
    plt.title('Hill Climbing: Solution Convergence Over Iterations')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_sa_convergence(iterations, best_costs_over_time, output_path):
    """Plot Simulated Annealing solution convergence over iterations."""
    _setup_plot()
    plt.plot(iterations, best_costs_over_time, color=_algo_colors['Simulated Annealing'], marker=_algo_markers['Simulated Annealing'])
    plt.xlabel('Iteration Number')
    plt.ylabel('Best Cost So Far (Median)')
    plt.title('Simulated Annealing: Solution Convergence Over Iterations')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_ga_convergence(generations, best_costs_over_time, output_path):
    """Plot Genetic Algorithm solution convergence over generations."""
    _setup_plot()
    plt.plot(generations, best_costs_over_time, color=_algo_colors['Genetic'], marker=_algo_markers['Genetic'])
    plt.xlabel('Generation Number')
    plt.ylabel('Best Cost So Far (Median)')
    plt.title('Genetic Algorithm: Solution Convergence Over Generations')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_local_search_relative_wall_time(results, output_path):
    """Plot wall time relative to A* for local search algorithms (uses pre-filtered data for sizes 5-10)."""
    # Use all sizes in the filtered results (should be 5-10 where A* ran)
    sizes = sorted({n for algo in ['Hill Climbing', 'Simulated Annealing', 'Genetic'] 
                   for n in results.get(algo, {}).keys()})
    
    _setup_plot(figsize=(12, 6))
    
    # Algorithm labels for cleaner display
    algo_names = {
        'Hill Climbing': 'HC', 
        'Simulated Annealing': 'SA', 
        'Genetic': 'GA'
    }
    
    for algo in ['Hill Climbing', 'Simulated Annealing', 'Genetic']:
        rel = []
        for n in sizes:
            try:
                algo_time = _safe_median(results[algo][n]['times'])
                astar_time = _safe_median(results['A*'][n]['times'])
                if astar_time > 0:
                    rel.append(algo_time / astar_time)
                else:
                    rel.append(np.nan)
            except Exception:
                rel.append(np.nan)
        
        plt.plot(sizes, rel, label=f'{algo_names[algo]} / A*', 
                color=_algo_colors[algo], marker=_algo_markers[algo], linewidth=2, markersize=8)
    
    plt.xlabel('Number of Cities')
    plt.ylabel('Wall Time Relative to A*')
    plt.title('Wall Time Relative to A*: Local Search Algorithms')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_local_search_relative_cpu_time(results, output_path):
    """Plot CPU time relative to A* for local search algorithms (uses pre-filtered data for sizes 5-10)."""
    # Use all sizes in the filtered results (should be 5-10 where A* ran)
    sizes = sorted({n for algo in ['Hill Climbing', 'Simulated Annealing', 'Genetic'] 
                   for n in results.get(algo, {}).keys()})
    
    _setup_plot(figsize=(12, 6))
    
    # Algorithm labels for cleaner display
    algo_names = {
        'Hill Climbing': 'HC', 
        'Simulated Annealing': 'SA', 
        'Genetic': 'GA'
    }
    
    for algo in ['Hill Climbing', 'Simulated Annealing', 'Genetic']:
        rel = []
        for n in sizes:
            try:
                algo_cpu = _safe_median(results[algo][n]['cpu_times'])
                astar_cpu = _safe_median(results['A*'][n]['cpu_times'])
                if astar_cpu > 0:
                    rel.append(algo_cpu / astar_cpu)
                else:
                    rel.append(np.nan)
            except Exception:
                rel.append(np.nan)
        
        plt.plot(sizes, rel, label=f'{algo_names[algo]} / A*', 
                color=_algo_colors[algo], marker=_algo_markers[algo], linewidth=2, markersize=8)
    
    plt.xlabel('Number of Cities')
    plt.ylabel('CPU Time Relative to A*')
    plt.title('CPU Time Relative to A*: Local Search Algorithms')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_local_search_relative_cost(results, output_path):
    """Plot solution cost relative to A* for local search algorithms (uses pre-filtered data for sizes 5-10)."""
    # Use all sizes in the filtered results (should be 5-10 where A* ran)
    sizes = sorted({n for algo in ['Hill Climbing', 'Simulated Annealing', 'Genetic'] 
                   for n in results.get(algo, {}).keys()})
    
    _setup_plot(figsize=(12, 6))
    
    # Algorithm labels for cleaner display
    algo_names = {
        'Hill Climbing': 'HC', 
        'Simulated Annealing': 'SA', 
        'Genetic': 'GA'
    }
    
    for algo in ['Hill Climbing', 'Simulated Annealing', 'Genetic']:
        rel = []
        for n in sizes:
            try:
                algo_cost = _safe_median(results[algo][n]['costs'])
                astar_cost = _safe_median(results['A*'][n]['costs'])
                if astar_cost > 0:
                    rel.append(algo_cost / astar_cost)
                else:
                    rel.append(np.nan)
            except Exception:
                rel.append(np.nan)
        
        plt.plot(sizes, rel, label=f'{algo_names[algo]} / A*', 
                color=_algo_colors[algo], marker=_algo_markers[algo], linewidth=2, markersize=8)
    
    plt.xlabel('Number of Cities')
    plt.ylabel('Solution Cost Relative to A*')
    plt.title('Solution Quality Relative to A*: Local Search Algorithms')
    plt.ylim(bottom=0.8)  # Start Y-axis at 0.8 to show performance relative to optimal
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

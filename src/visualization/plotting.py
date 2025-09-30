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

# PART I: NEAREST NEIGHBOR ALGORITHMS (5 plots)
def plot_rrnn_k_parameter(k_values, median_costs, output_path):
    """Plot RRNN solution cost vs k parameter."""
    _setup_plot()
    plt.plot(k_values, median_costs, color=_algo_colors['RRNN'], marker=_algo_markers['RRNN'])
    plt.xlabel('k Parameter')
    plt.ylabel('Median Solution Cost')
    plt.title('RRNN: Effect of k Parameter on Solution Quality')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_rrnn_repeats_parameter(num_repeats_values, median_costs, output_path):
    """Plot RRNN solution cost vs num_repeats parameter."""
    _setup_plot()
    plt.plot(num_repeats_values, median_costs, color=_algo_colors['RRNN'], marker=_algo_markers['RRNN'])
    plt.xlabel('num_repeats Parameter')
    plt.ylabel('Median Solution Cost')
    plt.title('RRNN: Effect of num_repeats on Solution Quality')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

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
    """Plot wall time relative to A* and nodes expanded."""
    sizes = _get_astar_sizes(results)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        rel = []
        for n in sizes:
            try:
                rel.append(_safe_median(results[algo][n]['times']) / _safe_median(results['A*'][n]['times']))
            except Exception:
                rel.append(np.nan)
        ax1.plot(sizes, rel, label=f'{algo} / A*', color=_algo_colors[algo], marker=_algo_markers[algo])
    ax1.set_xlabel('Number of Cities')
    ax1.set_ylabel('Wall Time Relative to A*')
    ax1.set_title('Wall Time Relative to A* (with Nodes Expanded)')
    ax1.grid(True)
    ax2 = ax1.twinx()
    nodes = [_safe_median(results['A*'][n]['nodes_expanded']) for n in sizes]
    ax2.plot(sizes, nodes, 'r*-', label='A* Nodes Expanded')
    ax2.set_ylabel('Nodes Expanded (A*)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_relative_cpu_time_with_nodes(results, output_path):
    """Plot CPU time relative to A* and nodes expanded."""
    sizes = _get_astar_sizes(results)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        rel = []
        for n in sizes:
            try:
                rel.append(_safe_median(results[algo][n]['cpu_times']) / _safe_median(results['A*'][n]['cpu_times']))
            except Exception:
                rel.append(np.nan)
        ax1.plot(sizes, rel, label=f'{algo} / A*', color=_algo_colors[algo], marker=_algo_markers[algo])
    ax1.set_xlabel('Number of Cities')
    ax1.set_ylabel('CPU Time Relative to A*')
    ax1.set_title('CPU Time Relative to A* (with Nodes Expanded)')
    ax1.grid(True)
    ax2 = ax1.twinx()
    nodes = [_safe_median(results['A*'][n]['nodes_expanded']) for n in sizes]
    ax2.plot(sizes, nodes, 'r*-', label='A* Nodes Expanded')
    ax2.set_ylabel('Nodes Expanded (A*)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_relative_cost_with_nodes(results, output_path):
    """Plot solution cost relative to A* and nodes expanded."""
    sizes = _get_astar_sizes(results)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        rel = []
        for n in sizes:
            try:
                rel.append(_safe_median(results[algo][n]['costs']) / _safe_median(results['A*'][n]['costs']))
            except Exception:
                rel.append(np.nan)
        ax1.plot(sizes, rel, label=f'{algo} / A*', color=_algo_colors[algo], marker=_algo_markers[algo])
    ax1.set_xlabel('Number of Cities')
    ax1.set_ylabel('Solution Cost Relative to A*')
    ax1.set_title('Solution Quality Relative to A* (with Nodes Expanded)')
    ax1.grid(True)
    ax2 = ax1.twinx()
    nodes = [_safe_median(results['A*'][n]['nodes_expanded']) for n in sizes]
    ax2.plot(sizes, nodes, 'r*-', label='A* Nodes Expanded')
    ax2.set_ylabel('Nodes Expanded (A*)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
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
def plot_hc_hyperparameter(num_restarts_values, median_costs, output_path):
    """Plot Hill Climbing solution cost vs num_restarts parameter."""
    _setup_plot()
    plt.plot(num_restarts_values, median_costs, color=_algo_colors['Hill Climbing'], marker=_algo_markers['Hill Climbing'])
    plt.xlabel('num_restarts Parameter')
    plt.ylabel('Median Solution Cost')
    plt.title('Hill Climbing: Effect of num_restarts on Solution Quality')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

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
    """Plot wall time relative to A* for local search algorithms."""
    sizes = _get_astar_sizes(results)
    _setup_plot(figsize=(12, 6))
    for algo in ['Hill Climbing', 'Simulated Annealing', 'Genetic']:
        rel = []
        for n in sizes:
            try:
                rel.append(_safe_median(results[algo][n]['times']) / _safe_median(results['A*'][n]['times']))
            except Exception:
                rel.append(np.nan)
        plt.plot(sizes, rel, label=f'{algo} / A*', color=_algo_colors[algo], marker=_algo_markers[algo])
    plt.xlabel('Number of Cities')
    plt.ylabel('Wall Time Relative to A*')
    plt.title('Wall Time Relative to A*: Local Search Algorithms')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_local_search_relative_cpu_time(results, output_path):
    """Plot CPU time relative to A* for local search algorithms."""
    sizes = _get_astar_sizes(results)
    _setup_plot(figsize=(12, 6))
    for algo in ['Hill Climbing', 'Simulated Annealing', 'Genetic']:
        rel = []
        for n in sizes:
            try:
                rel.append(_safe_median(results[algo][n]['cpu_times']) / _safe_median(results['A*'][n]['cpu_times']))
            except Exception:
                rel.append(np.nan)
        plt.plot(sizes, rel, label=f'{algo} / A*', color=_algo_colors[algo], marker=_algo_markers[algo])
    plt.xlabel('Number of Cities')
    plt.ylabel('CPU Time Relative to A*')
    plt.title('CPU Time Relative to A*: Local Search Algorithms')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_local_search_relative_cost(results, output_path):
    """Plot solution cost relative to A* for local search algorithms."""
    sizes = _get_astar_sizes(results)
    _setup_plot(figsize=(12, 6))
    for algo in ['Hill Climbing', 'Simulated Annealing', 'Genetic']:
        rel = []
        for n in sizes:
            num = _safe_median(results[algo][n]['costs']) if n in results[algo] and results[algo][n]['costs'] else np.nan
            denom = _safe_median(results['A*'][n]['costs']) if n in results['A*'] and results['A*'][n]['costs'] else np.nan
            rel.append(num / denom if denom and not np.isnan(denom) else np.nan)
        plt.plot(sizes, rel, label=f'{algo} / A*', color=_algo_colors[algo], marker=_algo_markers[algo])
    plt.xlabel('Number of Cities')
    plt.ylabel('Solution Cost Relative to A*')
    plt.title('Solution Quality Relative to A*: Local Search Algorithms')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

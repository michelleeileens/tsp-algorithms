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

# Helper for safe min/max calculation
def _safe_min(data):
    arr = np.array(data)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return np.min(arr)

def _safe_max(data):
    arr = np.array(data)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return np.max(arr)

# Helper for plotting with min/max error bars
def _plot_with_minmax(ax, sizes, algo, results, metric_key, label=None, x_offset=0):
    """Plot algorithm data with error bars showing min/max range."""
    medians = []
    mins = []
    maxs = []
    
    for n in sizes:
        if n in results[algo] and results[algo][n][metric_key]:
            data = results[algo][n][metric_key]
            medians.append(_safe_median(data))
            mins.append(_safe_min(data))
            maxs.append(_safe_max(data))
        else:
            medians.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
    
    medians = np.array(medians)
    mins = np.array(mins)
    maxs = np.array(maxs)
    
    # Calculate error bars (distance from median to min/max)
    err_lower = medians - mins
    err_upper = maxs - medians
    
    # Apply x-axis offset to prevent overlapping
    x_positions = np.array(sizes) + x_offset
    
    display_label = label if label else algo
    ax.errorbar(x_positions, medians, yerr=[err_lower, err_upper], 
                label=display_label, color=_algo_colors[algo], 
                marker=_algo_markers[algo], capsize=5, capthick=2, linewidth=2)

# Helper for plotting ratios relative to A* with proper error bars
def _plot_ratio_with_minmax(ax, sizes, algo, results, metric_key, baseline_algo='A*', label=None, x_offset=0):
    """Plot ratio (algo/baseline) with error bars showing min/max range of ratios."""
    ratios_medians = []
    ratios_mins = []
    ratios_maxs = []
    
    for n in sizes:
        if (n in results[algo] and results[algo][n][metric_key] and 
            n in results[baseline_algo] and results[baseline_algo][n][metric_key]):
            
            algo_data = np.array(results[algo][n][metric_key])
            baseline_data = np.array(results[baseline_algo][n][metric_key])
            
            # Calculate all ratios for this size
            ratios = []
            for i in range(min(len(algo_data), len(baseline_data))):
                if baseline_data[i] > 0:
                    ratios.append(algo_data[i] / baseline_data[i])
            
            if ratios:
                ratios_medians.append(np.median(ratios))
                ratios_mins.append(np.min(ratios))
                ratios_maxs.append(np.max(ratios))
            else:
                ratios_medians.append(np.nan)
                ratios_mins.append(np.nan)
                ratios_maxs.append(np.nan)
        else:
            ratios_medians.append(np.nan)
            ratios_mins.append(np.nan)
            ratios_maxs.append(np.nan)
    
    ratios_medians = np.array(ratios_medians)
    ratios_mins = np.array(ratios_mins)
    ratios_maxs = np.array(ratios_maxs)
    
    # Calculate error bars (distance from median to min/max)
    err_lower = ratios_medians - ratios_mins
    err_upper = ratios_maxs - ratios_medians
    
    # Apply x-axis offset to prevent overlapping
    x_positions = np.array(sizes) + x_offset
    
    display_label = label if label else algo
    ax.errorbar(x_positions, ratios_medians, yerr=[err_lower, err_upper], 
                label=display_label, color=_algo_colors[algo], 
                marker=_algo_markers[algo], capsize=5, capthick=2, linewidth=2)

# PART I: NEAREST NEIGHBOR ALGORITHMS (3 required plots + 2 unused parameter plots)

# RRNN HYPERPARAMETER OPTIMIZATION PLOTS
def plot_rrnn_k_parameter(k_results, output_path):
    """Plot RRNN cost vs k parameter."""
    _setup_plot(figsize=(10, 6))
    
    k_values = sorted(k_results.keys())
    k_scores = [k_results[k] for k in k_values]
    
    plt.plot(k_values, k_scores, linewidth=2, markersize=8, 
             color=_algo_colors['RRNN'], marker=_algo_markers['RRNN'])
    plt.xlabel('k (Number of Nearest Neighbors)')
    plt.ylabel('Median Cost')
    plt.title('RRNN: Effect of k Parameter on Cost')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_rrnn_repeats_parameter(repeats_results, output_path):
    """Plot RRNN cost vs num_repeats parameter."""
    _setup_plot(figsize=(10, 6))
    
    num_repeats_values = sorted(repeats_results.keys())
    repeats_scores = [repeats_results[r] for r in num_repeats_values]
    
    plt.plot(num_repeats_values, repeats_scores, linewidth=2, markersize=8, 
             color=_algo_colors['RRNN'], marker=_algo_markers['RRNN'])
    plt.xlabel('Number of Repeats')
    plt.ylabel('Median Cost')
    plt.title('RRNN: Effect of num_repeats Parameter on Cost')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_rrnn_hyperparameter_combined(k_results, repeats_results, output_path):
    """Create combined plot showing both k and num_repeats optimization results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: k vs median score
    k_values = sorted(k_results.keys())
    k_scores = [k_results[k] for k in k_values]
    ax1.plot(k_values, k_scores, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('k (Number of Nearest Neighbors)', fontsize=12)
    ax1.set_ylabel('Median Score (Distance)', fontsize=12)
    ax1.set_title('RRNN: k Parameter vs Median Score', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Plot 2: num_repeats vs median score
    num_repeats_values = sorted(repeats_results.keys())
    repeats_scores = [repeats_results[r] for r in num_repeats_values]
    ax2.plot(num_repeats_values, repeats_scores, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Repeats', fontsize=12)
    ax2.set_ylabel('Median Score (Distance)', fontsize=12)
    ax2.set_title('RRNN: num_repeats Parameter vs Median Score', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def _find_optimal_repeats(repeats_results):
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

# UNUSED PARAMETER TUNING PLOTS (commented out - available for detailed analysis)


def plot_nn_algorithms_wall_time(results, output_path):
    """Compare wall time for NN, NN-2Opt, RRNN across problem sizes with min/max error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    
    sizes = sorted({n for algo in ['NN', 'NN-2Opt', 'RRNN'] for n in results.get(algo, {})})
    
    # Define x-axis offsets to prevent overlapping error bars
    offsets = {'NN': -0.2, 'NN-2Opt': 0.0, 'RRNN': 0.2}
    
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        _plot_with_minmax(ax, sizes, algo, results, 'times', x_offset=offsets[algo])
    
    ax.set_xlabel('Number of Cities')
    ax.set_ylabel('Wall Time (seconds, median with min/max range)')
    ax.set_title('Wall Time Comparison: Nearest Neighbor Algorithms')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_nn_algorithms_cpu_time(results, output_path):
    """Compare CPU time for NN, NN-2Opt, RRNN across problem sizes with min/max error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    
    sizes = sorted({n for algo in ['NN', 'NN-2Opt', 'RRNN'] for n in results.get(algo, {})})
    
    # Define x-axis offsets to prevent overlapping error bars
    offsets = {'NN': -0.2, 'NN-2Opt': 0.0, 'RRNN': 0.2}
    
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        _plot_with_minmax(ax, sizes, algo, results, 'cpu_times', x_offset=offsets[algo])
    
    ax.set_xlabel('Number of Cities')
    ax.set_ylabel('CPU Time (seconds, median with min/max range)')
    ax.set_title('CPU Time Comparison: Nearest Neighbor Algorithms')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_nn_algorithms_cost(results, output_path):
    """Compare cost for NN, NN-2Opt, RRNN across problem sizes with min/max error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    
    sizes = sorted({n for algo in ['NN', 'NN-2Opt', 'RRNN'] for n in results.get(algo, {})})
    
    # Define x-axis offsets to prevent overlapping error bars
    offsets = {'NN': -0.2, 'NN-2Opt': 0.0, 'RRNN': 0.2}
    
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        _plot_with_minmax(ax, sizes, algo, results, 'costs', x_offset=offsets[algo])
    
    ax.set_xlabel('Number of Cities')
    ax.set_ylabel('Cost (median with min/max range)')
    ax.set_title('Cost Comparison: Nearest Neighbor Algorithms')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# PART II: A* ALGORITHM (4 plots)
def _get_astar_sizes(results):
    return sorted(results.get('A*', {}).keys())

# PART II: A* RELATIVE COMPARISON PLOTS (3 plots for NN algorithms)
def plot_nn_wall_time_relative_to_astar(results, output_path):
    """Plot NN algorithms wall time divided by A* wall time."""
    sizes = _get_astar_sizes(results)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(True, alpha=0.3)
    
    # Map algorithm names and define x-axis offsets
    algo_labels = {'NN': 'NN', 'NN-2Opt': 'NN-2Opt', 'RRNN': 'RRNN'}
    offsets = {'NN': -0.15, 'NN-2Opt': 0.0, 'RRNN': 0.15}
    
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        _plot_ratio_with_minmax(ax, sizes, algo, results, 'times', 'A*', algo_labels[algo], x_offset=offsets[algo])
    
    ax.set_xlabel('Graph Size')
    ax.set_ylabel('Wall Time / A* Wall Time (median with min/max range)')
    ax.set_title('Wall Time Comparison: NN Algorithms Divided by A*')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_nn_cpu_time_relative_to_astar(results, output_path):
    """Plot NN algorithms CPU time divided by A* CPU time."""
    sizes = _get_astar_sizes(results)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(True, alpha=0.3)
    
    # Map algorithm names and define x-axis offsets
    algo_labels = {'NN': 'NN', 'NN-2Opt': 'NN-2Opt', 'RRNN': 'RRNN'}
    offsets = {'NN': -0.15, 'NN-2Opt': 0.0, 'RRNN': 0.15}
    
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        _plot_ratio_with_minmax(ax, sizes, algo, results, 'cpu_times', 'A*', algo_labels[algo], x_offset=offsets[algo])
    
    ax.set_xlabel('Graph Size')
    ax.set_ylabel('CPU Time / A* CPU Time (median with min/max range)')
    ax.set_title('CPU Time Comparison: NN Algorithms Divided by A*')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_nn_cost_relative_to_astar(results, output_path):
    """Plot NN algorithms cost divided by A* cost."""
    sizes = _get_astar_sizes(results)
    _setup_plot()
    
    # Map algorithm names and define x-axis offsets
    algo_labels = {'NN': 'NN', 'NN-2Opt': 'NN-2Opt', 'RRNN': 'RRNN'}
    offsets = {'NN': -0.15, 'NN-2Opt': 0.0, 'RRNN': 0.15}
    
    for algo in ['NN', 'NN-2Opt', 'RRNN']:
        _plot_ratio_with_minmax(plt.gca(), sizes, algo, results, 'costs', 'A*', algo_labels[algo], x_offset=offsets[algo])
    
    plt.xlabel('Graph Size')
    plt.ylabel('Cost / A* Cost (median with min/max range)')
    plt.title('Cost Comparison: NN Algorithms Divided by A*')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_astar_cost_comparison(results, output_path):
    """Plot A* cost comparison with improved error bars."""
    sizes = _get_astar_sizes(results)
    _setup_plot()
    
    # Extract A* cost data
    costs_medians = []
    costs_mins = []
    costs_maxs = []
    
    for n in sizes:
        if n in results['A*'] and results['A*'][n]['costs']:
            data = results['A*'][n]['costs']
            costs_medians.append(_safe_median(data))
            costs_mins.append(_safe_min(data))
            costs_maxs.append(_safe_max(data))
        else:
            costs_medians.append(np.nan)
            costs_mins.append(np.nan)
            costs_maxs.append(np.nan)
    
    costs_medians = np.array(costs_medians)
    costs_mins = np.array(costs_mins)
    costs_maxs = np.array(costs_maxs)
    err_lower = costs_medians - costs_mins
    err_upper = costs_maxs - costs_medians
    
    plt.errorbar(sizes, costs_medians, yerr=[err_lower, err_upper], 
                fmt='o-', label='A* Cost', capsize=5, linewidth=2, markersize=6)
    

    plt.xlabel('Graph Size')
    plt.ylabel('Cost (median with min/max range)')
    plt.title('A* Algorithm Cost vs Graph Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# PART III: A* RELATIVE COMPARISON PLOTS (3 plots for Local Search algorithms)
def plot_local_search_wall_time_relative_to_astar(results, output_path):
    """Plot local search algorithms wall time divided by A* wall time."""
    sizes = _get_astar_sizes(results)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(True, alpha=0.3)
    
    # Map algorithm names and define x-axis offsets
    algo_labels = {'Hill Climbing': 'HC', 'Simulated Annealing': 'SA', 'Genetic': 'GA'}
    offsets = {'Hill Climbing': -0.15, 'Simulated Annealing': 0.0, 'Genetic': 0.15}
    
    for algo in ['Hill Climbing', 'Simulated Annealing', 'Genetic']:
        _plot_ratio_with_minmax(ax, sizes, algo, results, 'times', 'A*', algo_labels[algo], x_offset=offsets[algo])
    
    ax.set_xlabel('Graph Size')
    ax.set_ylabel('Wall Time / A* Wall Time (median with min/max range)')
    ax.set_title('Wall Time Comparison: Local Search Algorithms Divided by A*')
    ax.legend(loc='upper left')
    
    # Format y-axis to 1 decimal place
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_local_search_cpu_time_relative_to_astar(results, output_path):
    """Plot local search algorithms CPU time divided by A* CPU time."""
    sizes = _get_astar_sizes(results)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(True, alpha=0.3)
    
    # Map algorithm names and define x-axis offsets
    algo_labels = {'Hill Climbing': 'HC', 'Simulated Annealing': 'SA', 'Genetic': 'GA'}
    offsets = {'Hill Climbing': -0.15, 'Simulated Annealing': 0.0, 'Genetic': 0.15}
    
    for algo in ['Hill Climbing', 'Simulated Annealing', 'Genetic']:
        _plot_ratio_with_minmax(ax, sizes, algo, results, 'cpu_times', 'A*', algo_labels[algo], x_offset=offsets[algo])
    
    ax.set_xlabel('Graph Size')
    ax.set_ylabel('CPU Time / A* CPU Time (median with min/max range)')
    ax.set_title('CPU Time Comparison: Local Search Algorithms Divided by A*')
    ax.legend(loc='upper left')

    # Format y-axis to 1 decimal place and start from 0.5
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.set_ylim(bottom=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_local_search_cost_relative_to_astar(results, output_path):
    """Plot local search algorithms cost divided by A* cost."""
    sizes = _get_astar_sizes(results)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(True, alpha=0.3)
    
    # Map algorithm names and define x-axis offsets
    algo_labels = {'Hill Climbing': 'HC', 'Simulated Annealing': 'SA', 'Genetic': 'GA'}
    offsets = {'Hill Climbing': -0.15, 'Simulated Annealing': 0.0, 'Genetic': 0.15}
    
    for algo in ['Hill Climbing', 'Simulated Annealing', 'Genetic']:
        _plot_ratio_with_minmax(ax, sizes, algo, results, 'costs', 'A*', algo_labels[algo], x_offset=offsets[algo])
    
    ax.set_xlabel('Graph Size')
    ax.set_ylabel('Cost / A* Cost (median with min/max range)')
    ax.set_title('Cost Comparison: Local Search Algorithms Divided by A*')
    ax.legend(loc='upper left')
    
    # Format y-axis to 1 decimal place and start from 0.5
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.set_ylim(bottom=1.0)
    
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


# incorrect plot, refer to plot_astar_cost_comparison for correct A* cost plot
def plot_relative_cost_with_nodes(results, output_path):
    """Plot cost as multiple of optimal (A* cost)."""
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
    ax1.set_ylabel('Cost (multiple of optimal)')
    ax1.set_title('Cost Comparison: Multiple of Optimal (A*) Cost')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0.8)  # Start Y-axis at 0.8 to show performance relative to optimal
    ax1.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_astar_nodes_expanded(results, output_path):
    """Plot nodes expanded by A* vs problem size with error bars."""
    _setup_plot()
    sizes = _get_astar_sizes(results)
    
    # Extract nodes expanded data with min/max
    nodes_medians = []
    nodes_mins = []
    nodes_maxs = []
    
    for n in sizes:
        if n in results['A*'] and results['A*'][n]['nodes_expanded']:
            data = results['A*'][n]['nodes_expanded']
            nodes_medians.append(_safe_median(data))
            nodes_mins.append(_safe_min(data))
            nodes_maxs.append(_safe_max(data))
        else:
            nodes_medians.append(np.nan)
            nodes_mins.append(np.nan)
            nodes_maxs.append(np.nan)
    
    nodes_medians = np.array(nodes_medians)
    nodes_mins = np.array(nodes_mins)
    nodes_maxs = np.array(nodes_maxs)
    err_lower = nodes_medians - nodes_mins
    err_upper = nodes_maxs - nodes_medians
    
    plt.errorbar(sizes, nodes_medians, yerr=[err_lower, err_upper], 
                fmt='*-', label='A* Nodes Expanded', capsize=5, linewidth=2, markersize=8, color='red')
    plt.xlabel('Graph Size')
    plt.ylabel('Nodes Expanded (median with min/max range)')
    plt.title('A* Search Space Growth (Nodes Expanded)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# PART III: LOCAL SEARCH ALGORITHMS (9 plots)
# PART III: LOCAL SEARCH ALGORITHMS (3 required plots + 6 unused analysis plots)

# UNUSED HYPERPARAMETER TUNING AND CONVERGENCE PLOTS (commented out - available for detailed analysis)
def plot_hc_hyperparameter(num_restarts_values, median_costs, output_path):
    """Plot Hill Climbing cost vs num_restarts parameter."""
    _setup_plot()
    plt.plot(num_restarts_values, median_costs, color=_algo_colors['Hill Climbing'], marker=_algo_markers['Hill Climbing'])
    plt.xlabel('num_restarts Parameter')
    plt.ylabel('Median Cost')
    plt.title('Hill Climbing: Effect of num_restarts on Cost')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_sa_hyperparameter(param_name, param_values, median_costs, output_path):
    """Plot Simulated Annealing cost vs hyperparameter."""
    _setup_plot()
    plt.plot(param_values, median_costs, color=_algo_colors['Simulated Annealing'], marker=_algo_markers['Simulated Annealing'])
    plt.xlabel(f'{param_name} Parameter')
    plt.ylabel('Median Cost')
    plt.title(f'Simulated Annealing: Effect of {param_name} on Cost')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_ga_hyperparameter(param_name, param_values, median_costs, output_path):
    """Plot Genetic Algorithm cost vs hyperparameter."""
    _setup_plot()
    plt.plot(param_values, median_costs, color=_algo_colors['Genetic'], marker=_algo_markers['Genetic'])
    plt.xlabel(f'{param_name} Parameter')
    plt.ylabel('Median Cost')
    plt.title(f'Genetic Algorithm: Effect of {param_name} on Cost')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_sa_hyperparameter(param_name, param_values, median_costs, output_path):
    """Plot Simulated Annealing cost vs hyperparameter."""
    _setup_plot()
    plt.plot(param_values, median_costs, color=_algo_colors['Simulated Annealing'], marker=_algo_markers['Simulated Annealing'])
    plt.xlabel(f'{param_name} Parameter')
    plt.ylabel('Median Cost')
    plt.title(f'Simulated Annealing: Effect of {param_name} on Cost')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_ga_hyperparameter(param_name, param_values, median_costs, output_path):
    """Plot Genetic Algorithm cost vs hyperparameter."""
    _setup_plot()
    plt.plot(param_values, median_costs, color=_algo_colors['Genetic'], marker=_algo_markers['Genetic'])
    plt.xlabel(f'{param_name} Parameter')
    plt.ylabel('Median Cost')
    plt.title(f'Genetic Algorithm: Effect of {param_name} on Cost')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_hc_convergence(convergence_data_list, output_path):
    """Plot Hill Climbing convergence with candlestick-style error bars.
    
    Args:
        convergence_data_list: List of convergence histories from multiple runs
        output_path: Output file path
    """
    _setup_plot()
    
    # Find the maximum length across all runs
    max_length = max(len(conv) for conv in convergence_data_list)
    
    # Collect costs for each iteration across all runs
    iterations = list(range(1, min(26, max_length + 1)))  # Limit to 25 iterations
    costs_by_iteration = [[] for _ in iterations]
    
    for conv_history in convergence_data_list:
        for i, (iter_num, cost) in enumerate(conv_history[:25]):
            if i < len(iterations):
                costs_by_iteration[i].append(cost)
    
    # Calculate median, min, max for error bars
    medians = []
    mins = []
    maxs = []
    
    for costs in costs_by_iteration:
        if costs:
            medians.append(np.median(costs))
            mins.append(np.min(costs))
            maxs.append(np.max(costs))
        else:
            medians.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
    
    medians = np.array(medians)
    mins = np.array(mins)
    maxs = np.array(maxs)
    err_lower = medians - mins
    err_upper = maxs - medians
    
    plt.errorbar(iterations[:len(medians)], medians, yerr=[err_lower, err_upper], 
                fmt='-', color=_algo_colors['Hill Climbing'], 
                marker=_algo_markers['Hill Climbing'], capsize=5, linewidth=2, markersize=6)
    
    plt.xlabel('Iteration Number')
    plt.ylabel('Cost (median with min/max range)')
    plt.title('Hill Climbing: Solution Convergence Over Iterations')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_sa_convergence(convergence_data_list, output_path):
    """Plot Simulated Annealing convergence with candlestick-style error bars.
    
    Args:
        convergence_data_list: List of convergence histories from multiple runs
        output_path: Output file path
    """
    _setup_plot()
    
    # Find the maximum length across all runs
    max_length = max(len(conv) for conv in convergence_data_list)
    
    # Collect costs for each iteration across all runs
    iterations = list(range(1, min(26, max_length + 1)))  # Limit to 25 iterations
    costs_by_iteration = [[] for _ in iterations]
    
    for conv_history in convergence_data_list:
        for i, (iter_num, cost) in enumerate(conv_history[:25]):
            if i < len(iterations):
                costs_by_iteration[i].append(cost)
    
    # Calculate median, min, max for error bars
    medians = []
    mins = []
    maxs = []
    
    for costs in costs_by_iteration:
        if costs:
            medians.append(np.median(costs))
            mins.append(np.min(costs))
            maxs.append(np.max(costs))
        else:
            medians.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
    
    medians = np.array(medians)
    mins = np.array(mins)
    maxs = np.array(maxs)
    err_lower = medians - mins
    err_upper = maxs - medians
    
    plt.errorbar(iterations[:len(medians)], medians, yerr=[err_lower, err_upper], 
                fmt='-', color=_algo_colors['Simulated Annealing'], 
                marker=_algo_markers['Simulated Annealing'], capsize=5, linewidth=2, markersize=6)
    
    plt.xlabel('Iteration Number')
    plt.ylabel('Cost (median with min/max range)')
    plt.title('Simulated Annealing: Solution Convergence Over Iterations')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_ga_convergence(convergence_data_list, output_path):
    """Plot Genetic Algorithm convergence with candlestick-style error bars.
    
    Args:
        convergence_data_list: List of convergence histories from multiple runs
        output_path: Output file path
    """
    _setup_plot()
    
    # Find the maximum length across all runs
    max_length = max(len(conv) for conv in convergence_data_list)
    
    # Collect costs for each generation across all runs
    generations = list(range(1, min(26, max_length + 1)))  # Limit to 25 generations
    costs_by_generation = [[] for _ in generations]
    
    for conv_history in convergence_data_list:
        for i, (gen_num, cost) in enumerate(conv_history[:25]):
            if i < len(generations):
                costs_by_generation[i].append(cost)
    
    # Calculate median, min, max for error bars
    medians = []
    mins = []
    maxs = []
    
    for costs in costs_by_generation:
        if costs:
            medians.append(np.median(costs))
            mins.append(np.min(costs))
            maxs.append(np.max(costs))
        else:
            medians.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
    
    medians = np.array(medians)
    mins = np.array(mins)
    maxs = np.array(maxs)
    err_lower = medians - mins
    err_upper = maxs - medians
    
    plt.errorbar(generations[:len(medians)], medians, yerr=[err_lower, err_upper], 
                fmt='-', color=_algo_colors['Genetic'], 
                marker=_algo_markers['Genetic'], capsize=5, linewidth=2, markersize=6)
    
    plt.xlabel('Generation Number')
    plt.ylabel('Cost (median with min/max range)')
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
    """Plot cost relative to A* for local search algorithms (uses pre-filtered data for sizes 5-10)."""
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
    plt.title('Cost Relative to A*: Local Search Algorithms')
    plt.ylim(bottom=0.8)  # Start Y-axis at 0.8 to show performance relative to optimal
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

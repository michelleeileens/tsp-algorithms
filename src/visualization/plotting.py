import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import seaborn as sns

def plot_performance_metrics(metrics: Dict[str, Any], title: str = None):
    """Plot performance metrics for algorithms."""
    plt.figure(figsize=(12, 6))
    
    # Plot comparison bars
    algorithms = list(metrics['times'].keys())
    x = np.arange(len(algorithms))
    width = 0.35
    
    plt.bar(x - width/2, [metrics['times'][algo] for algo in algorithms], 
            width, label='CPU Time (s)')
    plt.bar(x + width/2, [metrics['costs'][algo] for algo in algorithms], 
            width, label='Path Cost')
    
    plt.xlabel('Algorithm')
    plt.ylabel('Value')
    plt.title(title or 'Algorithm Performance Comparison')
    plt.xticks(x, algorithms)
    plt.legend()
    
    if 'nodes_expanded' in metrics and metrics['nodes_expanded'].get('A*'):
        ax2 = plt.twinx()
        ax2.plot(x[algorithms.index('A*')], 
                metrics['nodes_expanded']['A*'], 
                'r*', label='A* Nodes Expanded')
        ax2.set_ylabel('Nodes Expanded')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return plt.gcf()

def plot_growth_comparison(sizes: List[int], 
                         results: Dict[str, Dict[int, Any]], 
                         metric: str = 'time'):
    """Plot how different algorithms scale with problem size."""
    plt.figure(figsize=(12, 6))
    
    for algo in results:
        values = [results[algo][size][metric] for size in sizes]
        plt.plot(sizes, values, 'o-', label=algo)
    
    plt.xlabel('Number of Cities')
    plt.ylabel({'time': 'CPU Time (s)', 
               'cost': 'Path Cost', 
               'nodes': 'Nodes Expanded'}[metric])
    plt.title(f'Algorithm Scaling: {metric.title()}')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_astar_analysis(sizes: List[int], 
                       nodes_expanded: List[int], 
                       times: List[float],
                       costs: List[float]):
    """Create detailed analysis plots for A* performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot nodes expanded vs problem size
    ax1.plot(sizes, nodes_expanded, 'bo-')
    ax1.set_xlabel('Number of Cities')
    ax1.set_ylabel('Nodes Expanded')
    ax1.set_title('A* Search Space Growth')
    ax1.grid(True)
    
    # Plot time vs nodes expanded
    ax2.scatter(nodes_expanded, times)
    ax2.set_xlabel('Nodes Expanded')
    ax2.set_ylabel('CPU Time (s)')
    ax2.set_title('Time vs. Search Space')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def save_plots(figure, filename: str):
    """Save plot to file."""
    figure.savefig(filename)
    plt.close(figure)
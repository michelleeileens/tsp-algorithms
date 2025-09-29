import numpy as np
from time import process_time_ns, time_ns
from typing import Tuple, List, Any
import os

def load_matrix(file_path: str) -> np.ndarray:
    """
    Load an adjacency matrix from a file.
    
    Args:
        file_path (str): Path to the matrix file
        
    Returns:
        np.ndarray: The loaded adjacency matrix
    """
    return np.loadtxt(file_path)

def time_tracked(func):
    """
    Decorator to track both wall clock and CPU time of a function.
    Also catches and reports any exceptions.
    """
    def wrapper(*args, **kwargs) -> Tuple[Any, float, float]:
        start_wall = time_ns()
        start_cpu = process_time_ns()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            raise
        cpu_time = (process_time_ns() - start_cpu) / 1e9  # Convert to seconds
        wall_time = (time_ns() - start_wall) / 1e9  # Convert to seconds
        return result, wall_time, cpu_time
    return wrapper

def calculate_path_cost(path: List[int], adj_matrix: np.ndarray) -> float:
    """
    Calculate the total cost of a path in the TSP.
    
    Args:
        path (List[int]): List of cities in order of visit
        adj_matrix (np.ndarray): Adjacency matrix with distances
        
    Returns:
        float: Total cost of the path
    """
    cost = 0.0
    for i in range(len(path) - 1):
        cost += adj_matrix[path[i]][path[i + 1]]
    # Add cost of returning to start
    if len(path) > 1:
        cost += adj_matrix[path[-1]][path[0]]
    return cost

def get_matrix_files(size: int = None) -> List[str]:
    """
    Get all matrix files for a specific size or all sizes if not specified.
    
    Args:
        size (int, optional): Size of matrices to find. Defaults to None.
        
    Returns:
        List[str]: List of matrix file paths
    """
    matrix_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mats_911')
    all_files = []
    
    for file in os.listdir(matrix_dir):
        if file.endswith('.txt') and not file.endswith('.txt:Zone.Identifier'):
            if size is None:
                all_files.append(os.path.join(matrix_dir, file))
            elif file.startswith(f'{size}_'):
                all_files.append(os.path.join(matrix_dir, file))
                
    return sorted(all_files)
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

def validate_tsp_solution(path: List[int], adj_matrix: np.ndarray, reported_cost: float = None) -> dict:
    """
    Comprehensive validation of a TSP solution.
    
    Args:
        path (List[int]): The tour path
        adj_matrix (np.ndarray): Adjacency matrix
        reported_cost (float, optional): Cost reported by algorithm
        
    Returns:
        dict: Validation results with detailed information
    """
    n = len(adj_matrix)
    validation = {
        'valid': True,
        'errors': [],
        'path_length_correct': False,
        'cities_complete': False, 
        'no_duplicates': False,
        'cost_matches': False,
        'calculated_cost': 0.0,
        'expected_path_length': n + 1
    }
    
    # Check 1: Path length should be n+1 (all cities + return to start)
    if len(path) == n + 1:
        validation['path_length_correct'] = True
    else:
        validation['valid'] = False
        validation['errors'].append(f"Path length {len(path)} != {n+1} (expected n+1)")
    
    # Check 2: Should start and end with same city
    if len(path) >= 2:
        if path[0] != path[-1]:
            validation['valid'] = False
            validation['errors'].append(f"Path doesn't return to start: {path[0]} != {path[-1]}")
    
    # Check 3: All cities 0 to n-1 should appear exactly once (excluding duplicate start/end)
    if len(path) >= 2:
        cities_in_path = set(path[:-1])  # Exclude the return-to-start city
        expected_cities = set(range(n))
        
        if cities_in_path == expected_cities:
            validation['cities_complete'] = True
        else:
            validation['valid'] = False
            missing = expected_cities - cities_in_path
            extra = cities_in_path - expected_cities
            if missing:
                validation['errors'].append(f"Missing cities: {sorted(missing)}")
            if extra:
                validation['errors'].append(f"Extra cities: {sorted(extra)}")
    
    # Check 4: No duplicate cities (except start/end)
    if len(path) >= 2:
        path_without_return = path[:-1]
        if len(path_without_return) == len(set(path_without_return)):
            validation['no_duplicates'] = True
        else:
            validation['valid'] = False
            validation['errors'].append("Duplicate cities found in path (excluding return)")
    
    # Check 5: Calculate actual cost and compare with reported cost
    if validation['path_length_correct']:
        calculated_cost = calculate_path_cost(path[:-1], adj_matrix)  # Remove duplicate end city
        validation['calculated_cost'] = calculated_cost
        
        if reported_cost is not None:
            if abs(calculated_cost - reported_cost) < 1e-10:
                validation['cost_matches'] = True
            else:
                validation['valid'] = False
                validation['errors'].append(f"Cost mismatch: calculated {calculated_cost:.6f} != reported {reported_cost:.6f}")
    
    return validation

def test_algorithm_on_ngon(algorithm_func, matrix_file: str, expected_cost: float = None) -> dict:
    """
    Test a TSP algorithm on an n-gon matrix and validate results.
    
    Args:
        algorithm_func: Function that takes adj_matrix and returns (path, cost)
        matrix_file: Path to n-gon matrix file
        expected_cost: Expected optimal cost (usually n for n-gon)
        
    Returns:
        dict: Test results with validation info
    """
    try:
        # Load matrix
        adj_matrix = load_matrix(matrix_file)
        n = len(adj_matrix)
        
        if expected_cost is None:
            expected_cost = float(n)  # For n-gon, optimal cost is n
        
        # Run algorithm  
        result = algorithm_func(adj_matrix)
        
        # Handle different return formats (some have timing info)
        if isinstance(result, tuple) and len(result) == 3:
            # Format: (solution, wall_time, cpu_time) from @time_tracked
            solution, wall_time, cpu_time = result
            if isinstance(solution, tuple) and len(solution) == 3:
                # A* format: (path, cost, nodes)
                path, cost, nodes = solution
            elif isinstance(solution, tuple) and len(solution) == 2:
                # Standard format: (path, cost)
                path, cost = solution
            else:
                # Fallback
                path = solution if isinstance(solution, list) else []
                cost = calculate_path_cost(path[:-1] if len(path) > 0 else [], adj_matrix)
        elif isinstance(result, tuple) and len(result) == 2:
            # Format: (path, cost)
            path, cost = result
        else:
            # Format: path only or unknown
            path = result if isinstance(result, list) else []
            cost = calculate_path_cost(path[:-1] if len(path) > 0 else [], adj_matrix)
        
        # Add return to start if not present
        if len(path) > 0 and path[0] != path[-1]:
            path = path + [path[0]]
        
        # Validate solution
        validation = validate_tsp_solution(path, adj_matrix, cost)
        
        # Add test-specific info
        test_result = {
            'matrix_file': matrix_file,
            'matrix_size': n,
            'expected_cost': expected_cost,
            'algorithm_path': path,
            'algorithm_cost': cost,
            'cost_ratio': cost / expected_cost if expected_cost > 0 else float('inf'),
            'validation': validation,
            'passed': validation['valid'] and cost <= expected_cost * 1.01  # Allow 1% tolerance
        }
        
        return test_result
        
    except Exception as e:
        return {
            'matrix_file': matrix_file,
            'error': str(e),
            'passed': False
        }

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
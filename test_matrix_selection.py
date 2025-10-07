# #!/usr/bin/env python3

# """Test the updated TSP system with new matrix selection and plotting."""

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.utils import load_matrix

# def test_matrix_selection():
#     """Test the new matrix selection logic."""
#     matrix_dir = "mats_911"
#     target_sizes = [5, 6]  # Test with just 2 sizes
    
#     # Find all available files
#     all_files = [f for f in os.listdir(matrix_dir) if f.endswith('.txt')]
    
#     def get_matrix_info(filename):
#         try:
#             parts = filename.split('_')
#             size = int(parts[0])
#             if len(parts) >= 4 and parts[1] == 'random':
#                 version = int(parts[4].split('.')[0])
#                 return size, version
#         except (ValueError, IndexError):
#             pass
#         return None, None
    
#     # Group files by size
#     files_by_size = {}
#     for file in all_files:
#         size, version = get_matrix_info(file)
#         if size is not None and version is not None:
#             if size not in files_by_size:
#                 files_by_size[size] = []
#             files_by_size[size].append(file)
    
#     # Test new selection logic
#     selected_files = []
#     for size in target_sizes:
#         if size in files_by_size and files_by_size[size]:
#             size_files = []
#             for i in range(10):
#                 expected_filename = f"{size}_random_adj_mat_{i}.txt"
#                 if expected_filename in files_by_size[size]:
#                     size_files.append(expected_filename)
            
#             if len(size_files) == 10:
#                 selected_files.extend(size_files)
#                 print(f"Selected for size {size}: all {len(size_files)} matrices (0-9)")
                
#                 # Test loading one matrix to verify it works
#                 test_matrix = load_matrix(os.path.join(matrix_dir, size_files[0]))
#                 print(f"  Test matrix {size_files[0]} loaded successfully, size: {len(test_matrix)}")
#             else:
#                 print(f"Warning: Expected 10 matrices for size {size}, found {len(size_files)}")
    
#     print(f"Total selected files: {len(selected_files)}")
#     return len(selected_files) == 20  # Should be 10 files per size * 2 sizes

# if __name__ == "__main__":
#     print("Testing updated matrix selection...")
#     success = test_matrix_selection()
#     print(f"Test {'PASSED' if success else 'FAILED'}")
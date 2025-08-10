#!/usr/bin/env python3
"""
Script to execute all Python files in the runners copy directory
and categorize them based on their errors.
"""

import os
import subprocess
import sys
from collections import defaultdict
import json

def execute_file(filepath):
    """Execute a Python file and capture its output and error."""
    try:
        result = subprocess.run(
            [sys.executable, filepath],
            cwd=os.path.dirname(filepath),
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': 'Execution timed out (30s)',
            'success': False
        }
    except Exception as e:
        return {
            'returncode': -2,
            'stdout': '',
            'stderr': f'Execution failed: {str(e)}',
            'success': False
        }

def categorize_error(stderr, stdout):
    """Categorize the type of error based on stderr and stdout."""
    combined_output = (stderr + ' ' + stdout).lower()
    
    # BLAS library errors
    if any(keyword in combined_output for keyword in [
        'blas', 'lapack', 'openblas', 'mkl', 'atlas',
        'undefined symbol', 'libblas', 'liblapack'
    ]):
        return 'BLAS_LIBRARY_ERROR'
    
    # Import errors
    elif 'importerror' in combined_output or 'modulenotfounderror' in combined_output:
        return 'IMPORT_ERROR'
    
    # DaCe errors
    elif 'dace' in combined_output and ('error' in combined_output or 'exception' in combined_output):
        return 'DACE_ERROR'
    
    # NumPy/array errors
    elif any(keyword in combined_output for keyword in [
        'numpy', 'ndarray', 'array', 'dtype', 'broadcasting'
    ]) and 'error' in combined_output:
        return 'NUMPY_ARRAY_ERROR'
    
    # Syntax errors
    elif 'syntaxerror' in combined_output:
        return 'SYNTAX_ERROR'
    
    # Memory errors
    elif 'memoryerror' in combined_output or 'out of memory' in combined_output:
        return 'MEMORY_ERROR'
    
    # Timeout
    elif 'timed out' in combined_output:
        return 'TIMEOUT'
    
    # Other runtime errors
    elif any(keyword in combined_output for keyword in [
        'runtimeerror', 'valueerror', 'typeerror', 'indexerror',
        'keyerror', 'attributeerror'
    ]):
        return 'RUNTIME_ERROR'
    
    # Unknown error
    elif stderr or not stdout:
        return 'UNKNOWN_ERROR'
    
    # Success
    else:
        return 'SUCCESS'

def main():
    base_dir = "/home/chris/SPCL/npbench/runners copy"
    
    # Get all Python files
    python_files = []
    for file in os.listdir(base_dir):
        if file.endswith('.py') and file != 'test_all_runners.py':
            python_files.append(os.path.join(base_dir, file))
    
    python_files.sort()
    
    results = {}
    categories = defaultdict(list)
    
    print(f"Found {len(python_files)} Python files to execute\n")
    print("=" * 80)
    
    for i, filepath in enumerate(python_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{i:2d}/{len(python_files)}] Executing {filename}...", end=' ')
        
        result = execute_file(filepath)
        category = categorize_error(result['stderr'], result['stdout'])
        
        results[filename] = {
            'category': category,
            'success': result['success'],
            'returncode': result['returncode'],
            'stdout_length': len(result['stdout']),
            'stderr_length': len(result['stderr']),
            'stderr_preview': result['stderr'][:200] if result['stderr'] else '',
            'stdout_preview': result['stdout'][:200] if result['stdout'] else ''
        }
        
        categories[category].append(filename)
        
        status = "✓" if result['success'] else "✗"
        print(f"{status} [{category}]")
    
    print("\n" + "=" * 80)
    print("SUMMARY BY CATEGORY:")
    print("=" * 80)
    
    for category, files in sorted(categories.items()):
        print(f"\n{category} ({len(files)} files):")
        for filename in sorted(files):
            result = results[filename]
            print(f"  - {filename}")
            if result['stderr_preview']:
                print(f"    Error: {result['stderr_preview']}")
    
    print(f"\n" + "=" * 80)
    print("OVERALL STATISTICS:")
    print("=" * 80)
    
    total_files = len(python_files)
    successful = len(categories['SUCCESS'])
    failed = total_files - successful
    
    print(f"Total files:     {total_files}")
    print(f"Successful:      {successful} ({successful/total_files*100:.1f}%)")
    print(f"Failed:          {failed} ({failed/total_files*100:.1f}%)")
    
    # Save detailed results
    with open('execution_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: execution_results.json")

if __name__ == '__main__':
    main()

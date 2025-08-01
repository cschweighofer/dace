#!/usr/bin/env python3

import dace
import numpy as np
import copy
import matplotlib.pyplot as plt
from real_lowering import change_fptype
import math
# import dace.frontend.python.mpf_replacements

# Create symbolic constants
N = dace.symbol('N', positive=True)
M = dace.symbol('M', positive=True)
K = dace.symbol('K', positive=True)

# Simple test program that just adds 1 to an array
@dace.program
def simple_add_one(A: dace.float64[10]):
    A[:] = A[:] + 1.1

# Harmonic sum program that computes sum of 1/i for i from 1 to n (dynamic N)
@dace.program
def harmonic_sum(result: dace.float64[1], n: dace.int32):
    harmonic_sum_val = dace.float64(0.0)
    for i in range(1, n + 1):
        harmonic_sum_val += 1.0 / dace.float64(i)
    result[0] = harmonic_sum_val


# Harmonic sum program that performs matrix operations
@dace.program
def harmonic_sum_matrix(A: dace.float64[M, K]):
    for i in range(M):
        for j in range(K):
            A[i, j] += 1.0 / dace.float64(i + j + 1)


def test_change_fptype_simple():
    print("Testing change_fptype with simple add operation...")
    
    # Original array
    original_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0], dtype=np.float64)

    # Test original SDFG
    print("\nOriginal SDFG:")
    test_data = original_data.copy()
    sdfg = simple_add_one.to_sdfg()
    sdfg(A=test_data)
    print("Input:", original_data)
    print("Output:", test_data)
    print("Expected:", original_data + 1.1)

    print("\nAfter conversion to mpf:")
    sdfg_copy = copy.deepcopy(sdfg)
    
    change_fptype(sdfg_copy, dace.float64, dace.rational)
    
    # Test the modified SDFG
    test_data2 = original_data.copy()
    sdfg_copy(_A=test_data2)
    print("Input:", original_data)
    print("Output:", test_data2)
    print("Expected:", original_data + 1.1)
    print("Difference:", np.abs(test_data2 - (original_data + 1.1)))
    print("Max difference:", np.max(np.abs(test_data2 - (original_data + 1.1))))

def test_change_fptype_harmonic(n_val = 10):
    # Calculate expected harmonic sum for numbers 1 to n_val
    expected_harmonic = sum(1.0/i for i in range(1, n_val + 1))
    print(f"Expected harmonic sum (1 to {n_val}): {expected_harmonic}")
    
    # Test original SDFG
    result_data = np.array([0.0], dtype=np.float64)
    sdfg = harmonic_sum.to_sdfg()
    sdfg(result=result_data, n=n_val)
    print("Original Output:", result_data[0])
    original_diff = abs(result_data[0] - expected_harmonic)
    print("Difference:", original_diff)
    
    # Now test with precise conversion
    sdfg_copy = copy.deepcopy(sdfg)
    
    change_fptype(sdfg_copy, dace.float64, dace.mpf)
    
    # Test the modified SDFG
    result_data2 = np.array([0.0], dtype=np.float64)
    sdfg_copy(_result=result_data2, n=n_val)
    print("Precise Output:", result_data2[0])
    precise_diff = abs(result_data2[0] - expected_harmonic)
    print("Difference:", precise_diff)

    return precise_diff

def test_change_fptype_harmonic_matrix(m_val=10, k_val=10):
    """
    Tests the harmonic_sum_matrix SDFG with float64 and rational types.
    """
    print(f"\n--- Testing harmonic_sum_matrix for a {m_val}x{k_val} matrix ---")
    
    # Create input matrix
    original_matrix = np.random.rand(m_val, k_val).astype(np.float64)

    # Calculate expected result with high precision
    expected_matrix = original_matrix.copy()
    for i in range(m_val):
        for j in range(k_val):
            expected_matrix[i, j] += 1.0 / (i + j + 1.0)

    # Test original SDFG
    sdfg = harmonic_sum_matrix.to_sdfg()

    output_matrix_float = original_matrix.copy()
    sdfg(A=output_matrix_float, M=m_val, K=k_val)

    original_error = np.max(np.abs(output_matrix_float - expected_matrix))
    print(f"Original (float64) max error: {original_error}")

    # Test with rational conversion
    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.rational)

    output_matrix_rational = original_matrix.copy()
    sdfg_copy(_A=output_matrix_rational, M=m_val, K=k_val)
    
    rational_error = np.max(np.abs(output_matrix_rational - expected_matrix))
    print(f"Rational max error: {rational_error}")

    return original_error, rational_error


def plot_harmonic_sum_errors():
    """Plot the error differences for harmonic sum with varying N values"""
    print("\nGenerating harmonic sum error plot...")

    # Test with a range of N values
    # n_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    # n_values = [5, 10, 15, 20, 25, 30, 35, 40]
    # n_values = [5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40]
    # n_values = [5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40]
    # n_values = [22, 23, 24, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    n_values = [5, 10, 50, 80, 100, 200, 300, 500, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    original_errors = []
    mpf_errors = []
    original_values = []
    mpf_values = []

    # Prepare SDFGs once to avoid recompilation in the loop
    sdfg_original = harmonic_sum.to_sdfg()
    sdfg_mpf = copy.deepcopy(sdfg_original)
    change_fptype(sdfg_mpf, dace.float64, dace.mpq)

    for n_val in n_values:
        print(f"Testing N = {n_val}...")
        
        expected_harmonic = sum(1.0/i for i in range(1, n_val + 1))
        
        # Test original SDFG
        result_data_orig = np.array([0.0], dtype=np.float64)
        sdfg_original(result=result_data_orig, n=n_val)
        original_values.append(result_data_orig[0])
        original_error = abs(result_data_orig[0] - expected_harmonic)
        original_errors.append(original_error)
        
        # Test MPF conversion
        result_data_mpf = np.array([0.0], dtype=np.float64)
        try:
            sdfg_mpf(_result=result_data_mpf, n=n_val)
            mpf_values.append(result_data_mpf[0])
            mpf_error = abs(result_data_mpf[0] - expected_harmonic)
            print(f"N={n_val}: Original={result_data_orig[0]}, MPF={result_data_mpf[0]}, Error={mpf_error}")
        except Exception as e:
            print(f"Warning: MPF conversion failed for N={n_val}: {e}")
            mpf_values.append(np.nan)
            mpf_error = float('inf')  # Mark as failed
        
        mpf_errors.append(mpf_error)
    
    # Create the plot
    plt.figure(figsize=(15, 12))
    
    # Plot actual harmonic sum values
    plt.subplot(3, 1, 1)
    expected_values = [sum(1.0/i for i in range(1, n_val + 1)) for n_val in n_values]
    
    plt.plot(n_values, expected_values, 'k--', label='Expected (Analytical)', linewidth=2, alpha=0.8)
    plt.plot(n_values, original_values, 'b-o', label='Float64 (Original)', linewidth=2, markersize=4)
    plt.plot(n_values, mpf_values, 'r-s', label='MPF Arithmetic', linewidth=2, markersize=4)
    plt.xlabel('N (Harmonic Sum Upper Bound)')
    plt.ylabel('Harmonic Sum Value')
    plt.title('Actual Harmonic Sum Values: Float64 vs MPF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot errors on log scale
    plt.subplot(3, 1, 2)
    plt.semilogy(n_values, [max(1e-17, err) for err in original_errors], 'b-o', label='Float64 (Original)', linewidth=2, markersize=6)
    plt.semilogy(n_values, [max(1e-17, err) for err in mpf_errors], 'r-s', label='MPF Arithmetic', linewidth=2, markersize=6)
    plt.xlabel('N (Harmonic Sum Upper Bound)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Harmonic Sum Computation Error vs N')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot relative improvement/degradation
    plt.subplot(3, 1, 3)
    improvements = []
    for i, n_val in enumerate(n_values):
        if mpf_errors[i] != float('inf') and original_errors[i] > 0:
            improvement = (original_errors[i] - mpf_errors[i]) / original_errors[i] * 100
        elif mpf_errors[i] == float('inf'):
            improvement = -1000  # Large negative for failed cases
        else:
            improvement = 0
        improvements.append(improvement)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    plt.bar(n_values, improvements, color=colors, alpha=0.7)
    plt.xlabel('N (Harmonic Sum Upper Bound)')
    plt.ylabel('Relative Improvement (%)')
    plt.title('MPF vs Float64: Relative Error Improvement')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('harmonic_sum_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as 'harmonic_sum_error_analysis.png'")
    return n_values, original_errors, mpf_errors


if __name__ == "__main__":
    # test_change_fptype_simple()

    # Test a few specific values and show detailed output
    # test_values = [40, 50, 60, 70, 80, 90, 100]
    test_values = [5, 10, 15, 20, 25, 30, 35, 40]
    # test_va
    # test_values = [5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40]
    # test_values = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 200, 300, 500, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for n in test_values:
        print(f"\n--- Testing with N = {n} ---")
        test_change_fptype_harmonic(n)
    
    # Generate comprehensive error analysis plot
    plot_harmonic_sum_errors()

    # Test matrix version
    # test_change_fptype_harmonic_matrix(10, 10)

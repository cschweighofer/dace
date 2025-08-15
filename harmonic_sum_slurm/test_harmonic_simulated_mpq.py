#!/usr/bin/env python3

import dace
import numpy as np
import copy
import matplotlib.pyplot as plt
from real_lowering import change_fptype
import math

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
def harmonic_sum(result: dace.float64[1], n: dace.int64):
    harmonic_sum_val = dace.float64(0.0)
    for i in range(1, n + 1):
        harmonic_sum_val += 1.0 / dace.float64(i)
    result[0] = harmonic_sum_val


def test_change_fptype_harmonic(n_val = 10):
    print(f"Testing harmonic sum with N = {n_val}")
    
    # Test original SDFG
    result_data = np.array([0.0], dtype=np.float64)
    sdfg = harmonic_sum.to_sdfg()
    sdfg(result=result_data, n=n_val)
    print("Original Output:", result_data[0])
    
    # Now test with precise conversion
    sdfg_copy = copy.deepcopy(sdfg)
    
    change_fptype(sdfg_copy, dace.float64, dace.mpf)
    
    # Test the modified SDFG
    result_data2 = np.array([0.0], dtype=np.float64)
    sdfg_copy(_result=result_data2, n=n_val)
    print("Precise Output:", result_data2[0])
    
    # Calculate difference between the two SDFGs
    sdfg_difference = abs(result_data[0] - result_data2[0])
    print("SDFG Difference:", sdfg_difference)

    return sdfg_difference

def plot_harmonic_sum_errors():
    """Plot the error differences between original and MPF SDFG implementations"""
    print("\nGenerating harmonic sum error plot...")

    # Test with a range of N values
    # n_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    # n_values = [5, 10, 15, 20, 25, 30, 35, 40]
    # n_values = [5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40]
    # n_values = [5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40]
    # n_values = [22, 23, 24, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    # n_values = [5, 10, 50, 80, 100, 200, 300, 500, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    n_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208]
    sdfg_differences = []
    original_values = []
    mpf_values = []

    # Prepare SDFGs once to avoid recompilation in the loop
    sdfg_original = harmonic_sum.to_sdfg()
    sdfg_mpf = copy.deepcopy(sdfg_original)
    change_fptype(sdfg_mpf, dace.float64, dace.simulated_double)

    for n_val in n_values:
        print(f"Testing N = {n_val}...")
        
        # Test original SDFG
        result_data_orig = np.array([0.0], dtype=np.float64)
        sdfg_original(result=result_data_orig, n=n_val)
        original_values.append(result_data_orig[0])
        
        # Test MPF conversion
        result_data_mpf = np.array([0.0], dtype=np.float64)
        try:
            sdfg_mpf(_result=result_data_mpf, n=n_val)
            mpf_values.append(result_data_mpf[0])
            sdfg_difference = abs(result_data_orig[0] - result_data_mpf[0])
            print(f"N={n_val}: Original={result_data_orig[0]}, MPF={result_data_mpf[0]}, Difference={sdfg_difference}")
        except Exception as e:
            print(f"Warning: MPF conversion failed for N={n_val}: {e}")
            mpf_values.append(np.nan)
            sdfg_difference = float('inf')  # Mark as failed
        
        sdfg_differences.append(sdfg_difference)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot actual harmonic sum values with log x-axis
    plt.subplot(2, 1, 1)
    
    plt.semilogx(n_values, original_values, 'b-o', label='Float64 (Original)', linewidth=2, markersize=4)
    plt.semilogx(n_values, mpf_values, 'r-s', label='MPF Arithmetic', linewidth=2, markersize=4)
    plt.xlabel('N (Harmonic Sum Upper Bound)')
    plt.ylabel('Harmonic Sum Value')
    plt.title('Actual Harmonic Sum Values: Float64 vs MPF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot SDFG differences on log-log scale
    plt.subplot(2, 1, 2)
    plt.loglog(n_values, [max(1e-17, diff) for diff in sdfg_differences], 'g-^', label='|Float64 - MPF|', linewidth=2, markersize=6)
    plt.xlabel('N (Harmonic Sum Upper Bound)')
    plt.ylabel('Absolute Difference (log scale)')
    plt.title('Difference Between Float64 and MPF SDFG Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('harmonic_sum_error_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print(f"\nPlot saved as 'harmonic_sum_error_analysis.png'")
    return n_values, sdfg_differences


if __name__ == "__main__":
    # test_change_fptype_simple()

    # Test a few specific values and show detailed output
    # test_values = [40, 50, 60, 70, 80, 90, 100]
    test_values = [5, 10, 15, 20, 25, 30, 35, 40]
    # test_va
    # test_values = [5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40]
    # test_values = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 200, 300, 500, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # for n in test_values:
    #     print(f"\n--- Testing with N = {n} ---")
    #     test_change_fptype_harmonic(n)
    
    # Generate comprehensive error analysis plot
    plot_harmonic_sum_errors()

    # Test matrix version
    # test_change_fptype_harmonic_matrix(10, 10)

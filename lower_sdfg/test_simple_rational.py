#!/usr/bin/env python3

import dace
import numpy as np
import copy
from real_lowering import change_fptype

# Simple test program that just adds 1 to an array
@dace.program
def simple_add_one(A: dace.float64[10]):
    A[:] = A[:] + 1.1

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

    # Now test with rational conversion
    print("\nAfter conversion to rational:")
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

if __name__ == "__main__":
    test_change_fptype_simple()

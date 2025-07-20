import dace
import numpy as np
#import cupy as cp

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import copy

import numpy as np
from scipy import stats
from real_lowering import InputSpec, analyze_sdfg_precision, change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float64):
    A = np.fromfunction(lambda i, j, k: (i + j + (N - k)) * 10 / N, (N, N, N),
                        dtype=datatype)
    B = np.copy(A)

    return A, B


sdfg = dace.SDFG.from_file("../sdfgs/heat_3d_auto_opt_cpu.sdfg")

def run_S():
    TSTEPS, N = (25, 25)
    A, B = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, A=A, B=B)
    print("Result: ", A, B)

def run_M():
    TSTEPS, N = (50, 40)
    A, B = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, A=A, B=B)

def run_L():
    TSTEPS, N = (100, 70)
    A, B = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, A=A, B=B)

def run_paper():
    TSTEPS, N = (500, 120)
    A, B = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, A=A, B=B)

def compare_to_rational():
    TSTEPS, N = (25, 25)
    A, B = initialize(N=N)
    # Make a deep copy of the SDFG
    sdfg_copy = copy.deepcopy(sdfg)
    # Call change_fptype with both src and internal type as FP64
    change_fptype(sdfg_copy, dace.float64, dace.rational)
    # Run the SDFG as in run_S
    A2, B2 = initialize(N=N)
    sdfg_copy(TSTEPS=TSTEPS, N=N, _A=A2, _B=B2)
    print("Result after change_fptype (FP64->FP64):", A2, B2)
    # Optionally, compare to original run_S result
    A_ref, B_ref = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, A=A_ref, B=B_ref)
    print("Original result:", A_ref, B_ref)
    diff = max(np.max(np.abs(A2 - A_ref)), np.max(np.abs(B2 - B_ref)))
    print("Difference:", diff)
    return diff


if __name__ == "__main__":
    compare_to_rational()

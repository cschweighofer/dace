# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    t0, p0, t1, p1 = rng.random((N, )), rng.random((N, )), rng.random(
        (N, )), rng.random((N, ))
    return t0, p0, t1, p1


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/arc_distance_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (100000)
    M = (1000000)
    L = (10000000)
    PAPER = (10000000)

    def __init__(self, N):
        self.N = N

    def init(self):
        theta_1, phi_1, theta_2, phi_2 = initialize(N=self.N)
        return self.N, theta_1, phi_1, theta_2, phi_2

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, theta_1, phi_1, theta_2, phi_2 = size.init()

    theta_1_ref = np.copy(theta_1)
    phi_1_ref = np.copy(phi_1)
    theta_2_ref = np.copy(theta_2)
    phi_2_ref = np.copy(phi_2)
    
    # Allocate output buffer for the distance matrix
    result_ref = np.zeros(N, dtype=np.float64)
    sdfg(N=N, theta_1=theta_1_ref, phi_1=phi_1_ref, theta_2=theta_2_ref, phi_2=phi_2_ref, __return=result_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    theta_12 = np.copy(theta_1)
    phi_12 = np.copy(phi_1)
    theta_22 = np.copy(theta_2)
    phi_22 = np.copy(phi_2)
    
    # Allocate output buffer for the modified precision distance matrix
    result_mpf = np.zeros(N, dtype=np.float64)
    sdfg_copy(N=N, _theta_1=theta_12, _phi_1=phi_12, _theta_2=theta_22, _phi_2=phi_22, ___return=result_mpf)

    # Compare the results instead of the input arrays
    diff_result = np.sqrt(np.mean((result_mpf - result_ref)**2))
    max_diff = np.max(np.abs(result_mpf - result_ref))
    print("RMS Difference in results:", diff_result)
    print("Max Difference in results:", max_diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

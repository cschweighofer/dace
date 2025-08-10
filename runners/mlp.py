# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(C_in, N, S0, S1, S2):
    from numpy.random import default_rng
    rng = default_rng(42)

    mlp_sizes = [S0, S1, S2]  # [300, 100, 10]
    # Inputs
    input = np.random.rand(N, C_in).astype(np.float32)
    # Weights
    w1 = rng.random((C_in, mlp_sizes[0]), dtype=np.float32)
    b1 = rng.random((mlp_sizes[0], ), dtype=np.float32)
    w2 = rng.random((mlp_sizes[0], mlp_sizes[1]), dtype=np.float32)
    b2 = rng.random((mlp_sizes[1], ), dtype=np.float32)
    w3 = rng.random((mlp_sizes[1], mlp_sizes[2]), dtype=np.float32)
    b3 = rng.random((mlp_sizes[2], ), dtype=np.float32)

    return input, w1, b1, w2, b2, w3, b3


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/mlp_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (3, 8, 30000, 2000, 2000)
    M = (3, 8, 30000, 10000, 10000)
    L = (3, 8, 30000, 30000, 30000)
    PAPER = (3, 8, 30000, 10000, 1000)

    def __init__(self, C_in, N, S0, S1, S2):
        self.C_in = C_in
        self.N = N
        self.S0 = S0
        self.S1 = S1
        self.S2 = S2

    def init(self):
        input, w1, b1, w2, b2, w3, b3 = initialize(C_in=self.C_in, N=self.N, S0=self.S0, S1=self.S1, S2=self.S2)
        return self.C_in, self.N, self.S0, self.S1, self.S2, input, w1, b1, w2, b2, w3, b3

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    C_in, N, S0, S1, S2, input, w1, b1, w2, b2, w3, b3 = size.init()

    input_ref = np.copy(input)
    w1_ref = np.copy(w1)
    b1_ref = np.copy(b1)
    w2_ref = np.copy(w2)
    b2_ref = np.copy(b2)
    w3_ref = np.copy(w3)
    b3_ref = np.copy(b3)
    sdfg(C_in=C_in, N=N, S0=S0, S1=S1, S2=S2, input=input_ref, w1=w1_ref, b1=b1_ref, w2=w2_ref, b2=b2_ref, w3=w3_ref, b3=b3_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    input2 = np.copy(input)
    w12 = np.copy(w1)
    b12 = np.copy(b1)
    w22 = np.copy(w2)
    b22 = np.copy(b2)
    w32 = np.copy(w3)
    b32 = np.copy(b3)
    sdfg_copy(C_in=C_in, N=N, S0=S0, S1=S1, S2=S2, _input=input2, _w1=w12, _b1=b12, _w2=w22, _b2=b22, _w3=w32, _b3=b32)

    diff_input = np.sqrt(np.mean((input2 - input_ref)**2))
    diff_w1 = np.sqrt(np.mean((w12 - w1_ref)**2))
    diff_b1 = np.sqrt(np.mean((b12 - b1_ref)**2))
    diff_w2 = np.sqrt(np.mean((w22 - w2_ref)**2))
    diff_b2 = np.sqrt(np.mean((b22 - b2_ref)**2))
    diff_w3 = np.sqrt(np.mean((w32 - w3_ref)**2))
    diff_b3 = np.sqrt(np.mean((b32 - b3_ref)**2))
    diff = max(np.max(np.abs(input2 - input_ref)), np.max(np.abs(w12 - w1_ref)), np.max(np.abs(b12 - b1_ref)), np.max(np.abs(w22 - w2_ref)), np.max(np.abs(b22 - b2_ref)), np.max(np.abs(w32 - w3_ref)), np.max(np.abs(b32 - b3_ref)))
    print("RMS Difference INPUT:", diff_input)
    print("RMS Difference W1:", diff_w1)
    print("RMS Difference B1:", diff_b1)
    print("RMS Difference W2:", diff_w2)
    print("RMS Difference B2:", diff_b2)
    print("RMS Difference W3:", diff_w3)
    print("RMS Difference B3:", diff_b3)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (C_IN={model_size.C_in}, N={model_size.N}, S0={model_size.S0}, S1={model_size.S1}, S2={model_size.S2})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

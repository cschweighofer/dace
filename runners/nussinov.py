# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.int32):
    seq = np.fromfunction(lambda i: (i + 1) % 4, (N, ), dtype=datatype)

    return seq


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/nussinov_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (40)
    M = (90)
    L = (200)
    PAPER = (500)

    def __init__(self, N):
        self.N = N

    def init(self):
        seq = initialize(N=self.N)
        return self.N, seq

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, seq = size.init()

    seq_ref = np.copy(seq)
    sdfg(N=N, seq=seq_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    seq2 = np.copy(seq)
    sdfg_copy(N=N, _seq=seq2)

    diff_seq = np.sqrt(np.mean((seq2 - seq_ref)**2))
    diff = max(np.max(np.abs(seq2 - seq_ref)))
    print("RMS Difference SEQ:", diff_seq)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

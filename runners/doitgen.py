# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(NR, NQ, NP, datatype=np.float64):
    A = np.fromfunction(lambda i, j, k: ((i * j + k) % NP) / NP, (NR, NQ, NP),
                        dtype=datatype)
    C4 = np.fromfunction(lambda i, j: (i * j % NP) / NP, (NP, NP),
                         dtype=datatype)

    return A, C4


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/doitgen_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (60, 60, 128)
    M = (110, 125, 256)
    L = (220, 250, 512)
    PAPER = (220, 250, 270)

    def __init__(self, NR, NQ, NP):
        self.NR = NR
        self.NQ = NQ
        self.NP = NP

    def init(self):
        A, C4 = initialize(NR=self.NR, NQ=self.NQ, NP=self.NP)
        return self.NR, self.NQ, self.NP, A, C4

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    NR, NQ, NP, A, C4 = size.init()

    A_ref = np.copy(A)
    C4_ref = np.copy(C4)
    sdfg(NR=NR, NQ=NQ, NP=NP, A=A_ref, C4=C4_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    A2 = np.copy(A)
    C42 = np.copy(C4)
    sdfg_copy(NR=NR, NQ=NQ, NP=NP, _A=A2, _C4=C42)

    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff_C4 = np.sqrt(np.mean((C42 - C4_ref)**2))
    diff = max(np.max(np.abs(A2 - A_ref)), np.max(np.abs(C42 - C4_ref)))
    print("RMS Difference A:", diff_A)
    print("RMS Difference C4:", diff_C4)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (NR={model_size.NR}, NQ={model_size.NQ}, NP={model_size.NP})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

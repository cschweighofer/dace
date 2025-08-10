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
    path = np.fromfunction(lambda i, j: i * j % 7 + 1, (N, N), dtype=datatype)
    for i in range(N):
        for j in range(N):
            if (i + j) % 13 == 0 or (i + j) % 7 == 0 or (i + j) % 11 == 0:
                path[i, j] = 999

    return path


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/floyd_warshall_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (200)
    M = (400)
    L = (850)
    PAPER = (2800)

    def __init__(self, N):
        self.N = N

    def init(self):
        path = initialize(N=self.N)
        return self.N, path

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, path = size.init()

    path_ref = np.copy(path)
    sdfg(N=N, path=path_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    path2 = np.copy(path)
    sdfg_copy(N=N, _path=path2)

    diff_path = np.sqrt(np.mean((path2 - path_ref)**2))
    diff = max(np.max(np.abs(path2 - path_ref)))
    print("RMS Difference PATH:", diff_path)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

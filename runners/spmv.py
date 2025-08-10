# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(M, N, nnz):
    from numpy.random import default_rng
    rng = default_rng(42)

    x = rng.random((N, ))

    from scipy.sparse import random

    matrix = random(M,
                    N,
                    density=nnz / (M * N),
                    format='csr',
                    dtype=np.float64,
                    random_state=rng)
    rows = np.uint32(matrix.indptr)
    cols = np.uint32(matrix.indices)
    vals = matrix.data

    return rows, cols, vals, x


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/spmv_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (4096, 4096, 8192)
    M = (32768, 32768, 65536)
    L = (262144, 262144, 262144)
    PAPER = (131072, 131072, 262144)

    def __init__(self, M, N, nnz):
        self.M = M
        self.N = N
        self.nnz = nnz

    def init(self):
        A_row, A_col, A_val, x = initialize(M=self.M, N=self.N, nnz=self.nnz)
        return self.M, self.N, self.nnz, A_row, A_col, A_val, x

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    M, N, nnz, A_row, A_col, A_val, x = size.init()

    A_row_ref = np.copy(A_row)
    A_col_ref = np.copy(A_col)
    A_val_ref = np.copy(A_val)
    x_ref = np.copy(x)
    sdfg(M=M, N=N, nnz=nnz, A_row=A_row_ref, A_col=A_col_ref, A_val=A_val_ref, x=x_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    A_row2 = np.copy(A_row)
    A_col2 = np.copy(A_col)
    A_val2 = np.copy(A_val)
    x2 = np.copy(x)
    sdfg_copy(M=M, N=N, nnz=nnz, _A_row=A_row2, _A_col=A_col2, _A_val=A_val2, _x=x2)

    diff_A_row = np.sqrt(np.mean((A_row2 - A_row_ref)**2))
    diff_A_col = np.sqrt(np.mean((A_col2 - A_col_ref)**2))
    diff_A_val = np.sqrt(np.mean((A_val2 - A_val_ref)**2))
    diff_x = np.sqrt(np.mean((x2 - x_ref)**2))
    diff = max(np.max(np.abs(A_row2 - A_row_ref)), np.max(np.abs(A_col2 - A_col_ref)), np.max(np.abs(A_val2 - A_val_ref)), np.max(np.abs(x2 - x_ref)))
    print("RMS Difference A_ROW:", diff_A_row)
    print("RMS Difference A_COL:", diff_A_col)
    print("RMS Difference A_VAL:", diff_A_val)
    print("RMS Difference X:", diff_x)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (M={model_size.M}, N={model_size.N}, NNZ={model_size.nnz})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

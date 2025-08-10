# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    fn = datatype(N)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=datatype)
    u1 = np.fromfunction(lambda i: i, (N, ), dtype=datatype)
    u2 = np.fromfunction(lambda i: ((i + 1) / fn) / 2.0, (N, ), dtype=datatype)
    v1 = np.fromfunction(lambda i: ((i + 1) / fn) / 4.0, (N, ), dtype=datatype)
    v2 = np.fromfunction(lambda i: ((i + 1) / fn) / 6.0, (N, ), dtype=datatype)
    w = np.zeros((N, ), dtype=datatype)
    x = np.zeros((N, ), dtype=datatype)
    y = np.fromfunction(lambda i: ((i + 1) / fn) / 8.0, (N, ), dtype=datatype)
    z = np.fromfunction(lambda i: ((i + 1) / fn) / 9.0, (N, ), dtype=datatype)

    return alpha, beta, A, u1, v1, u2, v2, w, x, y, z


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/gemver_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (1000)
    M = (3000)
    L = (10000)
    PAPER = (8000)

    def __init__(self, N):
        self.N = N

    def init(self):
        alpha, beta, A, u1, v1, u2, v2, w, x, y, z = initialize(N=self.N)
        return self.N, alpha, beta, A, u1, v1, u2, v2, w, x, y, z

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, alpha, beta, A, u1, v1, u2, v2, w, x, y, z = size.init()

    alpha_ref = np.copy(alpha)
    beta_ref = np.copy(beta)
    A_ref = np.copy(A)
    u1_ref = np.copy(u1)
    v1_ref = np.copy(v1)
    u2_ref = np.copy(u2)
    v2_ref = np.copy(v2)
    w_ref = np.copy(w)
    x_ref = np.copy(x)
    y_ref = np.copy(y)
    z_ref = np.copy(z)
    sdfg(N=N, alpha=alpha_ref, beta=beta_ref, A=A_ref, u1=u1_ref, v1=v1_ref, u2=u2_ref, v2=v2_ref, w=w_ref, x=x_ref, y=y_ref, z=z_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    alpha2 = np.copy(alpha)
    beta2 = np.copy(beta)
    A2 = np.copy(A)
    u12 = np.copy(u1)
    v12 = np.copy(v1)
    u22 = np.copy(u2)
    v22 = np.copy(v2)
    w2 = np.copy(w)
    x2 = np.copy(x)
    y2 = np.copy(y)
    z2 = np.copy(z)
    sdfg_copy(N=N, _alpha=alpha2, _beta=beta2, _A=A2, _u1=u12, _v1=v12, _u2=u22, _v2=v22, _w=w2, _x=x2, _y=y2, _z=z2)

    diff_alpha = np.sqrt(np.mean((alpha2 - alpha_ref)**2))
    diff_beta = np.sqrt(np.mean((beta2 - beta_ref)**2))
    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff_u1 = np.sqrt(np.mean((u12 - u1_ref)**2))
    diff_v1 = np.sqrt(np.mean((v12 - v1_ref)**2))
    diff_u2 = np.sqrt(np.mean((u22 - u2_ref)**2))
    diff_v2 = np.sqrt(np.mean((v22 - v2_ref)**2))
    diff_w = np.sqrt(np.mean((w2 - w_ref)**2))
    diff_x = np.sqrt(np.mean((x2 - x_ref)**2))
    diff_y = np.sqrt(np.mean((y2 - y_ref)**2))
    diff_z = np.sqrt(np.mean((z2 - z_ref)**2))
    diff = max(np.max(np.abs(alpha2 - alpha_ref)), np.max(np.abs(beta2 - beta_ref)), np.max(np.abs(A2 - A_ref)), np.max(np.abs(u12 - u1_ref)), np.max(np.abs(v12 - v1_ref)), np.max(np.abs(u22 - u2_ref)), np.max(np.abs(v22 - v2_ref)), np.max(np.abs(w2 - w_ref)), np.max(np.abs(x2 - x_ref)), np.max(np.abs(y2 - y_ref)), np.max(np.abs(z2 - z_ref)))
    print("RMS Difference ALPHA:", diff_alpha)
    print("RMS Difference BETA:", diff_beta)
    print("RMS Difference A:", diff_A)
    print("RMS Difference U1:", diff_u1)
    print("RMS Difference V1:", diff_v1)
    print("RMS Difference U2:", diff_u2)
    print("RMS Difference V2:", diff_v2)
    print("RMS Difference W:", diff_w)
    print("RMS Difference X:", diff_x)
    print("RMS Difference Y:", diff_y)
    print("RMS Difference Z:", diff_z)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

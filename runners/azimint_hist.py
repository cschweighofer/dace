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
    data, radius = rng.random((N, )), rng.random((N, ))
    return data, radius


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/azimint_hist_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (400000, 1000)
    M = (4000000, 1000)
    L = (40000000, 1000)
    PAPER = (1000000, 1000)

    def __init__(self, N, npt):
        self.N = N
        self.npt = npt

    def init(self):
        data, radius = initialize(N=self.N)
        return self.N, self.npt, data, radius

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, npt, data, radius = size.init()

    data_ref = np.copy(data)
    radius_ref = np.copy(radius)
    sdfg(N=N, npt=npt, data=data_ref, radius=radius_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    data2 = np.copy(data)
    radius2 = np.copy(radius)
    sdfg_copy(N=N, npt=npt, _data=data2, _radius=radius2)

    diff_data = np.sqrt(np.mean((data2 - data_ref)**2))
    diff_radius = np.sqrt(np.mean((radius2 - radius_ref)**2))
    diff = max(np.max(np.abs(data2 - data_ref)), np.max(np.abs(radius2 - radius_ref)))
    print("RMS Difference DATA:", diff_data)
    print("RMS Difference RADIUS:", diff_radius)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N}, NPT={model_size.npt})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

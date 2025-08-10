# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def rng_complex(shape, rng):
    return (rng.random(shape) + rng.random(shape) * 1j)


def initialize(NR, NM, slab_per_bc, num_int_pts):
    from numpy.random import default_rng
    rng = default_rng(42)
    Ham = rng_complex((slab_per_bc + 1, NR, NR), rng)
    int_pts = rng_complex((num_int_pts, ), rng)
    Y = rng_complex((NR, NM), rng)
    return Ham, int_pts, Y


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/contour_integral_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (50, 150, 2, 32)
    M = (200, 400, 2, 32)
    L = (600, 1000, 2, 32)
    PAPER = (500, 1000, 2, 32)

    def __init__(self, NR, NM, slab_per_bc, num_int_pts):
        self.NR = NR
        self.NM = NM
        self.slab_per_bc = slab_per_bc
        self.num_int_pts = num_int_pts

    def init(self):
        Ham, int_pts, Y = initialize(NR=self.NR, NM=self.NM, slab_per_bc=self.slab_per_bc, num_int_pts=self.num_int_pts)
        return self.NR, self.NM, self.slab_per_bc, self.num_int_pts, Ham, int_pts, Y

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    NR, NM, slab_per_bc, num_int_pts, Ham, int_pts, Y = size.init()

    Ham_ref = np.copy(Ham)
    int_pts_ref = np.copy(int_pts)
    Y_ref = np.copy(Y)
    sdfg(NR=NR, NM=NM, slab_per_bc=slab_per_bc, num_int_pts=num_int_pts, Ham=Ham_ref, int_pts=int_pts_ref, Y=Y_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    Ham2 = np.copy(Ham)
    int_pts2 = np.copy(int_pts)
    Y2 = np.copy(Y)
    sdfg_copy(NR=NR, NM=NM, slab_per_bc=slab_per_bc, num_int_pts=num_int_pts, _Ham=Ham2, _int_pts=int_pts2, _Y=Y2)

    diff_Ham = np.sqrt(np.mean((Ham2 - Ham_ref)**2))
    diff_int_pts = np.sqrt(np.mean((int_pts2 - int_pts_ref)**2))
    diff_Y = np.sqrt(np.mean((Y2 - Y_ref)**2))
    diff = max(np.max(np.abs(Ham2 - Ham_ref)), np.max(np.abs(int_pts2 - int_pts_ref)), np.max(np.abs(Y2 - Y_ref)))
    print("RMS Difference HAM:", diff_Ham)
    print("RMS Difference INT_PTS:", diff_int_pts)
    print("RMS Difference Y:", diff_Y)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (NR={model_size.NR}, NM={model_size.NM}, SLAB_PER_BC={model_size.slab_per_bc}, NUM_INT_PTS={model_size.num_int_pts})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

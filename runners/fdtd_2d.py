# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(TMAX, NX, NY, datatype=np.float64):
    ex = np.fromfunction(lambda i, j: (i * (j + 1)) / NX, (NX, NY),
                         dtype=datatype)
    ey = np.fromfunction(lambda i, j: (i * (j + 2)) / NY, (NX, NY),
                         dtype=datatype)
    hz = np.fromfunction(lambda i, j: (i * (j + 3)) / NX, (NX, NY),
                         dtype=datatype)
    _fict_ = np.fromfunction(lambda i: i, (TMAX, ), dtype=datatype)

    return ex, ey, hz, _fict_


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/fdtd_2d_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (20, 200, 220)
    M = (60, 400, 450)
    # L = (150, 800, 900)
    # PAPER = (500, 1000, 1200)

    def __init__(self, TMAX, NX, NY):
        self.TMAX = TMAX
        self.NX = NX
        self.NY = NY

    def init(self):
        ex, ey, hz, _fict_ = initialize(TMAX=self.TMAX, NX=self.NX, NY=self.NY)
        return self.TMAX, self.NX, self.NY, ex, ey, hz, _fict_

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    TMAX, NX, NY, ex, ey, hz, _fict_ = size.init()

    ex_ref = np.copy(ex)
    ey_ref = np.copy(ey)
    hz_ref = np.copy(hz)
    _fict__ref = np.copy(_fict_)
    sdfg(TMAX=TMAX, NX=NX, NY=NY, ex=ex_ref, ey=ey_ref, hz=hz_ref, _fict_=_fict__ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    ex2 = np.copy(ex)
    ey2 = np.copy(ey)
    hz2 = np.copy(hz)
    _fict_2 = np.copy(_fict_)
    sdfg_copy(TMAX=TMAX, NX=NX, NY=NY, _ex=ex2, _ey=ey2, _hz=hz2, __fict_=_fict_2)

    diff_ex = np.sqrt(np.mean((ex2 - ex_ref)**2))
    diff_ey = np.sqrt(np.mean((ey2 - ey_ref)**2))
    diff_hz = np.sqrt(np.mean((hz2 - hz_ref)**2))
    diff__fict_ = np.sqrt(np.mean((_fict_2 - _fict__ref)**2))
    diff = max(np.max(np.abs(ex2 - ex_ref)), np.max(np.abs(ey2 - ey_ref)), np.max(np.abs(hz2 - hz_ref)), np.max(np.abs(_fict_2 - _fict__ref)))
    print("RMS Difference EX:", diff_ex)
    print("RMS Difference EY:", diff_ey)
    print("RMS Difference HZ:", diff_hz)
    print("RMS Difference _FICT_:", diff__fict_)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (TMAX={model_size.TMAX}, NX={model_size.NX}, NY={model_size.NY})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(I, J, K):
    from numpy.random import default_rng
    rng = default_rng(42)

    dtr_stage = 3. / 20.

    # Define arrays
    utens_stage = rng.random((I, J, K))
    u_stage = rng.random((I, J, K))
    wcon = rng.random((I + 1, J, K))
    u_pos = rng.random((I, J, K))
    utens = rng.random((I, J, K))

    return dtr_stage, utens_stage, u_stage, wcon, u_pos, utens


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/vadv_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (60, 60, 40)
    M = (112, 112, 80)
    # L = (180, 180, 160)
    # PAPER = (256, 256, 160)

    def __init__(self, I, J, K):
        self.I = I
        self.J = J
        self.K = K

    def init(self):
        dtr_stage, utens_stage, u_stage, wcon, u_pos, utens = initialize(I=self.I, J=self.J, K=self.K)
        return self.I, self.J, self.K, dtr_stage, utens_stage, u_stage, wcon, u_pos, utens

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    I, J, K, dtr_stage, utens_stage, u_stage, wcon, u_pos, utens = size.init()

    utens_stage_ref = np.copy(utens_stage)
    u_stage_ref = np.copy(u_stage)
    wcon_ref = np.copy(wcon)
    u_pos_ref = np.copy(u_pos)
    utens_ref = np.copy(utens)
    dtr_stage_ref = np.copy(dtr_stage)
    sdfg(I=I, J=J, K=K, utens_stage=utens_stage_ref, u_stage=u_stage_ref, wcon=wcon_ref, u_pos=u_pos_ref, utens=utens_ref, dtr_stage=dtr_stage_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    utens_stage2 = np.copy(utens_stage)
    u_stage2 = np.copy(u_stage)
    wcon2 = np.copy(wcon)
    u_pos2 = np.copy(u_pos)
    utens2 = np.copy(utens)
    dtr_stage2 = np.copy(dtr_stage)
    sdfg_copy(I=I, J=J, K=K, _utens_stage=utens_stage2, _u_stage=u_stage2, _wcon=wcon2, _u_pos=u_pos2, _utens=utens2, _dtr_stage=dtr_stage2)

    diff_utens_stage = np.sqrt(np.mean((utens_stage2 - utens_stage_ref)**2))
    diff_u_stage = np.sqrt(np.mean((u_stage2 - u_stage_ref)**2))
    diff_wcon = np.sqrt(np.mean((wcon2 - wcon_ref)**2))
    diff_u_pos = np.sqrt(np.mean((u_pos2 - u_pos_ref)**2))
    diff_utens = np.sqrt(np.mean((utens2 - utens_ref)**2))
    diff_dtr_stage = np.sqrt(np.mean((dtr_stage2 - dtr_stage_ref)**2))
    diff = max(np.max(np.abs(utens_stage2 - utens_stage_ref)), np.max(np.abs(u_stage2 - u_stage_ref)), np.max(np.abs(wcon2 - wcon_ref)), np.max(np.abs(u_pos2 - u_pos_ref)), np.max(np.abs(utens2 - utens_ref)), np.max(np.abs(dtr_stage2 - dtr_stage_ref)))
    print("RMS Difference UTENS_STAGE:", diff_utens_stage)
    print("RMS Difference U_STAGE:", diff_u_stage)
    print("RMS Difference WCON:", diff_wcon)
    print("RMS Difference U_POS:", diff_u_pos)
    print("RMS Difference UTENS:", diff_utens)
    print("RMS Difference DTR_STAGE:", diff_dtr_stage)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (I={model_size.I}, J={model_size.J}, K={model_size.K})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

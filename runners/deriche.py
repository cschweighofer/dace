# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(W, H, datatype=np.float64):
    alpha = datatype(0.25)
    imgIn = np.fromfunction(lambda i, j:
                            ((313 * i + 991 * j) % 65536) / 65535.0, (W, H),
                            dtype=datatype)

    return alpha, imgIn


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/deriche_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (400, 200)
    M = (1500, 1000)
    L = (6000, 3000)
    PAPER = (7680, 4320)

    def __init__(self, W, H):
        self.W = W
        self.H = H

    def init(self):
        alpha, imgIn = initialize(W=self.W, H=self.H)
        return self.W, self.H, alpha, imgIn

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    W, H, alpha, imgIn = size.init()

    alpha_ref = np.copy(alpha)
    imgIn_ref = np.copy(imgIn)
    sdfg(W=W, H=H, alpha=alpha_ref, imgIn=imgIn_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    alpha2 = np.copy(alpha)
    imgIn2 = np.copy(imgIn)
    sdfg_copy(W=W, H=H, _alpha=alpha2, _imgIn=imgIn2)

    diff_alpha = np.sqrt(np.mean((alpha2 - alpha_ref)**2))
    diff_imgIn = np.sqrt(np.mean((imgIn2 - imgIn_ref)**2))
    diff = max(np.max(np.abs(alpha2 - alpha_ref)), np.max(np.abs(imgIn2 - imgIn_ref)))
    print("RMS Difference ALPHA:", diff_alpha)
    print("RMS Difference IMGIN:", diff_imgIn)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (W={model_size.W}, H={model_size.H})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

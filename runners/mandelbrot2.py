# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np

# No initialization needed


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/mandelbrot2_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (-2.0, 0.5, 200, -1.25, 1.25, 200, 40, 2.0)
    M = (-2.0, 0.5, 500, -1.25, 1.25, 500, 80, 2.0)
    L = (-2.25, 0.75, 1000, -1.5, 1.5, 1000, 100, 2.0)
    PAPER = (-2.25, 0.75, 1000, -1.25, 1.25, 1000, 200, 2.0)

    def __init__(self, xmin, xmax, XN, ymin, ymax, YN, maxiter, horizon):
        self.xmin = xmin
        self.xmax = xmax
        self.XN = XN
        self.ymin = ymin
        self.ymax = ymax
        self.YN = YN
        self.maxiter = maxiter
        self.horizon = horizon

    def init(self):
        pass
        return self.xmin, self.xmax, self.XN, self.ymin, self.ymax, self.YN, self.maxiter, self.horizon

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    xmin, xmax, XN, ymin, ymax, YN, maxiter, horizon = size.init()

    
    sdfg(xmin=xmin, xmax=xmax, XN=XN, ymin=ymin, ymax=ymax, YN=YN, maxiter=maxiter, horizon=horizon)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    
    sdfg_copy(xmin=xmin, xmax=xmax, XN=XN, ymin=ymin, ymax=ymax, YN=YN, maxiter=maxiter, horizon=horizon)

    
    

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (XMIN={model_size.xmin}, XMAX={model_size.xmax}, XN={model_size.XN}, YMIN={model_size.ymin}, YMAX={model_size.ymax}, YN={model_size.YN}, MAXITER={model_size.maxiter}, HORIZON={model_size.horizon})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

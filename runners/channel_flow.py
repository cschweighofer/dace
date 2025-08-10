# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(ny, nx):
    u = np.zeros((ny, nx), dtype=np.float64)
    v = np.zeros((ny, nx), dtype=np.float64)
    p = np.ones((ny, nx), dtype=np.float64)
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = .1 / ((nx - 1) * (ny - 1))
    return u, v, p, dx, dy, dt


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/channel_flow_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (61, 61, 5, 1.0, 0.1, 1.0)
    M = (121, 121, 10, 1.0, 0.1, 1.0)
    L = (201, 201, 20, 1.0, 0.1, 1.0)
    PAPER = (101, 101, 50, 1.0, 0.1, 1.0)

    def __init__(self, ny, nx, nit, rho, nu, F):
        self.ny = ny
        self.nx = nx
        self.nit = nit
        self.rho = rho
        self.nu = nu
        self.F = F

    def init(self):
        u, v, p, dx, dy, dt = initialize(ny=self.ny, nx=self.nx)
        return self.ny, self.nx, self.nit, self.rho, self.nu, self.F, u, v, p, dx, dy, dt

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    ny, nx, nit, rho, nu, F, u, v, p, dx, dy, dt = size.init()

    u_ref = np.copy(u)
    v_ref = np.copy(v)
    dt_ref = np.copy(dt)
    dx_ref = np.copy(dx)
    dy_ref = np.copy(dy)
    p_ref = np.copy(p)
    sdfg(ny=ny, nx=nx, nit=nit, rho=rho, nu=nu, F=F, u=u_ref, v=v_ref, dt=dt_ref, dx=dx_ref, dy=dy_ref, p=p_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    u2 = np.copy(u)
    v2 = np.copy(v)
    dt2 = np.copy(dt)
    dx2 = np.copy(dx)
    dy2 = np.copy(dy)
    p2 = np.copy(p)
    sdfg_copy(ny=ny, nx=nx, nit=nit, rho=rho, nu=nu, F=F, _u=u2, _v=v2, _dt=dt2, _dx=dx2, _dy=dy2, _p=p2)

    diff_u = np.sqrt(np.mean((u2 - u_ref)**2))
    diff_v = np.sqrt(np.mean((v2 - v_ref)**2))
    diff_dt = np.sqrt(np.mean((dt2 - dt_ref)**2))
    diff_dx = np.sqrt(np.mean((dx2 - dx_ref)**2))
    diff_dy = np.sqrt(np.mean((dy2 - dy_ref)**2))
    diff_p = np.sqrt(np.mean((p2 - p_ref)**2))
    diff = max(np.max(np.abs(u2 - u_ref)), np.max(np.abs(v2 - v_ref)), np.max(np.abs(dt2 - dt_ref)), np.max(np.abs(dx2 - dx_ref)), np.max(np.abs(dy2 - dy_ref)), np.max(np.abs(p2 - p_ref)))
    print("RMS Difference U:", diff_u)
    print("RMS Difference V:", diff_v)
    print("RMS Difference DT:", diff_dt)
    print("RMS Difference DX:", diff_dx)
    print("RMS Difference DY:", diff_dy)
    print("RMS Difference P:", diff_p)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (NY={model_size.ny}, NX={model_size.nx}, NIT={model_size.nit}, RHO={model_size.rho}, NU={model_size.nu}, F={model_size.F})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

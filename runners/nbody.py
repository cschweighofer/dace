# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, tEnd, dt):
    from numpy.random import default_rng
    rng = default_rng(42)
    mass = 20.0 * np.ones((N, 1)) / N  # total mass of particles is 20
    pos = rng.random((N, 3))  # randomly selected positions and velocities
    vel = rng.random((N, 3))
    Nt = int(np.ceil(tEnd / dt))
    return mass, pos, vel, Nt


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/nbody_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (25, 2.0, 0.05, 0.1, 1.0)
    M = (50, 5.0, 0.02, 0.1, 1.0)
    L = (100, 9.0, 0.01, 0.1, 1.0)
    PAPER = (100, 10.0, 0.01, 0.1, 1.0)

    def __init__(self, N, tEnd, dt, softening, G):
        self.N = N
        self.tEnd = tEnd
        self.dt = dt
        self.softening = softening
        self.G = G

    def init(self):
        mass, pos, vel, Nt = initialize(N=self.N, tEnd=self.tEnd, dt=self.dt)
        return self.N, self.tEnd, self.dt, self.softening, self.G, mass, pos, vel, Nt

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, tEnd, dt, softening, G, mass, pos, vel, Nt = size.init()

    mass_ref = np.copy(mass)
    pos_ref = np.copy(pos)
    vel_ref = np.copy(vel)
    Nt_ref = np.copy(Nt)
    sdfg(N=N, tEnd=tEnd, dt=dt, softening=softening, G=G, mass=mass_ref, pos=pos_ref, vel=vel_ref, Nt=Nt_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    mass2 = np.copy(mass)
    pos2 = np.copy(pos)
    vel2 = np.copy(vel)
    Nt2 = np.copy(Nt)
    sdfg_copy(N=N, tEnd=tEnd, dt=dt, softening=softening, G=G, _mass=mass2, _pos=pos2, _vel=vel2, _Nt=Nt2)

    diff_mass = np.sqrt(np.mean((mass2 - mass_ref)**2))
    diff_pos = np.sqrt(np.mean((pos2 - pos_ref)**2))
    diff_vel = np.sqrt(np.mean((vel2 - vel_ref)**2))
    diff_Nt = np.sqrt(np.mean((Nt2 - Nt_ref)**2))
    diff = max(np.max(np.abs(mass2 - mass_ref)), np.max(np.abs(pos2 - pos_ref)), np.max(np.abs(vel2 - vel_ref)), np.max(np.abs(Nt2 - Nt_ref)))
    print("RMS Difference MASS:", diff_mass)
    print("RMS Difference POS:", diff_pos)
    print("RMS Difference VEL:", diff_vel)
    print("RMS Difference NT:", diff_Nt)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N}, TEND={model_size.tEnd}, DT={model_size.dt}, SOFTENING={model_size.softening}, G={model_size.G})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

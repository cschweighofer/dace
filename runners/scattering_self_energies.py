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


def initialize(Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb):
    from numpy.random import default_rng
    rng = default_rng(42)

    neigh_idx = np.ndarray([NA, NB], dtype=np.int32)
    for i in range(NA):
        neigh_idx[i] = np.positive(np.arange(i - NB / 2, i + NB / 2) % NA)
    dH = rng_complex([NA, NB, N3D, Norb, Norb], rng)
    G = rng_complex([Nkz, NE, NA, Norb, Norb], rng)
    D = rng_complex([Nqz, Nw, NA, NB, N3D, N3D], rng)
    Sigma = np.zeros([Nkz, NE, NA, Norb, Norb], dtype=np.complex128)

    return neigh_idx, dH, G, D, Sigma


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/scattering_self_energies_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (2, 4, 2, 2, 2, 6, 2, 3)
    M = (3, 5, 3, 2, 2, 10, 3, 4)
    L = (3, 10, 3, 3, 3, 12, 4, 4)
    PAPER = (4, 10, 4, 3, 3, 20, 4, 4)

    def __init__(self, Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb):
        self.Nkz = Nkz
        self.NE = NE
        self.Nqz = Nqz
        self.Nw = Nw
        self.N3D = N3D
        self.NA = NA
        self.NB = NB
        self.Norb = Norb

    def init(self):
        neigh_idx, dH, G, D, Sigma = initialize(Nkz=self.Nkz, NE=self.NE, Nqz=self.Nqz, Nw=self.Nw, N3D=self.N3D, NA=self.NA, NB=self.NB, Norb=self.Norb)
        return self.Nkz, self.NE, self.Nqz, self.Nw, self.N3D, self.NA, self.NB, self.Norb, neigh_idx, dH, G, D, Sigma

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb, neigh_idx, dH, G, D, Sigma = size.init()

    neigh_idx_ref = np.copy(neigh_idx)
    dH_ref = np.copy(dH)
    G_ref = np.copy(G)
    D_ref = np.copy(D)
    Sigma_ref = np.copy(Sigma)
    sdfg(Nkz=Nkz, NE=NE, Nqz=Nqz, Nw=Nw, N3D=N3D, NA=NA, NB=NB, Norb=Norb, neigh_idx=neigh_idx_ref, dH=dH_ref, G=G_ref, D=D_ref, Sigma=Sigma_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    neigh_idx2 = np.copy(neigh_idx)
    dH2 = np.copy(dH)
    G2 = np.copy(G)
    D2 = np.copy(D)
    Sigma2 = np.copy(Sigma)
    sdfg_copy(Nkz=Nkz, NE=NE, Nqz=Nqz, Nw=Nw, N3D=N3D, NA=NA, NB=NB, Norb=Norb, _neigh_idx=neigh_idx2, _dH=dH2, _G=G2, _D=D2, _Sigma=Sigma2)

    diff_neigh_idx = np.sqrt(np.mean((neigh_idx2 - neigh_idx_ref)**2))
    diff_dH = np.sqrt(np.mean((dH2 - dH_ref)**2))
    diff_G = np.sqrt(np.mean((G2 - G_ref)**2))
    diff_D = np.sqrt(np.mean((D2 - D_ref)**2))
    diff_Sigma = np.sqrt(np.mean((Sigma2 - Sigma_ref)**2))
    diff = max(np.max(np.abs(neigh_idx2 - neigh_idx_ref)), np.max(np.abs(dH2 - dH_ref)), np.max(np.abs(G2 - G_ref)), np.max(np.abs(D2 - D_ref)), np.max(np.abs(Sigma2 - Sigma_ref)))
    print("RMS Difference NEIGH_IDX:", diff_neigh_idx)
    print("RMS Difference DH:", diff_dH)
    print("RMS Difference G:", diff_G)
    print("RMS Difference D:", diff_D)
    print("RMS Difference SIGMA:", diff_Sigma)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (NKZ={model_size.Nkz}, NE={model_size.NE}, NQZ={model_size.Nqz}, NW={model_size.Nw}, N3D={model_size.N3D}, NA={model_size.NA}, NB={model_size.NB}, NORB={model_size.Norb})")
        run_model(model_size)

if __name__ == "__main__":
    run_all_models()

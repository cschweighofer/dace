# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype
from enhanced_input_sensitivity_analysis import (
    AdaptiveSensitivityAnalyzer, NoiseSpec, NoiseType, DistributionType,
    plot_grid_sensitivity_over_sizes_rmse
)
# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float64):
    A = np.fromfunction(lambda i, j, k: (i + j + (N - k)) * 10 / N, (N, N, N),
                        dtype=datatype)
    B = np.copy(A)

    return A, B


sdfg = dace.SDFG.from_file("/home/chris/SPCL/npbench/runners/sdfgs/heat_3d_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (25, 25)
    M = (50, 40)
    L = (100, 70)
    PAPER = (200, 100)
    # PAPER = (500, 120)

    def __init__(self, TSTEPS, N):
        self.TSTEPS = TSTEPS
        self.N = N

    def init(self):
        A, B = initialize(N=self.N)
        return self.TSTEPS, self.N, A, B

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    TSTEPS, N, A, B = size.init()

    A_ref = np.copy(A)
    B_ref = np.copy(B)
    sdfg(TSTEPS=TSTEPS, N=N, A=A_ref, B=B_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    A2 = np.copy(A)
    B2 = np.copy(B)
    sdfg_copy(TSTEPS=TSTEPS, N=N, _A=A2, _B=B2)

    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff_B = np.sqrt(np.mean((B2 - B_ref)**2))
    diff = max(np.max(np.abs(A2 - A_ref)), np.max(np.abs(B2 - B_ref)))
    print("RMS Difference A:", diff_A)
    print("RMS Difference B:", diff_B)
    print("Max Difference:", diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (TSTEPS={model_size.TSTEPS}, N={model_size.N})")
        run_model(model_size)

def run_sensitivity_analysis(size, target_input, noise_level, num_trials):

    print(f"Running sensitivity analysis on {target_input} with {size.name} model")
    
    # Initialize data
    TSTEPS, N, A, B = size.init()
    
    sdfg_specialized = copy.deepcopy(sdfg)
    sdfg_specialized.specialize({'N': N})

    baseline_inputs = {
        'TSTEPS': TSTEPS,
        'A': A.copy(),
        'B': B.copy()
    }
    
    # Create noise specification
    noise_spec = NoiseSpec(
        noise_type=NoiseType.MULTIPLICATIVE,
        noise_distribution=DistributionType.NORMAL,
        noise_level=noise_level
    )
    
    # Create analyzer and run analysis with specialized SDFG
    analyzer = AdaptiveSensitivityAnalyzer(sdfg_specialized, compile_sdfg=True)
    
    try:
        results = analyzer.perform_single_input_sensitivity_analysis(
            target_input_name=target_input,
            noise_spec=noise_spec,
            baseline_inputs=baseline_inputs,
            num_trials=num_trials
        )
        
        # Generate report
        report = analyzer.generate_report(results, f'heat_3d_{target_input}_sensitivity_report.txt')
        print(report)
        
        # Generate plots
        analyzer.plot_sensitivity_results(results, f'heat_3d_{target_input}_sensitivity_plots.png')
        
        return results, analyzer
        
    except Exception as e:
        print(f"Sensitivity analysis failed: {e}")
        return None, None

def run_quick_sensitivity_analysis():
    inputs_to_test = ['A', 'B']

    print("Running quick sensitivity analysis on all heat_3d inputs...")
    print("=" * 60)
    
    for input_name in inputs_to_test:
        print(f"\nAnalyzing sensitivity to {input_name}...")
        run_sensitivity_analysis(ModelSize.M, input_name, 0.01, num_trials=1)
        print("-" * 40)


def run_grid_sensitivity_analysis():
    # Define columns (sizes)
    def make_init_fn(ms: ModelSize):
        def _fn():
            TSTEPS, N, A, B = ms.init()
            specialize = {'N': N}
            baseline = {'TSTEPS': TSTEPS, 'A': A, 'B': B}
            return specialize, baseline
        return _fn

    sizes = [
        ('S', make_init_fn(ModelSize.S)),
        ('M', make_init_fn(ModelSize.M)),
        ('L', make_init_fn(ModelSize.L)),
        ('PAPER', make_init_fn(ModelSize.PAPER)),
    ]

    # Define rows (multiplicative noise specs):
    noise_specs = [
        ('G-σ1', NoiseSpec(
            noise_type=NoiseType.MULTIPLICATIVE,
            noise_distribution=DistributionType.NORMAL,
            noise_level=0.01
        )),
        ('G-σ2', NoiseSpec(
            noise_type=NoiseType.MULTIPLICATIVE,
            noise_distribution=DistributionType.NORMAL,
            noise_level=0.05
        )),
        ('L-b1', NoiseSpec(
            noise_type=NoiseType.MULTIPLICATIVE,
            noise_distribution=DistributionType.LAPLACE,
            noise_level=0.007071
        )),
        ('L-b2', NoiseSpec(
            noise_type=NoiseType.MULTIPLICATIVE,
            noise_distribution=DistributionType.LAPLACE,
            noise_level=0.035355
        )),
    ]

    # Build one 4x4 grid per target input (A and B)
    plot_grid_sensitivity_over_sizes_rmse(
        base_sdfg=sdfg,
        sizes=sizes,
        noise_specs=noise_specs,
    target_input_name='A',
        save_path='heat_3d_grid_A.png'
    )
    plot_grid_sensitivity_over_sizes_rmse(
        base_sdfg=sdfg,
        sizes=sizes,
        noise_specs=noise_specs,
    target_input_name='B',
        save_path='heat_3d_grid_B.png'
    )

if __name__ == "__main__":
    # run_all_models()
    # run_quick_sensitivity_analysis()
    run_grid_sensitivity_analysis()

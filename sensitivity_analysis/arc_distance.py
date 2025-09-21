# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from real_lowering import change_fptype

# Import sensitivity analysis
try:
    from enhanced_input_sensitivity_analysis import (
    AdaptiveSensitivityAnalyzer, NoiseSpec, NoiseType, DistributionType,
    plot_grid_sensitivity_over_sizes_rmse
    )
    SENSITIVITY_AVAILABLE = True
except ImportError:
    print("Enhanced sensitivity analysis not available")
    SENSITIVITY_AVAILABLE = False

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    t0, p0, t1, p1 = rng.random((N, )), rng.random((N, )), rng.random(
        (N, )), rng.random((N, ))
    return t0, p0, t1, p1


# Use workspace-relative path that exists
sdfg = dace.SDFG.from_file("/home/chris/SPCL/dace/runners/sdfgs/arc_distance_auto_opt_cpu.sdfg")

class ModelSize(Enum):
    S = (100000)
    M = (1000000)
    L = (10000000)
    PAPER = (10000000)

    def __init__(self, N):
        self.N = N

    def init(self):
        theta_1, phi_1, theta_2, phi_2 = initialize(N=self.N)
        return self.N, theta_1, phi_1, theta_2, phi_2

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    N, theta_1, phi_1, theta_2, phi_2 = size.init()

    theta_1_ref = np.copy(theta_1)
    phi_1_ref = np.copy(phi_1)
    theta_2_ref = np.copy(theta_2)
    phi_2_ref = np.copy(phi_2)
    
    # Allocate output buffer for the distance matrix
    result_ref = np.zeros(N, dtype=np.float64)
    sdfg(N=N, theta_1=theta_1_ref, phi_1=phi_1_ref, theta_2=theta_2_ref, phi_2=phi_2_ref, __return=result_ref)

    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    theta_12 = np.copy(theta_1)
    phi_12 = np.copy(phi_1)
    theta_22 = np.copy(theta_2)
    phi_22 = np.copy(phi_2)
    
    # Allocate output buffer for the modified precision distance matrix
    result_mpf = np.zeros(N, dtype=np.float64)
    sdfg_copy(N=N, _theta_1=theta_12, _phi_1=phi_12, _theta_2=theta_22, _phi_2=phi_22, ___return=result_mpf)

    # Compare the results instead of the input arrays
    diff_result = np.sqrt(np.mean((result_mpf - result_ref)**2))
    max_diff = np.max(np.abs(result_mpf - result_ref))
    print("RMS Difference in results:", diff_result)
    print("Max Difference in results:", max_diff)

def run_all_models():
    """Run the model with all size configurations."""
    for model_size in ModelSize:
        print(f"Running {model_size.name} model (N={model_size.N})")
        run_model(model_size)

def run_sensitivity_analysis(size: ModelSize = ModelSize.S, target_input: str = 'theta_1', 
                           noise_level: float = 0.01, num_trials: int = 3):
    """
    Run sensitivity analysis on arc_distance with specified parameters.
    
    Args:
        size: Model size to use (S, M, L, PAPER)
        target_input: Input to analyze ('theta_1', 'phi_1', 'theta_2', 'phi_2')
        noise_level: Level of noise to apply
        num_trials: Number of trials to run
    """
    if not SENSITIVITY_AVAILABLE:
        print("Sensitivity analysis not available. Please ensure enhanced_input_sensitivity_analysis.py is working.")
        return
    
    print(f"Running sensitivity analysis on {target_input} with {size.name} model")
    
    # Initialize data
    N, theta_1, phi_1, theta_2, phi_2 = size.init()
    
    # Create a copy of the SDFG and specialize it with the specific N value
    sdfg_specialized = copy.deepcopy(sdfg)
    sdfg_specialized.specialize({'N': N})
    
    # Prepare baseline inputs (only the array inputs, not the symbol N)
    baseline_inputs = {
        'theta_1': theta_1.copy(),
        'phi_1': phi_1.copy(), 
        'theta_2': theta_2.copy(),
        'phi_2': phi_2.copy()
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
        report = analyzer.generate_report(results, f'arc_distance_{target_input}_sensitivity_report.txt')
        print(report)
        
        # Generate plots
        analyzer.plot_sensitivity_results(results, f'arc_distance_{target_input}_sensitivity_plots.png')
        
        return results, analyzer
        
    except Exception as e:
        print(f"Sensitivity analysis failed: {e}")
        return None, None

def run_quick_sensitivity_analysis():
    """
    Run a quick sensitivity analysis on all arc_distance inputs with default settings.
    """
    if not SENSITIVITY_AVAILABLE:
        print("Sensitivity analysis not available.")
        return
    
    inputs_to_test = ['theta_1', 'phi_1', 'theta_2', 'phi_2']
    
    print("Running quick sensitivity analysis on all arc_distance inputs...")
    print("=" * 60)
    
    for input_name in inputs_to_test:
        print(f"\nAnalyzing sensitivity to {input_name}...")
        run_sensitivity_analysis(ModelSize.M, input_name, 0.01, num_trials=1)
        print("-" * 40)


def run_grid_sensitivity_analysis():
    if not SENSITIVITY_AVAILABLE:
        print("Sensitivity analysis not available.")
        return

    # Sizes: provide init wrappers returning (specialize_symbols, baseline_inputs)
    def make_init_fn(ms: ModelSize):
        def _fn():
            N, theta_1, phi_1, theta_2, phi_2 = ms.init()
            specialize = {'N': N}
            baseline = {
                'theta_1': theta_1,
                'phi_1': phi_1,
                'theta_2': theta_2,
                'phi_2': phi_2,
            }
            return specialize, baseline
        return _fn

    sizes = [
        ('S', make_init_fn(ModelSize.S)),
        ('M', make_init_fn(ModelSize.M)),
        ('L', make_init_fn(ModelSize.L)),
        ('PAPER', make_init_fn(ModelSize.PAPER)),
    ]

    # Noise rows (multiplicative)
    noise_specs = [
        ('G-σ1', NoiseSpec(NoiseType.MULTIPLICATIVE, DistributionType.NORMAL, 0.01)),
        ('G-σ2', NoiseSpec(NoiseType.MULTIPLICATIVE, DistributionType.NORMAL, 0.05)),
        ('L-b1', NoiseSpec(NoiseType.MULTIPLICATIVE, DistributionType.LAPLACE, 0.007071)),
        ('L-b2', NoiseSpec(NoiseType.MULTIPLICATIVE, DistributionType.LAPLACE, 0.035355)),
    ]

    # Each input affects a single output array named 'result' in this SDFG
    for target in ['theta_1', 'phi_1', 'theta_2', 'phi_2']:
        plot_grid_sensitivity_over_sizes_rmse(
            base_sdfg=sdfg,
            sizes=sizes,
            noise_specs=noise_specs,
            target_input_name=target,
            save_path=f'arc_distance_grid_{target}.png',
            preferred_output_name='result',
            annot_fontsize=13,
        )

# Usage instructions
"""
SENSITIVITY ANALYSIS USAGE:

1. Run sensitivity analysis on a specific input:
   python arc_distance.py sensitivity <input_name> [noise_level]
   
   Examples:
   - python arc_distance.py sensitivity theta_1 0.01
   - python arc_distance.py sensitivity phi_1 0.001
   - python arc_distance.py sensitivity theta_2

2. Run sensitivity analysis on all inputs:
   python arc_distance.py sensitivity all

3. Run normal arc_distance model execution:
   python arc_distance.py

Available inputs: theta_1, phi_1, theta_2, phi_2
Default noise level: 0.01
Model size: S (100,000 points) for sensitivity analysis
"""

if __name__ == "__main__":
    import sys
    # Default to grid analysis for all inputs to create 4x4 scatter grids per input
    run_grid_sensitivity_analysis()
    
    # # Check if sensitivity analysis is requested
    # if len(sys.argv) > 1 and sys.argv[1] == "sensitivity":
    #     if len(sys.argv) > 2 and sys.argv[2] == "all":
    #         # Run sensitivity analysis on all inputs
    #         run_quick_sensitivity_analysis()
    #     else:
    #         # Run sensitivity analysis with optional parameters
    #         target_input = sys.argv[2] if len(sys.argv) > 2 else 'theta_1'
    #         noise_level = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
            
    #         print(f"Running sensitivity analysis on {target_input} with noise level {noise_level}")
    #         run_sensitivity_analysis(ModelSize.M, target_input, noise_level, num_trials=3)
    # else:
    #     # Run normal model execution
    #     run_all_models()

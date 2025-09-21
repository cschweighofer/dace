# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import dace
import numpy as np
import copy
from enum import Enum
from scipy import stats
from typing import Dict
from real_lowering import (
    SDFGTypeLowerer,
    InputSpec,
    analyze_sdfg_precision,
    analyze_sdfg_precision_across_samples,
    change_fptype,
    PrecisionType,
)

def analyze_sdfg_precision_with_model_sizes(sdfg, model_sizes, max_error_abs=None, max_error_rel=None):
    """Heat_3d specific implementation of precision analysis using model sizes as test cases."""
    
    # Ensure model_sizes are in the right format
    if isinstance(model_sizes[0], ModelSize):
        model_size_list = list(model_sizes)
    else:
        model_size_list = [ModelSize(size) if isinstance(size, tuple) else size for size in model_sizes]
    
    # For heat_3d with variable sizes, let's test each model size independently
    # and choose the most restrictive precision that works for all
    print(f"Testing {len(model_size_list)} model sizes individually...")
    
    # Build value lists across all sizes
    A_vals, B_vals, extras = [], [], []
    for i, model_size in enumerate(model_size_list):
        print(f"Preparing model size {model_size.name} ({i+1}/{len(model_size_list)})...")
        TSTEPS, N, A_init, B_init = model_size.init()
        A_vals.append(A_init)
        B_vals.append(B_init)
        extras.append({'TSTEPS': TSTEPS, 'N': N})

    input_specs = [
        InputSpec('A', value_list=A_vals),
        InputSpec('B', value_list=B_vals),
    ]

    # Delegate aggregation of most-restrictive precision to the core helper
    lowered_sdfg, agg_results = analyze_sdfg_precision_across_samples(
        copy.deepcopy(sdfg), input_specs,
        max_error_abs=max_error_abs, max_error_rel=max_error_rel,
        extra_args_list=extras,
    )
    # Annotate which sizes were tested
    agg_results['tested_model_sizes'] = [ms.name for ms in model_size_list]
    return lowered_sdfg, agg_results

sdfg = dace.SDFG.from_file("../sdfgs/heat_3d_auto_opt_cpu.sdfg")

def initialize(N, datatype=np.float64):
    A = np.fromfunction(lambda i, j, k: (i + j + (N - k)) * 10 / N, (N, N, N),
                        dtype=datatype)
    B = np.copy(A)

    return A, B

class ModelSize(Enum):
    S = (25, 25)
    M = (50, 40)
    L = (100, 70)
    # PAPER = (500, 120)

    def __init__(self, tsteps, n):
        self.tsteps = tsteps
        self.n = n

    def init(self):
        return self.tsteps, self.n, *initialize(self.n)

def run_precision_lowering_for_all_sizes():
    """Run precision lowering using all 4 model sizes as test cases."""
    # Set error bounds - you can adjust these based on your precision requirements
    max_error_abs = 1e-14   # Maximum absolute error
    max_error_rel = 1e-15   # Maximum relative error
    
    print(f"Applying precision lowering with error bounds: abs={max_error_abs}, rel={max_error_rel}")
    print("Using all 4 model sizes as test cases")
    
    try:
        # Use the precision lowering algorithm with model size based testing
        lowered_sdfg, analysis_results = analyze_sdfg_precision_with_model_sizes(
            sdfg=copy.deepcopy(sdfg),
            model_sizes=list(ModelSize),
            max_error_abs=max_error_abs,
            max_error_rel=max_error_rel
        )
        
        print(f"Selected precision type: {analysis_results.get('selected_type', 'None')}")
        
        # Print detailed error metrics
        if 'error_metrics' in analysis_results:
            for precision_type, metrics in analysis_results['error_metrics'].items():
                print(f"Error metrics for {precision_type}:")
                print(f"  Max absolute error: {metrics['max_absolute_error']:.2e}")
                print(f"  Max relative error: {metrics['max_relative_error']:.2e}")
        
        return lowered_sdfg, analysis_results
        
    except Exception as e:
        print(f"Error during precision lowering: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_model(size: ModelSize):
    """Run the model with the specified size configuration."""
    TSTEPS, N, A_initial, B_initial = size.init()

    # Run reference with original float64 precision
    A_ref, B_ref = np.copy(A_initial), np.copy(B_initial)
    sdfg(TSTEPS=TSTEPS, N=N, A=A_ref, B=B_ref)

    print(f"Model {size.name}: TSTEPS={TSTEPS}, N={N}")
    print(f"Initial A sum: {np.sum(A_initial):.2e}, B sum: {np.sum(B_initial):.2e}")
    print(f"Final A sum: {np.sum(A_ref):.2e}, B sum: {np.sum(B_ref):.2e}")
    return A_ref, B_ref

def run_all_models():
    """Run precision lowering analysis using all model sizes."""
    print("Running precision lowering analysis for all model sizes...")
    lowered_sdfg, analysis_results = run_precision_lowering_for_all_sizes()
    
    if lowered_sdfg is not None:
        print("\nNow testing individual models against MPF reference...")

        # Build MPF-typed reference SDFG
        ref_sdfg = copy.deepcopy(sdfg)
        change_fptype(ref_sdfg, dace.float64, PrecisionType.MPF.value)
        for model_size in ModelSize:
            print(f"\nTesting {model_size.name} model:")
            # Prepare initial conditions
            TSTEPS, N, A_init, B_init = model_size.init()
            A_mpf_ref, B_mpf_ref = np.copy(A_init), np.copy(B_init)
            # Execute MPF reference (underscore external arrays)
            ref_sdfg(TSTEPS=TSTEPS, N=N, _A=A_mpf_ref, _B=B_mpf_ref)
            
            # Test with the lowered precision SDFG
            A_lowered, B_lowered = np.copy(A_init), np.copy(B_init)
            
            # After precision lowering, arrays are renamed with underscore prefix
            lowered_sdfg(TSTEPS=TSTEPS, N=N, _A=A_lowered, _B=B_lowered)
            
            # Compare results
            abs_diff_A = np.abs(A_lowered - A_mpf_ref)
            abs_diff_B = np.abs(B_lowered - B_mpf_ref)
            max_diff = max(np.max(abs_diff_A), np.max(abs_diff_B))

            # Relative errors vs MPF
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_A = np.abs((A_lowered - A_mpf_ref) / np.where(A_mpf_ref != 0, A_mpf_ref, 1.0))
                rel_A = np.where(A_mpf_ref == 0, abs_diff_A, rel_A)
                rel_B = np.abs((B_lowered - B_mpf_ref) / np.where(B_mpf_ref != 0, B_mpf_ref, 1.0))
                rel_B = np.where(B_mpf_ref == 0, abs_diff_B, rel_B)
            max_rel = max(np.max(rel_A), np.max(rel_B))

            # Final run: only report these two lines
            print(f"  Max absolute error: {max_diff:.2e}")
            print(f"  Max relative error: {max_rel:.2e}")

if __name__ == "__main__":
    run_all_models()





"""
I want you to create pseudocode in python for the following description.
I have an SDFG in "Real" FP type which does not exist, it needs to be lowered to be one of the available types e.g. GNU MP Rational, MPFR 128bit, MPFR 256bit, FP64, FP32, FP16.
To do this the user provides the input intervals and the assumed distribution (taken from scipy) for the inputs and desired max error for the output.
For this I take the SDFG, generate random data, run the SDFG with GNU MP Rational and and then with the lowest available type going to higher precision until I am within the error bounds
"""

"""
SDFG Type Lowering Pseudocode
Lowers "Real" FP type SDFG to available precision types based on error bounds
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from scipy import stats
from enum import Enum
import dace
import copy

class PrecisionType(Enum):
    """Available precision types in order from lowest to highest"""
    # FP16 = dace.float16
    # FP32 = dace.float32
    FP64 = dace.float64
    # MPFR_128 = dace.float64  
    # MPFR_256 = dace.float64 
    GNU_MP_RATIONAL = dace.rational  # Using actual dace rational type

class InputSpec:
    """Specification for SDFG input"""
    def __init__(self, array_name: str, distribution: stats.rv_continuous):
        self.array_name = array_name
        self.distribution = distribution  # scipy distribution

class SDFGTypeLowerer:
    """Main class for lowering SDFG precision types"""

    def __init__(self, sdfg, input_specs: List[InputSpec], max_error_abs: float = None, max_error_rel: float = None, num_samples: int = 1000, extra_args: dict = None):
        """
        Args:
            sdfg: The original SDFG with "Real" type
            input_specs: List of InputSpec
            max_error_abs: Maximum allowed absolute error (optional)
            max_error_rel: Maximum allowed relative error (optional)
            num_samples: Number of samples for testing
        At least one of max_error_abs or max_error_rel must be specified.
        """
        if max_error_abs is None and max_error_rel is None:
            raise ValueError("At least one of max_error_abs or max_error_rel must be specified.")
        self.sdfg = sdfg
        self.input_specs = input_specs
        self.max_error_abs = max_error_abs
        self.max_error_rel = max_error_rel
        self.num_samples = num_samples

        self.extra_args = extra_args or {}

        # Precision types ordered from lowest to highest
        self.precision_hierarchy = [
            # PrecisionType.FP16,
            # PrecisionType.FP32,
            PrecisionType.FP64,
            # PrecisionType.MPFR_128,
            # PrecisionType.MPFR_256,
            PrecisionType.GNU_MP_RATIONAL
        ]

    def generate_test_data(self) -> Dict[str, np.ndarray]:
        """Generate random test data based on input specifications"""
        test_data = {}
        
        for input_spec in self.input_specs:
            # Get array shape from SDFG
            array_desc = self.sdfg.arrays[input_spec.array_name]
            shape = array_desc.shape
            
            # Generate samples using the specified scipy distribution
            samples = input_spec.distribution.rvs(size=(self.num_samples, *shape))
            test_data[input_spec.array_name] = samples.astype(np.float64)
        
        return test_data
    
    def create_typed_sdfg(self, precision_type: PrecisionType):
        """Create a copy of SDFG with specified precision type using change_fptype function"""
        # Deep copy the SDFG
        typed_sdfg = copy.deepcopy(self.sdfg)
        
        #print(f"DEBUG: Converting SDFG to {precision_type.name} (value: {precision_type.value})")
        #print(f"DEBUG: Original SDFG arrays before conversion: {[(name, arr.dtype) for name, arr in typed_sdfg.arrays.items()]}")
        
        # Use the change_fptype function to convert from float64 to target precision
        change_fptype(typed_sdfg, dace.float64, precision_type.value)
        
        #print(f"DEBUG: SDFG arrays after conversion: {[(name, arr.dtype) for name, arr in typed_sdfg.arrays.items()]}")
        
        return typed_sdfg

    def execute_sdfg(self, sdfg, test_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute SDFG with given test data and return results"""
        # Compile the SDFG
        compiled_sdfg = sdfg.compile()
        
        # Find all arrays that the SDFG expects as arguments
        sdfg_args = compiled_sdfg.argnames
        #print(f"DEBUG: SDFG expects arguments: {sdfg_args}")
        #print(f"DEBUG: Available arrays in SDFG: {[(name, arr.dtype, arr.transient) for name, arr in sdfg.arrays.items()]}")
        
        # Check which arrays are transient vs non-transient
        transient_arrays = {name: arr for name, arr in sdfg.arrays.items() if arr.transient}
        non_transient_arrays = {name: arr for name, arr in sdfg.arrays.items() if not arr.transient}
        #print(f"DEBUG: Transient arrays: {[(name, arr.dtype) for name, arr in transient_arrays.items()]}")
        #print(f"DEBUG: Non-transient arrays: {[(name, arr.dtype) for name, arr in non_transient_arrays.items()]}")
        
        # The change_fptype function creates underscore-prefixed external arrays
        # We need to use these for the actual SDFG execution
        input_names = {spec.array_name for spec in self.input_specs}

        # Only treat input arrays as outputs (in-place semantics)
        external_input_mapping = {}
        for arr_name, arr in non_transient_arrays.items():
            if arr_name.startswith('_'):
                original_name = arr_name[1:]  # Remove underscore
                if original_name in input_names:
                    external_input_mapping[original_name] = arr_name

        # Only collect outputs for input arrays (in-place arrays)
        external_output_arrays = {name: (external_input_mapping[name], non_transient_arrays[external_input_mapping[name]]) for name in input_names}

        # This ensures that the output of the rational SDFG and the lowered SDFG are compared in the same way as the original SDFG usage (run_S).
        
        # Execute for each sample
        all_outputs = {name: [] for name in external_output_arrays.keys()}
        
        for sample_idx in range(self.num_samples):
            # Prepare all arguments using the external array names
            all_args = {}
            
            # Add input data using underscore-prefixed names
            for input_spec in self.input_specs:
                external_name = external_input_mapping[input_spec.array_name]
                # Make a copy to avoid numpy view issues
                sample_data = np.copy(test_data[input_spec.array_name][sample_idx])
                all_args[external_name] = sample_data
                #print(f"DEBUG: Input '{external_name}' dtype: {sample_data.dtype}, sample val: {sample_data.flat[0]}")
            
            # Add output arrays using underscore-prefixed names
            for original_name, (external_name, arr_desc) in external_output_arrays.items():
                # External interface always uses the original numpy-compatible types
                # Resolve symbolic shapes using extra_args
                resolved_shape = []
                for dim in arr_desc.shape:
                    if hasattr(dim, 'name') and dim.name in self.extra_args:
                        resolved_shape.append(self.extra_args[dim.name])
                    elif isinstance(dim, str) and dim in self.extra_args:
                        resolved_shape.append(self.extra_args[dim])
                    elif isinstance(dim, (int, np.integer)):
                        resolved_shape.append(int(dim))
                    else:
                        try:
                            resolved_shape.append(int(dim))
                        except Exception:
                            raise ValueError(f"Cannot resolve symbolic dimension '{dim}' for array '{original_name}'")
                all_args[external_name] = np.zeros(tuple(resolved_shape), dtype=arr_desc.dtype.as_numpy_dtype())
                #print(f"DEBUG: Output '{external_name}' initialized with dtype: {all_args[external_name].dtype}")
            
            # Add extra (non-array) arguments if needed
            all_args.update(self.extra_args)
            # Execute
            try:
                compiled_sdfg(**all_args)
            except Exception as e:
                print(f"  Error: {str(e)}")
            
            # Store results using original names
            for original_name, (external_name, _) in external_output_arrays.items():
                result = all_args[external_name].copy()
                print(f"DEBUG: Output '{original_name}' value: {result}, dtype: {result.dtype}")
                all_outputs[original_name].append(result)
                #print(f"DEBUG: After execution, output '{external_name}' value: {result.flat[0]}, dtype: {result.dtype}")
        
        # Convert to numpy arrays
        for out_name in all_outputs:
            all_outputs[out_name] = np.array(all_outputs[out_name])
            #print(f"DEBUG: Final output '{out_name}' dtype: {all_outputs[out_name].dtype}, shape: {all_outputs[out_name].shape}")
            if len(all_outputs[out_name].flat) > 0:
                sample_val = all_outputs[out_name].flat[0]
                #print(f"DEBUG: Sample output value: {sample_val} (type: {type(sample_val)})")
        
        return all_outputs
    
    def compute_error_metrics(self, reference_results: Dict[str, np.ndarray],
                            test_results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute various error metrics between reference and test results
        
        The reference results should be computed using dace.rational type for exact arithmetic,
        while test results use the precision type being evaluated.
        """
        
        all_max_abs_errors = []
        all_mean_abs_errors = []
        all_max_rel_errors = []
        all_mean_rel_errors = []
        
        for output_name in reference_results.keys():
            if output_name not in test_results:
                raise ValueError(f"Output {output_name} not found in test results")
            
            ref = reference_results[output_name]
            test = test_results[output_name]
            
            # Handle potential shape mismatches
            if ref.shape != test.shape:
                raise ValueError(f"Result shapes don't match for {output_name}: {ref.shape} vs {test.shape}")
            
            # Convert reference to rational for exact computation, test to float64
            # The reference should already be computed with rational type
            # Convert test to float64 for compatibility with numpy operations
            test_float = test.astype(np.float64)
            
            # For rational reference, we need to convert to float64 for numpy operations
            # but we maintain higher precision by using the rational computation
            # Note: Even when using rational internally, the output interface is float64
            # but the computation was done with higher precision rational arithmetic
            if hasattr(ref.flat[0], 'get_value'):  # Check if it's a rational type
                # Reference is rational - convert to high precision for computation
                ref_float = np.array([r.get_value() if hasattr(r, 'get_value') else float(r) for r in ref.flat]).reshape(ref.shape)
                #print(f"DEBUG: Reference uses rational type internally, converted to float64 for comparison")
            else:
                # Reference uses float64 interface but rational computation was done internally
                ref_float = ref.astype(np.float64)
                #print(f"DEBUG: Reference computed with rational arithmetic, interface type: {ref.dtype}")
            
            #print(f"DEBUG: Test result dtype: {test.dtype}, Reference dtype after conversion: {ref_float.dtype}")
            #print(f"DEBUG: Sample values - Test: {test_float.flat[0]:.10f}, Ref: {ref_float.flat[0]:.10f}")
            
            # Compute absolute errors using rational precision when possible
            abs_errors = np.abs(ref_float - test_float)
            max_abs_error = np.max(abs_errors)
            mean_abs_error = np.mean(abs_errors)
            
            # Compute relative errors (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_errors = np.abs((ref_float - test_float) / np.where(ref_float != 0, ref_float, 1.0))
                rel_errors = np.where(ref_float == 0, abs_errors, rel_errors)
            
            max_rel_error = np.max(rel_errors)
            mean_rel_error = np.mean(rel_errors)
            
            all_max_abs_errors.append(max_abs_error)
            all_mean_abs_errors.append(mean_abs_error)
            all_max_rel_errors.append(max_rel_error)
            all_mean_rel_errors.append(mean_rel_error)
        
        return {
            'max_absolute_error': max(all_max_abs_errors),
            'mean_absolute_error': np.mean(all_mean_abs_errors),
            'max_relative_error': max(all_max_rel_errors),
            'mean_relative_error': np.mean(all_mean_rel_errors)
        }

    def check_error_bounds(self, error_metrics: Dict[str, float]) -> bool:
        """Check if error metrics satisfy the specified bounds (absolute and/or relative)"""
        abs_ok = True
        rel_ok = True
        if self.max_error_abs is not None:
            abs_ok = error_metrics['max_absolute_error'] <= self.max_error_abs
        if self.max_error_rel is not None:
            rel_ok = error_metrics['max_relative_error'] <= self.max_error_rel
        return abs_ok and rel_ok

    def lower_precision(self) -> Tuple[PrecisionType, Dict[str, Any]]:
        """
        Main method to find the lowest precision type that meets error bounds (absolute and/or relative)
        Returns: (selected_precision_type, analysis_results)
        """
        print("Generating test data...")
        test_data = self.generate_test_data()

        print("Computing reference results with GNU MP Rational...")
        reference_sdfg = self.create_typed_sdfg(PrecisionType.GNU_MP_RATIONAL)
        reference_results = self.execute_sdfg(reference_sdfg, test_data)

        analysis_results = {
            'tested_types': [],
            'error_metrics': {},
            'selected_type': None,
            'test_data_size': self.num_samples
            ,'max_error_abs': self.max_error_abs
            ,'max_error_rel': self.max_error_rel
        }

        print("Testing precision types from lowest to highest...")

        # Test from lowest precision to highest (excluding GNU_MP_RATIONAL)
        test_types = self.precision_hierarchy[:-1]  # Exclude reference type

        for precision_type in test_types:
            print(f"Testing {precision_type.name}...")

            try:
                # Create SDFG with this precision type
                typed_sdfg = self.create_typed_sdfg(precision_type)

                # Execute with test data
                test_results = self.execute_sdfg(typed_sdfg, test_data)

                # Compute error metrics
                error_metrics = self.compute_error_metrics(reference_results, test_results)

                # Store results
                analysis_results['tested_types'].append(precision_type.name)
                analysis_results['error_metrics'][precision_type.name] = error_metrics

                print(f"  Max absolute error: {error_metrics['max_absolute_error']:.2e}")
                print(f"  Max relative error: {error_metrics['max_relative_error']:.2e}")

                # Check if this precision meets our requirements
                if self.check_error_bounds(error_metrics):
                    print(f"✓ {precision_type.name} meets error bounds!")
                    analysis_results['selected_type'] = precision_type.name
                    return precision_type, analysis_results
                else:
                    print(f"✗ {precision_type.name} exceeds error bounds")
                    
            except Exception as e:
                print(f"⚠ Error testing {precision_type.name}: {e}")
                continue

        # If no type meets the requirements, use highest precision
        highest_precision = test_types[-1]  # MPFR_256
        print(f"⚠ No type meets error bounds, using {highest_precision.name}")
        analysis_results['selected_type'] = highest_precision.name

        return highest_precision, analysis_results

    def generate_lowered_sdfg(self) -> Tuple[Any, Dict[str, Any]]:
        """Return the type lowered SDFG"""
        selected_type, analysis_results = self.lower_precision()
        print(f"Selected precision type: {selected_type.name}")
        final_sdfg = self.create_typed_sdfg(selected_type)
        print(f"Final SDFG generated successfully")
        return final_sdfg, analysis_results

@dace.program
def test_program(A: dace.float64[10, 10], B: dace.float64[10, 10], C: dace.float64[10, 10]):
    C[:] = A[:] + B[:]

@dace.program  
def test_program2(A: dace.float64[10, 10], B: dace.float64[10, 10], C: dace.float64[10, 10]):
    C[:] = (A * 1000.0) / (B + 0.001) - (A / B) * 999.0

def change_fptype(sdfg: dace.SDFG, src_fptype: dace.dtypes.typeclass, dace_internal_fptype: dace.dtypes.typeclass):
    # Add simulated double transient arrays to the SDFG
    arrays_that_become_transient = set()
    for arr_name, arr in sdfg.arrays.items():
        if arr.dtype == src_fptype:
            arr.dtype = dace_internal_fptype
            if not arr.transient:
                arr.transient = True
                arrays_that_become_transient.add((arr_name, arr))

    # For all arrays that we made transient, we need to ensure we add the appropriate copy-in and copy-out nodes
    copy_in_state = sdfg.add_state_before(state=sdfg.start_block, label="copy_in")
    last_block = [node for node in sdfg.nodes() if sdfg.out_degree(node) == 0][0] # Only last block has no successors
    copy_out_state = sdfg.add_state_after(state=last_block, label="copy_out")

    # Add a new array descriptor for each transient array, and add copy-in or copy-out
    src_dst_paris = set()
    for arr_name, arr in arrays_that_become_transient:
        nontransient_arr_desc = copy.deepcopy(arr)
        nontransient_arr_desc.transient = False
        nontransient_arr_desc.dtype = src_fptype # Change back to float64 for the copy
        sdfg.add_datadesc(name="_" + arr_name, datadesc=nontransient_arr_desc, find_new_name=False)
        src_dst_paris.add(((arr_name, arr), ("_" + arr_name, nontransient_arr_desc)))

    def add_copy_map(state: dace.SDFGState, src_arr_name:str, src_arr:dace.data.Data, dst_arr_name:str, dst_arr:dace.data.Data):
        """
        Add a copy map to the given state in the SDFG.
        """
        assert src_arr.shape == dst_arr.shape, "Source and destination arrays must have the same shape."
        # Create a new map node
        map_ranges = dict()
        for dim, size in enumerate(src_arr.shape):
            map_ranges[f"i{dim}"] = f"0:{size}"

        map_entry, map_exit = state.add_map(name=f"copy_map_{src_arr_name}_to_{dst_arr_name}", ndrange=map_ranges)

        # Add access nodes for source and destination arrays
        src_access = state.add_access(src_arr_name)
        dst_access = state.add_access(dst_arr_name)

        # Add edges from the map to the access nodes, care about the connector
        state.add_edge(src_access, None, map_entry, f"IN_{src_arr_name}", dace.memlet.Memlet.from_array(src_arr_name, src_arr))
        state.add_edge(map_exit, f"OUT_{dst_arr_name}", dst_access, None, dace.memlet.Memlet.from_array(dst_arr_name, dst_arr))
        map_entry.add_in_connector(f"IN_{src_arr_name}")
        map_entry.add_out_connector(f"OUT_{src_arr_name}")
        map_exit.add_in_connector(f"IN_{dst_arr_name}")
        map_exit.add_out_connector(f"OUT_{dst_arr_name}")

        # Add a tasklet that perfmorms the type cast
        tasklet = state.add_tasklet(
            name=f"copy_{src_arr_name}_to_{dst_arr_name}",
            inputs={"in"},
            outputs={"out"},
            code=f"out = static_cast<{dst_arr.dtype.ctype}>(in);",
            language=dace.Language.CPP)

        access_str = f", ".join([str(s) for s in map_ranges.keys()])
        state.add_edge(map_entry, f"OUT_{src_arr_name}", tasklet, "in", dace.Memlet(expr=f"{src_arr_name}[{access_str}]"))
        state.add_edge(tasklet, "out", map_exit, f"IN_{dst_arr_name}", dace.Memlet(expr=f"{dst_arr_name}[{access_str}]"))


    for (transient_arr_name, transient_arr), (nontransient_arr_name, nontransient_arr) in src_dst_paris:
        add_copy_map(state=copy_in_state,
                    src_arr_name=nontransient_arr_name,
                    src_arr=nontransient_arr,
                    dst_arr_name=transient_arr_name,
                    dst_arr=transient_arr)
        add_copy_map(state=copy_out_state,
                    src_arr_name=transient_arr_name,
                    src_arr=transient_arr,
                    dst_arr_name=nontransient_arr_name,
                    dst_arr=nontransient_arr)    

def change_fptype2(sdfg: dace.SDFG, src_fptype: dace.dtypes.typeclass, dace_internal_fptype: dace.dtypes.typeclass):
    # Add simulated double transient arrays to the SDFG
    arrays_that_become_transient = set()
    for arr_name, arr in sdfg.arrays.items():
        if arr.dtype == src_fptype:
            arr.dtype = dace_internal_fptype
            if not arr.transient:
                arr.transient = True
                arrays_that_become_transient.add((arr_name, arr))

    # For all arrays that we made transient, we need to ensure we add the appropriate copy-in and copy-out nodes
    copy_in_state = sdfg.add_state_before(state=sdfg.start_block, label="copy_in")
    last_block = [node for node in sdfg.nodes() if sdfg.out_degree(node) == 0][0] # Only last block has no successors
    copy_out_state = sdfg.add_state_after(state=last_block, label="copy_out")

    # Add a new array descriptor for each transient array, and add copy-in or copy-out
    src_dst_pairs = set()
    for arr_name, arr in arrays_that_become_transient:
        nontransient_arr_desc = copy.deepcopy(arr)
        nontransient_arr_desc.transient = False
        
        # For rational types, we still use float64 external interface
        # The internal computation will use rational, but we convert at boundaries
        nontransient_arr_desc.dtype = src_fptype  # Always use original type for external interface
            
        sdfg.add_datadesc(name="_" + arr_name, datadesc=nontransient_arr_desc, find_new_name=False)
        src_dst_pairs.add(((arr_name, arr), ("_" + arr_name, nontransient_arr_desc)))

    def add_copy_map(state: dace.SDFGState, src_arr_name:str, src_arr:dace.data.Data, dst_arr_name:str, dst_arr:dace.data.Data):
        """
        Add a copy map to the given state in the SDFG.
        """
        assert src_arr.shape == dst_arr.shape, "Source and destination arrays must have the same shape."
        # Create a new map node
        map_ranges = dict()
        for dim, size in enumerate(src_arr.shape):
            map_ranges[f"i{dim}"] = f"0:{size}"

        map_entry, map_exit = state.add_map(name=f"copy_map_{src_arr_name}_to_{dst_arr_name}", ndrange=map_ranges)

        # Add access nodes for source and destination arrays
        src_access = state.add_access(src_arr_name)
        dst_access = state.add_access(dst_arr_name)

        # Add edges from the map to the access nodes, care about the connector
        state.add_edge(src_access, None, map_entry, f"IN_{src_arr_name}", dace.memlet.Memlet.from_array(src_arr_name, src_arr))
        state.add_edge(map_exit, f"OUT_{dst_arr_name}", dst_access, None, dace.memlet.Memlet.from_array(dst_arr_name, dst_arr))
        map_entry.add_in_connector(f"IN_{src_arr_name}")
        map_entry.add_out_connector(f"OUT_{src_arr_name}")
        map_exit.add_in_connector(f"IN_{dst_arr_name}")
        map_exit.add_out_connector(f"OUT_{dst_arr_name}")

        # Add a tasklet that performs the type cast
        tasklet = state.add_tasklet(
            name=f"copy_{src_arr_name}_to_{dst_arr_name}",
            inputs={"in"},
            outputs={"out"},
            code=f"out = static_cast<{dst_arr.dtype.ctype}>(in);",
            language=dace.Language.CPP)

        access_str = f", ".join([str(s) for s in map_ranges.keys()])
        state.add_edge(map_entry, f"OUT_{src_arr_name}", tasklet, "in", dace.Memlet(expr=f"{src_arr_name}[{access_str}]"))
        state.add_edge(tasklet, "out", map_exit, f"IN_{dst_arr_name}", dace.Memlet(expr=f"{dst_arr_name}[{access_str}]"))

    for (transient_arr_name, transient_arr), (nontransient_arr_name, nontransient_arr) in src_dst_pairs:
        add_copy_map(state=copy_in_state,
                    src_arr_name=nontransient_arr_name,
                    src_arr=nontransient_arr,
                    dst_arr_name=transient_arr_name,
                    dst_arr=transient_arr)
        add_copy_map(state=copy_out_state,
                    src_arr_name=transient_arr_name,
                    src_arr=transient_arr,
                    dst_arr_name=nontransient_arr_name,
                    dst_arr=nontransient_arr)

def analyze_sdfg_precision(
    sdfg, input_specs, max_error_abs=None, max_error_rel=None, num_samples=1000, extra_args=None
):
    """Convenience function to run SDFG precision analysis and return the lowered SDFG and results."""
    lowerer = SDFGTypeLowerer(
        sdfg, input_specs, max_error_abs=max_error_abs, max_error_rel=max_error_rel, num_samples=num_samples, extra_args=extra_args
    )
    return lowerer.generate_lowered_sdfg()

if __name__ == "__main__":
    # Example usage - get SDFG without arrays
    sdfg = test_program.to_sdfg()
    
    input_specs = [
        # Using larger numbers and a different range to amplify precision differences
        InputSpec('A', stats.uniform(1000.0, 9000.0)),
        InputSpec('B', stats.uniform(0.001, 0.099))
    ]
    
    # Example: Only absolute error bound
    lowerer = SDFGTypeLowerer(sdfg, input_specs, max_error_abs=1e-5, num_samples=1000)
    # Example: Only relative error bound (uncomment to use)
    # lowerer = SDFGTypeLowerer(sdfg, input_specs, max_error_rel=1e-3, num_samples=1000)
    # Example: Both error bounds (uncomment to use)
    # lowerer = SDFGTypeLowerer(sdfg, input_specs, max_error_abs=1e-5, max_error_rel=1e-3, num_samples=1000)
    
    try:
        # Generate the final lowered SDFG (this includes the analysis)
        final_sdfg, analysis_results = lowerer.generate_lowered_sdfg()
        print("Selected precision type:", analysis_results.get('selected_type', 'None'))
        # print("Analysis results:", analysis_results)
        # print("Final SDFG generated successfully")
        
    except Exception as e:
        print(f"Error during precision lowering: {e}")
        import traceback
        traceback.print_exc()
"""
SDFG Type Lowering Pseudocode
Lowers "Real" FP type SDFG to available precision types based on error bounds
i.e. increase precision until error bounds are met
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from scipy import stats
from enum import Enum
import dace
import copy

class PrecisionType(Enum):
    """Available precision types in order from lowest to highest"""
    FP32 = dace.float32
    SD = dace.simulated_double
    FP64 = dace.float64
    MPF = dace.mpf  # GNU MPFR

class InputSpec:
    """Specification for SDFG input"""
    def __init__(self, array_name: str, distribution: stats.rv_continuous = None, value_list: List[np.ndarray] = None):
        """
        Args:
            array_name: Name of the array input
            distribution: scipy distribution for random sampling (optional)
            value_list: List of explicit array values to test (optional)
        
        Exactly one of distribution or value_list must be provided.
        """
        if (distribution is None) == (value_list is None):
            raise ValueError("Exactly one of distribution or value_list must be provided")
        
        self.array_name = array_name
        self.distribution = distribution  # scipy distribution
        self.value_list = value_list  # List of explicit numpy arrays

class SDFGTypeLowerer:
    """Main class for lowering SDFG precision types"""

    def __init__(self, sdfg, input_specs: List[InputSpec], max_error_abs: float = None, max_error_rel: float = None, num_samples: int = 1000, extra_args: dict = None, extra_args_list: List[dict] = None):
        """
        Args:
            sdfg: The original SDFG with "Real" type
            input_specs: List of InputSpec
            max_error_abs: Maximum allowed absolute error (optional)
            max_error_rel: Maximum allowed relative error (optional)
            num_samples: Number of samples for testing
            extra_args: Single set of extra arguments for all samples (optional)
            extra_args_list: List of extra arguments per sample (optional)
        At least one of max_error_abs or max_error_rel must be specified.
        Either extra_args or extra_args_list should be provided, not both.
        """
        if max_error_abs is None and max_error_rel is None:
            raise ValueError("At least one of max_error_abs or max_error_rel must be specified.")
        
        if extra_args is not None and extra_args_list is not None:
            raise ValueError("Provide either extra_args or extra_args_list, not both.")
        
        self.sdfg = sdfg
        self.input_specs = input_specs
        self.max_error_abs = max_error_abs
        self.max_error_rel = max_error_rel
        self.num_samples = num_samples
        self.extra_args = extra_args or {}
        self.extra_args_list = extra_args_list or []

        # Precision types ordered from lowest to highest
        self.precision_hierarchy = [
            PrecisionType.FP32,
            PrecisionType.SD,
            PrecisionType.FP64,
            PrecisionType.MPF
        ]

    def generate_test_data(self) -> Dict[str, np.ndarray]:
        """Generate test data based on input specifications (either distributions or explicit values)"""
        test_data = {}
        
        # Determine the number of samples needed
        num_samples = self.num_samples
        for input_spec in self.input_specs:
            if input_spec.value_list is not None:
                # For explicit value lists, use the number of values provided
                num_samples = len(input_spec.value_list)
                break
        
        # Update extra_args_list if using value_list but no extra_args_list provided
        if not self.extra_args_list and any(spec.value_list is not None for spec in self.input_specs):
            # Use the same extra_args for all samples
            self.extra_args_list = [self.extra_args] * num_samples
        
        for input_spec in self.input_specs:
            if input_spec.distribution is not None:
                # Generate random samples using distribution
                array_desc = self.sdfg.arrays[input_spec.array_name]
                shape = array_desc.shape
                
                # Use the first extra_args for shape resolution (or self.extra_args if no list)
                current_extra_args = self.extra_args_list[0] if self.extra_args_list else self.extra_args
                
                # Resolve symbolic shapes using extra_args
                resolved_shape = []
                for dim in shape:
                    if hasattr(dim, 'name') and dim.name in current_extra_args:
                        resolved_shape.append(current_extra_args[dim.name])
                    elif isinstance(dim, str) and dim in current_extra_args:
                        resolved_shape.append(current_extra_args[dim])
                    elif isinstance(dim, (int, np.integer)):
                        resolved_shape.append(int(dim))
                    else:
                        try:
                            resolved_shape.append(int(dim))
                        except Exception:
                            raise ValueError(f"Cannot resolve symbolic dimension '{dim}' for array '{input_spec.array_name}'. Make sure to provide it in extra_args.")
                
                # Generate samples using the specified scipy distribution
                samples = input_spec.distribution.rvs(size=(num_samples, *tuple(resolved_shape)))
                test_data[input_spec.array_name] = samples.astype(np.float64)
                
            elif input_spec.value_list is not None:
                # Use explicit values provided - store as list since shapes may vary
                test_data[input_spec.array_name] = input_spec.value_list
        
        # Update num_samples for consistency
        self.num_samples = num_samples
        
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
            
            # Get the appropriate extra_args for this sample
            current_extra_args = self.extra_args_list[sample_idx] if self.extra_args_list else self.extra_args
            
            # Add input data using underscore-prefixed names
            for input_spec in self.input_specs:
                external_name = external_input_mapping[input_spec.array_name]
                # Make a copy to avoid numpy view issues
                # Handle both numpy arrays and lists
                if isinstance(test_data[input_spec.array_name], list):
                    sample_data = np.copy(test_data[input_spec.array_name][sample_idx])
                else:
                    sample_data = np.copy(test_data[input_spec.array_name][sample_idx])
                all_args[external_name] = sample_data
                #print(f"DEBUG: Input '{external_name}' dtype: {sample_data.dtype}, sample val: {sample_data.flat[0]}")
            
            # Add output arrays using underscore-prefixed names
            for original_name, (external_name, arr_desc) in external_output_arrays.items():
                # If this external name is already provided as an input (in-place),
                # don't overwrite it with zeros. Only allocate when it's not present.
                if external_name in all_args:
                    continue
                # External interface always uses the original numpy-compatible types
                # Resolve symbolic shapes using current_extra_args
                resolved_shape = []
                for dim in arr_desc.shape:
                    if hasattr(dim, 'name') and dim.name in current_extra_args:
                        resolved_shape.append(current_extra_args[dim.name])
                    elif isinstance(dim, str) and dim in current_extra_args:
                        resolved_shape.append(current_extra_args[dim])
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
            all_args.update(current_extra_args)
            # Execute
            try:
                compiled_sdfg(**all_args)
            except Exception as e:
                print(f"  Error: {str(e)}")
            
            # Store results using original names
            for original_name, (external_name, _) in external_output_arrays.items():
                result = all_args[external_name].copy()
                # print(f"DEBUG: Output '{original_name}' value: {result}, dtype: {result.dtype}")
                all_outputs[original_name].append(result)
                #print(f"DEBUG: After execution, output '{external_name}' value: {result.flat[0]}, dtype: {result.dtype}")
        
        # Convert to numpy arrays or keep as lists depending on the data type
        for out_name in all_outputs:
            # Check if all outputs have the same shape
            if len(all_outputs[out_name]) > 0:
                first_shape = all_outputs[out_name][0].shape
                same_shape = all(arr.shape == first_shape for arr in all_outputs[out_name])
                
                if same_shape:
                    # Convert to numpy array if shapes are consistent
                    all_outputs[out_name] = np.array(all_outputs[out_name])
                else:
                    # Keep as list if shapes vary
                    pass
            
            #print(f"DEBUG: Final output '{out_name}' type: {type(all_outputs[out_name])}")
            if isinstance(all_outputs[out_name], np.ndarray) and len(all_outputs[out_name].flat) > 0:
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
            
            # Handle both numpy arrays and lists
            if isinstance(ref, list) and isinstance(test, list):
                # Both are lists (variable shapes)
                if len(ref) != len(test):
                    raise ValueError(f"Number of samples don't match for {output_name}: {len(ref)} vs {len(test)}")
                
                for i in range(len(ref)):
                    ref_sample = ref[i]
                    test_sample = test[i]
                    
                    if ref_sample.shape != test_sample.shape:
                        raise ValueError(f"Sample {i} shapes don't match for {output_name}: {ref_sample.shape} vs {test_sample.shape}")
                    
                    # Process this sample
                    self._process_sample_errors(ref_sample, test_sample, all_max_abs_errors, all_mean_abs_errors, all_max_rel_errors, all_mean_rel_errors)
                    
            else:
                # At least one is a numpy array (consistent shapes)
                if isinstance(ref, list):
                    ref = np.array(ref)
                if isinstance(test, list):
                    test = np.array(test)
                
                # Handle potential shape mismatches
                if ref.shape != test.shape:
                    raise ValueError(f"Result shapes don't match for {output_name}: {ref.shape} vs {test.shape}")
                
                # Process all samples at once
                for i in range(ref.shape[0]):
                    self._process_sample_errors(ref[i], test[i], all_max_abs_errors, all_mean_abs_errors, all_max_rel_errors, all_mean_rel_errors)
        
        return {
            'max_absolute_error': max(all_max_abs_errors) if all_max_abs_errors else 0.0,
            'mean_absolute_error': np.mean(all_mean_abs_errors) if all_mean_abs_errors else 0.0,
            'max_relative_error': max(all_max_rel_errors) if all_max_rel_errors else 0.0,
            'mean_relative_error': np.mean(all_mean_rel_errors) if all_mean_rel_errors else 0.0
        }
    
    def _process_sample_errors(self, ref_sample, test_sample, all_max_abs_errors, all_mean_abs_errors, all_max_rel_errors, all_mean_rel_errors):
        """Process error metrics for a single sample pair"""
        # Convert reference to rational for exact computation, test to float64
        # The reference should already be computed with rational type
        # Convert test to float64 for compatibility with numpy operations
        test_float = test_sample.astype(np.float64)
        
        # For rational reference, we need to convert to float64 for numpy operations
        # but we maintain higher precision by using the rational computation
        # Note: Even when using rational internally, the output interface is float64
        # but the computation was done with higher precision rational arithmetic
        if hasattr(ref_sample.flat[0], 'get_value'):  # Check if it's a rational type
            # Reference is rational - convert to high precision for computation
            ref_float = np.array([r.get_value() if hasattr(r, 'get_value') else float(r) for r in ref_sample.flat]).reshape(ref_sample.shape)
            #print(f"DEBUG: Reference uses rational type internally, converted to float64 for comparison")
        else:
            # Reference uses float64 interface but rational computation was done internally
            ref_float = ref_sample.astype(np.float64)
            #print(f"DEBUG: Reference computed with rational arithmetic, interface type: {ref_sample.dtype}")
        
        #print(f"DEBUG: Test result dtype: {test_sample.dtype}, Reference dtype after conversion: {ref_float.dtype}")
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

        print("Computing reference results with GNU MP...")
        reference_sdfg = self.create_typed_sdfg(PrecisionType.MPF)
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

        # Test from lowest precision to highest (excluding MPF which is our reference)
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

def change_fptype(sdfg: dace.SDFG, src_fptype: dace.dtypes.typeclass, dace_internal_fptype: dace.dtypes.typeclass):
    # Add simulated double transient arrays to the SDFG
    arrays_that_become_transient = set()
    for arr_name, arr in sdfg.arrays.items():
        # print("Name: ", arr_name, "Type: ", arr.dtype, "Transient: ", arr.transient)
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
    
    def _enforce_conn_types(state: dace.SDFGState, node: dace.nodes.NestedSDFG | dace.nodes.Tasklet, dtype):
        for in_conn in node.in_connectors:
            for ie in state.in_edges_by_connector(node, in_conn):
                if state.sdfg.arrays[ie.data.data].dtype ==  dtype and node.in_connectors[in_conn] != dtype:
                    node.in_connectors[in_conn] = dtype
        for out_conn in node.out_connectors:
            for oe in state.out_edges_by_connector(node, out_conn):
                if state.sdfg.arrays[oe.data.data].dtype ==  dtype and node.out_connectors[out_conn] != dtype:
                    node.out_connectors[out_conn] = dtype

    def fix_connector_types(sdfg: dace.SDFG, dtype):
        for state in sdfg.all_states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    # Recursively fix connector types in nested SDFGs
                    _enforce_conn_types(state, node, dtype)
                    fix_connector_types(node.sdfg, dtype)
                elif isinstance(node, dace.nodes.Tasklet):
                    _enforce_conn_types(state, node, dtype)

    fix_connector_types(sdfg, dace_internal_fptype)

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


@dace.program
def test_program(A: dace.float64[10, 10], B: dace.float64[10, 10], C: dace.float64[10, 10]):
    C[:] = A[:] + B[:]

def analyze_sdfg_precision(
    sdfg, input_specs, max_error_abs=None, max_error_rel=None, num_samples=1000, extra_args=None, extra_args_list=None
):
    """Convenience function to run SDFG precision analysis and return the lowered SDFG and results.

    This runs all samples together. If your samples have variable shapes and you
    experience issues with reference execution (e.g., MPF backend), consider
    using analyze_sdfg_precision_across_samples instead.
    """
    lowerer = SDFGTypeLowerer(
        sdfg, input_specs, max_error_abs=max_error_abs, max_error_rel=max_error_rel, 
        num_samples=num_samples, extra_args=extra_args, extra_args_list=extra_args_list
    )
    return lowerer.generate_lowered_sdfg()

def analyze_sdfg_precision_across_samples(
    sdfg, input_specs, max_error_abs=None, max_error_rel=None, extra_args_list=None
):
    """Robust precision analysis across samples with possibly different shapes.

    Runs precision analysis per sample (num_samples is inferred from the first
    value_list) and aggregates the most restrictive precision that satisfies
    the error bounds across all samples. Returns a typed SDFG created for the
    most restrictive precision and an aggregated analysis results dictionary.
    """
    # Infer number of samples from input_specs value_list
    if not input_specs:
        raise ValueError("input_specs must be a non-empty list")
    first = input_specs[0]
    if first.value_list is None:
        raise ValueError("across_samples requires value_list-based input_specs")
    num_samples = len(first.value_list)

    # Basic validation: all specs with value_list must have same length
    for spec in input_specs:
        if spec.value_list is not None and len(spec.value_list) != num_samples:
            raise ValueError("All value_list entries must have the same length")

    # Helper mapping
    name_to_enum = { 'FP32': PrecisionType.FP32, 'SD': PrecisionType.SD, 'FP64': PrecisionType.FP64, 'MPF': PrecisionType.MPF }
    enum_to_name = { v: k for k, v in name_to_enum.items() }
    order = [PrecisionType.FP32, PrecisionType.SD, PrecisionType.FP64, PrecisionType.MPF]

    per_sample = []
    highest_needed = PrecisionType.FP32

    for i in range(num_samples):
        # Build per-sample specs
        per_specs = []
        for spec in input_specs:
            if spec.value_list is None:
                raise ValueError("across_samples requires value_list-based input_specs for all inputs")
            per_specs.append(InputSpec(spec.array_name, value_list=[spec.value_list[i]]))

        extra_args = (extra_args_list[i] if extra_args_list and i < len(extra_args_list) else None)

        lowerer = SDFGTypeLowerer(
            copy.deepcopy(sdfg), per_specs,
            max_error_abs=max_error_abs, max_error_rel=max_error_rel,
            num_samples=1, extra_args=extra_args
        )
        _, ar = lowerer.generate_lowered_sdfg()
        sel_name = ar.get('selected_type', 'FP32')
        sel_enum = name_to_enum.get(sel_name, PrecisionType.FP32)
        per_sample.append(ar)
        if order.index(sel_enum) > order.index(highest_needed):
            highest_needed = sel_enum

    # Create a single typed SDFG for the most restrictive precision
    typed = copy.deepcopy(sdfg)
    change_fptype(typed, dace.float64, highest_needed.value)

    # Aggregate results
    result = {
        'tested_types': list({t for ar in per_sample for t in ar.get('tested_types', [])}),
        'selected_type': enum_to_name[highest_needed],
        'max_error_abs': max_error_abs,
        'max_error_rel': max_error_rel,
        'per_sample': per_sample,
    }
    return typed, result

def analyze_sdfg_precision_with_model_sizes(
    sdfg, model_sizes, max_error_abs=None, max_error_rel=None
):
    """Convenience function to run SDFG precision analysis using model sizes as test cases."""
    # This will be implemented in heat_3d.py to create appropriate InputSpec objects
    # with value_list instead of distribution
    raise NotImplementedError("This function should be implemented in the specific use case file (e.g., heat_3d.py)")

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
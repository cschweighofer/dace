import copy
import logging
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import dace
from dace import dtypes
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributionType(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    LAPLACE = "laplace"
    BETA = "beta"
    GAMMA = "gamma"

class NoiseType(Enum):
    ADDITIVE = "additive"          # noise = data + noise_sample
    MULTIPLICATIVE = "multiplicative"  # noise = data * (1 + noise_sample)

class SensitivityMetric(Enum):
    ABSOLUTE_ERROR = "absolute_error"
    RELATIVE_ERROR = "relative_error"
    MAX_ERROR = "max_error"
    RMS_ERROR = "rms_error"
    CONDITION_NUMBER = "condition_number"
    SIGNAL_TO_NOISE = "signal_to_noise"

@dataclass
class InputGenSpec:
    """Specification for generating input data"""
    shape: Tuple[int, ...]
    dtype: dace.dtypes.typeclass
    distribution: DistributionType
    interval: Tuple[float, float]
    distribution_params: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.distribution_params is None:
            self.distribution_params = {}

@dataclass
class NoiseSpec:
    """Specification for applying noise to a specific input"""
    noise_type: NoiseType
    noise_distribution: DistributionType
    noise_level: float  # Standard deviation or range depending on distribution
    distribution_params: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.distribution_params is None:
            self.distribution_params = {}

@dataclass
class SensitivityResults:
    target_input_name: str
    baseline_data: Dict[str, np.ndarray]
    perturbed_data: Dict[str, np.ndarray]
    baseline_outputs: Dict[str, np.ndarray]
    perturbed_outputs: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    array_wise_metrics: Dict[str, Dict[str, float]]
    noise_spec: NoiseSpec

class AdaptiveSensitivityAnalyzer:
    
    def __init__(self, sdfg: dace.SDFG, 
                 input_generator: Optional[Callable[[str], Dict[str, np.ndarray]]] = None,
                 compile_sdfg: bool = True):
        """
        Initialize the adaptive sensitivity analyzer.
        
        Args:
            sdfg: The SDFG to analyze
            input_generator: Optional function that generates input data
            compile_sdfg: Whether to compile the SDFG immediately
        """
        self.sdfg = sdfg
        self.compiled_sdfg = None
        self.input_generator = input_generator
        
        if compile_sdfg:
            self.compile()
    
    def compile(self):
        """Compile the SDFG for execution"""
        logger.info("Compiling SDFG...")
        self.compiled_sdfg = self.sdfg.compile()
        logger.info("SDFG compilation complete")
    
    def generate_single_input(self, array_name: str, spec: InputGenSpec) -> np.ndarray:
        """
        Generate data for a single input according to specification.
        
        Args:
            array_name: Name of the array
            spec: Input generation specification
            
        Returns:
            Generated array
        """
        logger.info(f"Generating data for {array_name} with shape {spec.shape}")
        
        # Generate data based on distribution type
        if spec.distribution == DistributionType.UNIFORM:
            arr = np.random.uniform(
                low=spec.interval[0], 
                high=spec.interval[1], 
                size=spec.shape
            )
        elif spec.distribution == DistributionType.NORMAL:
            mean = spec.distribution_params.get('mean', 
                                              (spec.interval[0] + spec.interval[1]) / 2)
            std = spec.distribution_params.get('std', 
                                             (spec.interval[1] - spec.interval[0]) / 6)
            arr = np.random.normal(mean, std, size=spec.shape)
            # Clip to interval
            arr = np.clip(arr, spec.interval[0], spec.interval[1])
        elif spec.distribution == DistributionType.LAPLACE:
            # Laplace (double exponential), centered at 'loc' with scale 'scale'
            loc = spec.distribution_params.get('loc', (spec.interval[0] + spec.interval[1]) / 2)
            scale = spec.distribution_params.get('scale', (spec.interval[1] - spec.interval[0]) / 6)
            arr = np.random.laplace(loc, scale, size=spec.shape)
            # Clip to interval to respect bounds
            arr = np.clip(arr, spec.interval[0], spec.interval[1])
        elif spec.distribution == DistributionType.BETA:
            alpha = spec.distribution_params.get('alpha', 2.0)
            beta = spec.distribution_params.get('beta', 2.0)
            arr = np.random.beta(alpha, beta, size=spec.shape)
            # Scale to interval
            arr = spec.interval[0] + arr * (spec.interval[1] - spec.interval[0])
        elif spec.distribution == DistributionType.GAMMA:
            shape = spec.distribution_params.get('shape', 2.0)
            scale = spec.distribution_params.get('scale', 1.0)
            arr = np.random.gamma(shape, scale, size=spec.shape)
            # Scale and shift to fit interval
            arr = spec.interval[0] + (arr / arr.max()) * (spec.interval[1] - spec.interval[0])
        else:
            raise ValueError(f"Unsupported distribution: {spec.distribution}")
        
        # Convert to appropriate dtype
        if spec.dtype in [dace.int32, dace.int64]:
            arr = arr.astype(np.int64 if spec.dtype == dace.int64 else np.int32)
        elif spec.dtype in [dace.float32, dace.float64]:
            arr = arr.astype(np.float64 if spec.dtype == dace.float64 else np.float32)
        else:
            arr = arr.astype(spec.dtype.as_numpy_dtype())
        
        return arr
    
    def apply_noise_to_array(self, array: Union[np.ndarray, float, int], noise_spec: NoiseSpec) -> Union[np.ndarray, float]:
        """
        Apply noise perturbations to a single array or scalar.
        
        Args:
            array: Original array or scalar
            noise_spec: Noise specification
            
        Returns:
            Perturbed array or scalar
        """
        logger.info(f"Applying {noise_spec.noise_type.value} noise")
        
        # Handle scalar values
        if np.isscalar(array):
            # Generate single noise value
            if noise_spec.noise_distribution == DistributionType.UNIFORM:
                noise = np.random.uniform(-noise_spec.noise_level, noise_spec.noise_level)
            elif noise_spec.noise_distribution == DistributionType.NORMAL:
                noise = np.random.normal(0, noise_spec.noise_level)
            elif noise_spec.noise_distribution == DistributionType.LAPLACE:
                noise = np.random.laplace(0, noise_spec.noise_level)
            else:
                raise ValueError(f"Unsupported noise distribution: {noise_spec.noise_distribution}")
            
            # Apply noise based on type
            if noise_spec.noise_type == NoiseType.ADDITIVE:
                return float(array + noise)
            elif noise_spec.noise_type == NoiseType.MULTIPLICATIVE:
                return float(array * (1 + noise))
            else:
                raise ValueError(f"Unsupported noise type: {noise_spec.noise_type}")
                   
        # Handle array values
        else:
            # Generate noise based on distribution
            if noise_spec.noise_distribution == DistributionType.UNIFORM:
                noise = np.random.uniform(
                    -noise_spec.noise_level, 
                    noise_spec.noise_level, 
                    size=array.shape
                )
            elif noise_spec.noise_distribution == DistributionType.NORMAL:
                noise = np.random.normal(0, noise_spec.noise_level, size=array.shape)
            elif noise_spec.noise_distribution == DistributionType.LAPLACE:
                # Use Laplace distribution (double exponential) for symmetric behavior
                noise = np.random.laplace(0, noise_spec.noise_level, size=array.shape)
            else:
                raise ValueError(f"Unsupported noise distribution: {noise_spec.noise_distribution}")
            
            # Apply noise based on type
            if noise_spec.noise_type == NoiseType.ADDITIVE:
                return array + noise
            elif noise_spec.noise_type == NoiseType.MULTIPLICATIVE:
                return array * (1 + noise)
        
        return array

    def execute_sdfg(self, input_data: Dict[str, Union[np.ndarray, float, int]]) -> Dict[str, np.ndarray]:
        """
        Execute the SDFG with given input data.
        
        Args:
            input_data: Dictionary of input arrays and scalars
            
        Returns:
            Dictionary of output arrays
        """
        if self.compiled_sdfg is None:
            self.compile()
        
        # Get SDFG arguments based on data arrays
        sdfg_args = {}
        
        # Add input arrays and scalars
        for name, value in input_data.items():
            if isinstance(value, np.ndarray):
                sdfg_args[name] = value.copy()  # Copy arrays to avoid modifying original data
            else:
                sdfg_args[name] = value  # Keep scalars as-is
        
        # Get the specialized symbol values from the SDFG
        symbol_values = {}
        for symbol, value in self.sdfg.constants.items():
            if hasattr(value, 'get'):
                symbol_values[str(symbol)] = value.get()
            else:
                symbol_values[str(symbol)] = value
        
        # for name, value in input_data.items():
            # if isinstance(value, np.ndarray) and len(value.shape) > 0:
            #     if name in ['u', 'v', 'p'] and len(value.shape) == 2:  # Cavity flow arrays
            #         symbol_values['ny'] = value.shape[0]
            #         symbol_values['nx'] = value.shape[1]
            #     elif name in ['A', 'B'] and len(value.shape) == 3:  # Heat 3D arrays
            #         symbol_values['N'] = value.shape[0]
        
        # Allocate output arrays based on SDFG array descriptors
        for array_name, array_desc in self.sdfg.arrays.items():
            if array_name not in input_data:  # This is likely an output
                # Handle symbolic shapes using the symbol values
                shape = []
                for s in array_desc.shape:
                    if hasattr(s, 'get'):
                        shape.append(s.get())
                    elif isinstance(s, (int, np.integer)):
                        shape.append(s)
                    else:
                        # Try to evaluate symbolic expression using known symbol values
                        try:
                            s_str = str(s)
                            # Replace known symbols with their values
                            for symbol, value in symbol_values.items():
                                s_str = s_str.replace(symbol, str(value))
                            
                            # Evaluate the expression
                            evaluated_dim = eval(s_str)
                            shape.append(int(evaluated_dim))
                            
                        except (ValueError, TypeError, NameError, SyntaxError):
                            # Fallback: try to infer from common patterns
                            s_str = str(s)
                            # if any(pattern in s_str for pattern in ['tmp', '__tmp']):
                            #     # For temporary arrays, skip warnings and use minimal size
                            #     if 'nx' in s_str and 'nx' in symbol_values:
                            #         shape.append(max(1, symbol_values['nx'] - 2))
                            #     elif 'ny' in s_str and 'ny' in symbol_values:
                            #         shape.append(max(1, symbol_values['ny'] - 2))
                            #     elif 'N' in s_str and 'N' in symbol_values:
                            #         shape.append(max(1, symbol_values['N'] - 2))
                            #     else:
                            #         shape.append(1)  # Minimal fallback
                            # else:
                            #     # Only warn for non-temporary arrays
                            logger.warning(f"Could not resolve shape dimension {s} for array {array_name}, using default 1")
                            shape.append(1)
                
                dtype = array_desc.dtype.as_numpy_dtype()
                sdfg_args[array_name] = np.zeros(shape, dtype=dtype)
        
        try:
            # Execute SDFG
            self.compiled_sdfg(**sdfg_args)
        except Exception as e:
            logger.error(f"SDFG execution failed: {e}")
            raise
        
        # Return output arrays - include both explicit outputs AND modified input arrays
        output_data = {}
        for name, array in sdfg_args.items():
            if isinstance(array, np.ndarray):
                # Include explicit output arrays (not in input_data)
                if (name not in input_data and 
                    not name.startswith('__tmp') and 
                    'tmp' not in name):
                    output_data[name] = array.copy()
                # Also include input arrays that may have been modified in-place
                elif (name in input_data and 
                      isinstance(input_data[name], np.ndarray) and
                      not name.startswith('__tmp')):
                    output_data[name] = array.copy()
        
        return output_data

    def compute_sensitivity_metrics(self, baseline_outputs: Dict[str, np.ndarray],
                                perturbed_outputs: Dict[str, np.ndarray],
                                target_input_name: str,
                                baseline_input: Union[np.ndarray, float],
                                perturbed_input: Union[np.ndarray, float]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Compute various sensitivity metrics for the target input.
        """
        global_metrics = {}
        array_wise_metrics = {}

        # Compute input difference - Enhanced for scalars
        if np.isscalar(baseline_input):
            input_diff_mean = float(abs(perturbed_input - baseline_input))
            input_diff_max = input_diff_mean
            input_diff_relative = float(abs(perturbed_input - baseline_input) / (abs(baseline_input) + 1e-15))
            
            # Add scalar-specific metrics
            global_metrics['input_baseline_value'] = float(baseline_input)
            global_metrics['input_perturbed_value'] = float(perturbed_input)
            global_metrics['input_relative_change'] = input_diff_relative
            global_metrics['input_percent_change'] = input_diff_relative * 100
        else:
            input_diff = np.abs(perturbed_input - baseline_input)
            input_diff_mean = float(np.mean(input_diff))
            input_diff_max = float(np.max(input_diff))
            input_diff_relative = float(np.mean(input_diff / (np.abs(baseline_input) + 1e-15)))
            
            # Add array-specific metrics
            global_metrics['input_std_perturbation'] = float(np.std(input_diff))
            global_metrics['input_min_perturbation'] = float(np.min(input_diff))
            global_metrics['input_max_perturbation'] = float(np.max(input_diff))
        
        global_metrics['input_diff_mean'] = input_diff_mean
        global_metrics['input_diff_max'] = input_diff_max
        global_metrics['input_relative_change'] = input_diff_relative
        
        # Enhanced output analysis
        input_output_relationships = {}
        
        for array_name in baseline_outputs:
            if array_name not in perturbed_outputs:
                continue
                
            baseline = baseline_outputs[array_name]
            perturbed = perturbed_outputs[array_name]

            # Compute array-wise metrics
            metrics = {}
            
            # Absolute error
            abs_error = np.abs(perturbed - baseline)
            metrics['max_absolute_error'] = float(np.max(abs_error))
            metrics['mean_absolute_error'] = float(np.mean(abs_error))
            metrics['rms_error'] = float(np.sqrt(np.mean(abs_error**2)))
            metrics['std_absolute_error'] = float(np.std(abs_error))
            
            # Relative error (avoid division by zero)
            baseline_nonzero = baseline + np.finfo(float).eps
            rel_error = abs_error / np.abs(baseline_nonzero)
            metrics['max_relative_error'] = float(np.max(rel_error))
            metrics['mean_relative_error'] = float(np.mean(rel_error))
            metrics['std_relative_error'] = float(np.std(rel_error))
            
            # Signal-to-noise ratio
            signal_power = np.mean(baseline**2)
            noise_power = np.mean(abs_error**2)
            if noise_power > 0:
                metrics['snr_db'] = float(10 * np.log10(signal_power / noise_power))
            else:
                metrics['snr_db'] = float('inf')
            
            # Enhanced sensitivity ratios for scalars
            output_diff_mean = float(np.mean(abs_error))
            output_diff_max = float(np.max(abs_error))
            output_diff_std = float(np.std(abs_error))
            
            if input_diff_mean > 1e-15:
                metrics['sensitivity_ratio'] = output_diff_mean / input_diff_mean
                metrics['max_sensitivity_ratio'] = output_diff_max / input_diff_mean
                
                # Condition number estimate: ||δf||/||f|| / (||δx||/||x||)
                # This approximates κ = ||J|| where J is the Jacobian
                output_relative_change = output_diff_mean / (np.linalg.norm(baseline) + np.finfo(float).eps)
                input_relative_change_norm = input_diff_mean / (np.linalg.norm(baseline_input) + np.finfo(float).eps)
                
                if input_relative_change_norm > 1e-15:
                    metrics['condition_number_estimate'] = output_relative_change / input_relative_change_norm
                else:
                    metrics['condition_number_estimate'] = 0.0
                
                # Alternative condition number using max norms
                output_relative_change_max = output_diff_max / (np.max(np.abs(baseline)) + np.finfo(float).eps)
                input_relative_change_max = input_diff_max / (np.max(np.abs(baseline_input)) + np.finfo(float).eps)
                
                if input_relative_change_max > 1e-15:
                    metrics['condition_number_estimate_max'] = output_relative_change_max / input_relative_change_max
                else:
                    metrics['condition_number_estimate_max'] = 0.0
                
                # For scalars, we can compute per-element sensitivity
                if np.isscalar(baseline_input):
                    output_diff_elements = abs_error.flatten()
                    input_change = abs(perturbed_input - baseline_input)
                    if input_change > 1e-15:
                        element_sensitivities = output_diff_elements / input_change
                        metrics['min_element_sensitivity'] = float(np.min(element_sensitivities))
                        metrics['max_element_sensitivity'] = float(np.max(element_sensitivities))
                        metrics['std_element_sensitivity'] = float(np.std(element_sensitivities))
                        metrics['median_element_sensitivity'] = float(np.median(element_sensitivities))
            else:
                metrics['sensitivity_ratio'] = 0.0
                metrics['max_sensitivity_ratio'] = 0.0
            
            # Amplification factor (how much the output changes relative to input change)
            baseline_magnitude = float(np.mean(np.abs(baseline)))
            if baseline_magnitude > 1e-15 and input_diff_relative > 1e-15:
                output_relative_change = output_diff_mean / baseline_magnitude
                metrics['amplification_factor'] = output_relative_change / input_diff_relative
            else:
                metrics['amplification_factor'] = 0.0
            
            # Stability metrics
            if signal_power > 0:
                metrics['relative_stability'] = float(1.0 / (1.0 + noise_power / signal_power))
            else:
                metrics['relative_stability'] = 0.0
            
            array_wise_metrics[array_name] = metrics

            # Enhanced input-output relationship tracking
            input_output_relationships[f"{target_input_name}_to_{array_name}"] = {
                'input_diff': input_diff_mean,
                'input_relative_change': input_diff_relative,
                'output_diff': output_diff_mean,
                'output_relative_change': output_diff_mean / (baseline_magnitude + 1e-15),
                'sensitivity_ratio': metrics['sensitivity_ratio'],
                'amplification_factor': metrics['amplification_factor'],
                'input_name': target_input_name,
                'output_name': array_name,
                'is_scalar_input': np.isscalar(baseline_input)
            }
        
        # Store relationships in global metrics
        global_metrics['input_output_relationships'] = input_output_relationships
        
        # Enhanced global metrics computation
        if array_wise_metrics:
            all_max_abs_errors = [m['max_absolute_error'] for m in array_wise_metrics.values()]
            all_mean_abs_errors = [m['mean_absolute_error'] for m in array_wise_metrics.values()]
            all_max_rel_errors = [m['max_relative_error'] for m in array_wise_metrics.values()]
            all_mean_rel_errors = [m['mean_relative_error'] for m in array_wise_metrics.values()]
            all_snr = [m['snr_db'] for m in array_wise_metrics.values() if m['snr_db'] != float('inf')]
            all_sensitivity_ratios = [m['sensitivity_ratio'] for m in array_wise_metrics.values()]
            all_amplification_factors = [m['amplification_factor'] for m in array_wise_metrics.values()]
            
            # Condition number estimates (if available)
            all_condition_numbers = [m.get('condition_number_estimate', 0.0) for m in array_wise_metrics.values()]
            all_condition_numbers_max = [m.get('condition_number_estimate_max', 0.0) for m in array_wise_metrics.values()]
            
            global_metrics['max_absolute_error'] = float(max(all_max_abs_errors))
            global_metrics['mean_absolute_error'] = float(np.mean(all_mean_abs_errors))
            global_metrics['max_relative_error'] = float(max(all_max_rel_errors))
            global_metrics['mean_relative_error'] = float(np.mean(all_mean_rel_errors))
            global_metrics['max_sensitivity_ratio'] = float(max(all_sensitivity_ratios))
            global_metrics['mean_sensitivity_ratio'] = float(np.mean(all_sensitivity_ratios))
            global_metrics['max_amplification_factor'] = float(max(all_amplification_factors))
            global_metrics['mean_amplification_factor'] = float(np.mean(all_amplification_factors))
            
            # Add condition number estimates to global metrics
            if any(cn > 0 for cn in all_condition_numbers):
                valid_condition_numbers = [cn for cn in all_condition_numbers if cn > 0]
                global_metrics['condition_number_estimate'] = float(np.mean(valid_condition_numbers))
                global_metrics['max_condition_number_estimate'] = float(max(valid_condition_numbers))
            
            if any(cn > 0 for cn in all_condition_numbers_max):
                valid_condition_numbers_max = [cn for cn in all_condition_numbers_max if cn > 0]
                global_metrics['condition_number_estimate_max_norm'] = float(np.mean(valid_condition_numbers_max))
                global_metrics['max_condition_number_estimate_max_norm'] = float(max(valid_condition_numbers_max))
            
            if all_snr:
                global_metrics['avg_snr_db'] = float(np.mean(all_snr))
                global_metrics['min_snr_db'] = float(min(all_snr))
        
        return global_metrics, array_wise_metrics

    def perform_single_input_sensitivity_analysis(self, 
                                                target_input_name: str,
                                                noise_spec: NoiseSpec,
                                                baseline_inputs: Dict[str, np.ndarray],
                                                num_trials: int = 1) -> SensitivityResults:
        """
        Perform sensitivity analysis on a single input while keeping others constant.
        
        Args:
            target_input_name: Name of the input to analyze
            noise_spec: Noise specification for the target input
            baseline_inputs: All input data (target input will be perturbed)
            num_trials: Number of trials to run
            
        Returns:
            Sensitivity analysis results
        """
        if target_input_name not in baseline_inputs:
            raise ValueError(f"Target input '{target_input_name}' not found in baseline inputs")
        
        logger.info(f"Starting sensitivity analysis for input '{target_input_name}' with {num_trials} trials")
        
        all_global_metrics = []
        all_array_metrics = []
        
        first_baseline_outputs = None
        first_perturbed_outputs = None
        first_perturbed_input = None
        
        for trial in range(num_trials):
            logger.info(f"Running trial {trial + 1}/{num_trials}")
            
            try:
                # Create perturbed version of target input
                target_baseline = baseline_inputs[target_input_name]
                target_perturbed = self.apply_noise_to_array(target_baseline, noise_spec)
                
                # Create full input sets
                perturbed_inputs = baseline_inputs.copy()
                perturbed_inputs[target_input_name] = target_perturbed
                
                # Execute SDFG
                baseline_outputs = self.execute_sdfg(baseline_inputs)
                perturbed_outputs = self.execute_sdfg(perturbed_inputs)
                
                # Compute metrics
                global_metrics, array_metrics = self.compute_sensitivity_metrics(
                    baseline_outputs, perturbed_outputs, target_input_name,
                    target_baseline, target_perturbed
                )
                
                all_global_metrics.append(global_metrics)
                all_array_metrics.append(array_metrics)
                
                # Store first trial data
                if trial == 0:
                    first_baseline_outputs = baseline_outputs
                    first_perturbed_outputs = perturbed_outputs
                    first_perturbed_input = target_perturbed
                        
            except Exception as e:
                logger.error(f"Trial {trial + 1} failed: {e}")
                if trial == 0:
                    raise
                else:
                    logger.warning(f"Skipping failed trial {trial + 1}")
                    continue
        
        if not all_global_metrics:
            raise RuntimeError("All trials failed - cannot compute sensitivity analysis")
        
        # Average metrics across trials
        if len(all_global_metrics) > 1:
            avg_global_metrics = {}
            avg_array_metrics = {}
            
            # Average global metrics (only numeric values)
            for key in all_global_metrics[0]:
                if key != 'input_output_relationships':  # Skip complex data structures
                    values = [m[key] for m in all_global_metrics if key in m and isinstance(m[key], (int, float))]
                    if values:
                        avg_global_metrics[key] = np.mean(values)
                        avg_global_metrics[f'{key}_std'] = np.std(values)
            
            # Keep the first trial's relationships
            avg_global_metrics['input_output_relationships'] = all_global_metrics[0].get('input_output_relationships', {})
            
            # Average array-wise metrics
            for array_name in all_array_metrics[0]:
                avg_array_metrics[array_name] = {}
                for metric_name in all_array_metrics[0][array_name]:
                    values = [m[array_name][metric_name] for m in all_array_metrics 
                            if array_name in m and metric_name in m[array_name] 
                            and isinstance(m[array_name][metric_name], (int, float))]
                    if values:
                        avg_array_metrics[array_name][metric_name] = np.mean(values)
                        avg_array_metrics[array_name][f'{metric_name}_std'] = np.std(values)
        else:
            avg_global_metrics = all_global_metrics[0]
            avg_array_metrics = all_array_metrics[0]
        
        # Create perturbed data dict
        perturbed_data = baseline_inputs.copy()
        perturbed_data[target_input_name] = first_perturbed_input
        
        results = SensitivityResults(
            target_input_name=target_input_name,
            baseline_data=baseline_inputs,
            perturbed_data=perturbed_data,
            baseline_outputs=first_baseline_outputs,
            perturbed_outputs=first_perturbed_outputs,
            metrics=avg_global_metrics,
            array_wise_metrics=avg_array_metrics,
            noise_spec=noise_spec
        )
        
        logger.info("Sensitivity analysis complete")
        return results

    def generate_report(self, results: SensitivityResults, 
                    save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive sensitivity analysis report.
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("ADAPTIVE INPUT SENSITIVITY ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Analysis summary
        report_lines.append(f"TARGET INPUT: {results.target_input_name}")
        report_lines.append(f"NOISE TYPE: {results.noise_spec.noise_type.value}")
        report_lines.append(f"NOISE DISTRIBUTION: {results.noise_spec.noise_distribution.value}")
        report_lines.append(f"NOISE LEVEL: {results.noise_spec.noise_level}")
        report_lines.append("")
        
        # Global metrics
        report_lines.append("GLOBAL SENSITIVITY METRICS")
        report_lines.append("-" * 30)
        for metric, value in results.metrics.items():
            if isinstance(value, float):
                if 'snr' in metric.lower():
                    report_lines.append(f"{metric:30}: {value:12.2f} dB")
                elif 'error' in metric.lower() or 'diff' in metric.lower():
                    report_lines.append(f"{metric:30}: {value:12.6e}")
                else:
                    report_lines.append(f"{metric:30}: {value:12.6f}")
        report_lines.append("")
        
        # Filter out temporary arrays from array-wise metrics - ADD DEBUG INFO
        filtered_metrics = {}
        all_arrays = list(results.array_wise_metrics.keys())
        logger.info(f"All arrays before filtering: {all_arrays}")
        
        for array_name, metrics in results.array_wise_metrics.items():
            filter_condition = not (array_name.startswith('__tmp') or 
                    'tmp' in array_name.lower() or 
                    array_name.startswith('__s') or  # SDFG internal arrays
                    (array_name.startswith('_') and len(array_name) > 2))  # Only exclude long names starting with _
            logger.info(f"Array '{array_name}': filter_condition = {filter_condition}")
            
            if filter_condition:
                filtered_metrics[array_name] = metrics
        
        logger.info(f"Filtered arrays for report: {list(filtered_metrics.keys())}")
        
        # Array-wise metrics
        if filtered_metrics:
            report_lines.append("OUTPUT-WISE SENSITIVITY METRICS")
            report_lines.append("-" * 35)
            for array_name, metrics in filtered_metrics.items():
                report_lines.append(f"\nOutput Array: {array_name}")
                report_lines.append("  " + "-" * (len(array_name) + 13))
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        if 'snr' in metric.lower():
                            report_lines.append(f"  {metric:28}: {value:12.2f} dB")
                        elif 'error' in metric.lower():
                            report_lines.append(f"  {metric:28}: {value:12.6e}")
                        else:
                            report_lines.append(f"  {metric:28}: {value:12.6f}")
            report_lines.append("")
        
        # Input-output relationships (filtered)
        relationships = results.metrics.get('input_output_relationships', {})
        filtered_relationships = {}
        for rel_name, rel_data in relationships.items():
            output_name = rel_data.get('output_name', '')
            if not (output_name.startswith('__tmp') or 
                    'tmp' in output_name.lower() or 
                    output_name.startswith('_')):
                filtered_relationships[rel_name] = rel_data
        
        if filtered_relationships:
            report_lines.append("INPUT-OUTPUT SENSITIVITY RELATIONSHIPS")
            report_lines.append("-" * 40)
            for rel_name, rel_data in filtered_relationships.items():
                input_diff = rel_data['input_diff']
                output_diff = rel_data['output_diff']
                if input_diff > 1e-15:
                    sensitivity = output_diff / input_diff
                    report_lines.append(f"{rel_name:40}: {sensitivity:12.6e}")
            report_lines.append("")
        
        # Data summary
        report_lines.append("DATA SUMMARY")
        report_lines.append("-" * 12)
        target_value = results.baseline_data[results.target_input_name]
        if isinstance(target_value, np.ndarray):
            report_lines.append(f"Target input shape: {target_value.shape}")
            report_lines.append(f"Target input dtype: {target_value.dtype}")
        else:
            report_lines.append(f"Target input value: {target_value}")
            report_lines.append(f"Target input type: {type(target_value)}")
        
        num_outputs = len([name for name in results.baseline_outputs.keys() 
                        if not (name.startswith('__tmp') or 
                               'tmp' in name.lower() or 
                               name.startswith('__s') or  # SDFG internal arrays
                               (name.startswith('_') and len(name) > 2))])  # Only exclude long names starting with _
        report_lines.append(f"Number of meaningful outputs: {num_outputs}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        
        return report

    def plot_sensitivity_results(self, results: SensitivityResults, 
                                save_path: Optional[str] = None):
        """
        Create visualization plots for sensitivity analysis results.
        """
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
        
        # Use the EXACT SAME filtering logic as in the report generation
        filtered_outputs = {}
        filtered_metrics = {}
        
        all_output_arrays = list(results.baseline_outputs.keys())
        all_metric_arrays = list(results.array_wise_metrics.keys())
        logger.info(f"All output arrays before filtering: {all_output_arrays}")
        logger.info(f"All metric arrays before filtering: {all_metric_arrays}")
        
        for name in results.baseline_outputs.keys():
            filter_condition = not (name.startswith('__tmp') or 
                    'tmp' in name.lower() or 
                    name.startswith('__s') or  # SDFG internal arrays
                    (name.startswith('_') and len(name) > 2))  # Only exclude long names starting with _
            logger.info(f"Output array '{name}': filter_condition = {filter_condition}")
            
            if filter_condition:
                filtered_outputs[name] = results.baseline_outputs[name]
                if name in results.array_wise_metrics:
                    filtered_metrics[name] = results.array_wise_metrics[name]
        
        num_outputs = len(filtered_outputs)
        logger.info(f"Found {num_outputs} meaningful outputs for plotting: {list(filtered_outputs.keys())}")
        logger.info(f"Found {len(filtered_metrics)} meaningful metrics for plotting: {list(filtered_metrics.keys())}")
        
        if num_outputs == 0:
            logger.warning("No meaningful output arrays to plot after filtering")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Sensitivity Analysis: {results.target_input_name}', fontsize=16)
        
        output_names = list(filtered_metrics.keys())
        
        # Plot 1: Sensitivity ratios by output
        ax = axes[0, 0]
        if output_names:
            sensitivity_ratios = [filtered_metrics[name]['sensitivity_ratio'] 
                                for name in output_names]
            
            ax.bar(output_names, sensitivity_ratios)
            ax.set_xlabel('Output Arrays')
            ax.set_ylabel('Sensitivity Ratio')
            ax.set_title('Sensitivity Ratios by Output')
            plt.setp(ax.get_xticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, 'No meaningful outputs', ha='center', va='center', transform=ax.transAxes)
        
        # Plot 2: Error comparison
        ax = axes[0, 1]
        if output_names:
            max_abs_errors = [filtered_metrics[name]['max_absolute_error'] 
                            for name in output_names]
            max_rel_errors = [filtered_metrics[name]['max_relative_error'] 
                            for name in output_names]
            
            x = np.arange(len(output_names))
            width = 0.35
            
            ax.bar(x - width/2, max_abs_errors, width, label='Max Absolute Error')
            ax.bar(x + width/2, max_rel_errors, width, label='Max Relative Error')
            ax.set_xlabel('Output Arrays')
            ax.set_ylabel('Error')
            ax.set_title('Maximum Errors by Output')
            ax.set_xticks(x)
            ax.set_xticklabels(output_names, rotation=45)
            ax.legend()
            
            if max(max_abs_errors + max_rel_errors) > 0:
                ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No meaningful outputs', ha='center', va='center', transform=ax.transAxes)
        
        # Plot 3: Output error distribution for first meaningful output
        if output_names:
            ax = axes[1, 0]
            first_output = output_names[0]
            baseline = filtered_outputs[first_output].flatten()
            perturbed = results.perturbed_outputs[first_output].flatten()
            errors = np.abs(perturbed - baseline)
            
            if np.max(errors) > 1e-15:
                ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
                if np.max(errors) > np.min(errors[errors > 0]) * 10:
                    ax.set_yscale('log')
                ax.set_xlabel('Absolute Error')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Error Distribution - {first_output}')
            else:
                ax.text(0.5, 0.5, 'No significant errors\n(Perfect stability)', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Error Distribution - {first_output}')
        else:
            ax.text(0.5, 0.5, 'No meaningful outputs', ha='center', va='center', transform=ax.transAxes)
        
        if output_names:
            ax = axes[1, 1]
            first_output = output_names[0]
            baseline = filtered_outputs[first_output].flatten()
            perturbed = results.perturbed_outputs[first_output].flatten()
            
            # Sample points for readability
            max_points = 1000
            if len(baseline) > max_points:
                indices = np.random.choice(len(baseline), max_points, replace=False)
                baseline_sample = baseline[indices]
                perturbed_sample = perturbed[indices]
            else:
                baseline_sample = baseline
                perturbed_sample = perturbed
            
            # Calculate differences to determine if we need special handling
            abs_diff = np.abs(perturbed_sample - baseline_sample)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)
            
            # If differences are very small relative to values, show difference plot instead
            relative_diff = max_diff / (np.max(np.abs(baseline_sample)) + 1e-15)
            
            # Debug logging
            logger.info(f"Plot decision for {first_output}: max_diff={max_diff:.6e}, relative_diff={relative_diff:.6f}")
            
            if relative_diff < 0.01:  # Less than 1% relative difference
                logger.info(f"Using difference plot for {first_output} (relative_diff < 1%)")
                # Show difference plot instead of scatter
                ax.scatter(baseline_sample, perturbed_sample - baseline_sample, alpha=0.6, s=1, c='blue')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
                ax.set_xlabel('Baseline Output')
                ax.set_ylabel('Output Difference (Perturbed - Baseline)')
                ax.set_title(f'Output Differences - {first_output}\n(Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e})')
                
                # Add text annotation about stability
                ax.text(0.02, 0.98, f'Very stable: max relative change {relative_diff*100:.3f}%', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                logger.info(f"Using standard scatter plot for {first_output} (relative_diff >= 1%)")
                # Standard scatter plot for larger differences
                ax.scatter(baseline_sample, perturbed_sample, alpha=0.6, s=1)
                min_val = min(baseline_sample.min(), perturbed_sample.min())
                max_val = max(baseline_sample.max(), perturbed_sample.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                ax.set_xlabel('Baseline Output')
                ax.set_ylabel('Perturbed Output')
                ax.set_title(f'Baseline vs Perturbed - {first_output}')
        else:
            ax.text(0.5, 0.5, 'No meaningful outputs', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        
        try:
            plt.show()
        except Exception as e:
            logger.info(f"Could not display plot (headless environment): {e}")


# --- Helpers for multi-size grid plotting ---
def _filter_metric_arrays_for_plot(array_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Filter out temporary/internal arrays (same logic as report/plots)."""
    filtered: Dict[str, Dict[str, float]] = {}
    for array_name, metrics in array_metrics.items():
        if not (array_name.startswith('__tmp') or 
                'tmp' in array_name.lower() or 
                array_name.startswith('__s') or
                (array_name.startswith('_') and len(array_name) > 2)):
            filtered[array_name] = metrics
    return filtered


def plot_grid_sensitivity_over_sizes(
    base_sdfg: dace.SDFG,
    sizes: List[Tuple[str, Callable[[], Tuple[Dict[str, Any], Dict[str, np.ndarray]]]]],
    noise_specs: List[Tuple[str, NoiseSpec]],
    target_input_name: str,
    save_path: Optional[str] = None,
    preferred_output_name: Optional[str] = None,
):
    """
    Create a grid where columns are sizes (e.g., S, M, L, PAPER) and rows are noise specs.
    Each cell shows a bar chart of sensitivity_ratio by output array for the given size+noise.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # noqa: F811 (shadowed import for safe backend)

    rows = len(noise_specs)
    cols = len(sizes)
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.0 * rows), squeeze=False)#, sharey=True, sharex=True
    fig.suptitle(f"Sensitivity grid (target input: {target_input_name})", fontsize=14)

    # Pre-build one analyzer per column (size) to reuse compilation across rows
    analyzers: List[Optional[AdaptiveSensitivityAnalyzer]] = [None] * cols
    baselines: List[Optional[Dict[str, np.ndarray]]] = [None] * cols

    for c, (size_label, init_fn) in enumerate(sizes):
        # Prepare inputs for this size via init function
        specialize_symbols, baseline_inputs = init_fn()
        sdfg_spec = copy.deepcopy(base_sdfg)
        try:
            if specialize_symbols:
                sdfg_spec.specialize(specialize_symbols)
        except Exception as e:
            logger.warning(f"Could not specialize SDFG with symbols {specialize_symbols}: {e}")
        analyzers[c] = AdaptiveSensitivityAnalyzer(sdfg_spec, compile_sdfg=True)
        baselines[c] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in baseline_inputs.items()}

        # Column titles
        axes[0, c].set_title(size_label, fontsize=12)

    for r, (noise_label, nspec) in enumerate(noise_specs):
        for c, _ in enumerate(sizes):
            ax = axes[r, c]
            try:
                analyzer = analyzers[c]
                baseline_inputs = baselines[c]
                assert analyzer is not None and baseline_inputs is not None
                results = analyzer.perform_single_input_sensitivity_analysis(
                    target_input_name=target_input_name,
                    noise_spec=nspec,
                    baseline_inputs=baseline_inputs,
                    num_trials=1
                )

                # Choose first meaningful output
                filtered_outputs = {name: arr for name, arr in results.baseline_outputs.items()
                                    if not (name.startswith('__tmp') or 'tmp' in name.lower() or
                                            name.startswith('__s') or (name.startswith('_') and len(name) > 2))}
                if not filtered_outputs:
                    ax.text(0.5, 0.5, 'No outputs', ha='center', va='center')
                else:
                    # Prefer a specific output if requested
                    if preferred_output_name and preferred_output_name in filtered_outputs:
                        first_output = preferred_output_name
                    else:
                        first_output = list(filtered_outputs.keys())[0]
                    baseline = results.baseline_outputs[first_output].flatten()
                    perturbed = results.perturbed_outputs[first_output].flatten()

                    # Sample for readability
                    max_points = 1000
                    if len(baseline) > max_points:
                        idx = np.random.choice(len(baseline), max_points, replace=False)
                        bsample = baseline[idx]
                        psample = perturbed[idx]
                    else:
                        bsample = baseline
                        psample = perturbed

                    abs_diff = np.abs(psample - bsample)
                    max_diff = float(np.max(abs_diff)) if abs_diff.size else 0.0
                    rel = max_diff / (float(np.max(np.abs(bsample))) + 1e-15) if bsample.size else 0.0

                    # Plot difference scatter if very stable, else baseline vs perturbed
                    if rel < 0.01:
                        ax.scatter(bsample, psample - bsample, alpha=0.5, s=2, c='tab:blue')
                        ax.axhline(y=0, color='r', linestyle='--', alpha=0.6, linewidth=0.8)
                        ax.set_ylabel(noise_label if c == 0 else '')
                    else:
                        ax.scatter(bsample, psample, alpha=0.5, s=2, c='tab:blue')
                        mn = min(bsample.min(initial=0.0), psample.min(initial=0.0))
                        mx = max(bsample.max(initial=0.0), psample.max(initial=0.0))
                        ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.6, linewidth=0.8)
                        ax.set_ylabel(noise_label if c == 0 else '')

            except Exception as e:
                logger.error(f"Grid cell ({r},{c}) failed: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', color='red')

            # Clean up axis for dense grid
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(alpha=0.15, linestyle=':')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Grid plot saved to {save_path}")
    try:
        plt.show()
    except Exception as e:
        logger.info(f"Could not display plot (headless environment): {e}")
    return fig


def plot_grid_sensitivity_over_sizes_rmse(
    base_sdfg: dace.SDFG,
    sizes: List[Tuple[str, Callable[[], Tuple[Dict[str, Any], Dict[str, np.ndarray]]]]],
    noise_specs: List[Tuple[str, NoiseSpec]],
    target_input_name: str,
    save_path: Optional[str] = None,
    preferred_output_name: Optional[str] = None,
    annot_fontsize: int = 11,
):
    """
    Same grid as plot_grid_sensitivity_over_sizes, but each cell encodes the
    RMSE error (between perturbed and baseline outputs) as a heatmap value.

    - Columns: model sizes
    - Rows: noise specs
    - Value: RMSE of selected output array (preferred_output_name if provided,
      else the first meaningful output).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # noqa: F811
    import matplotlib.patheffects as pe

    rows = len(noise_specs)
    cols = len(sizes)

    # Pre-build one analyzer per column (size) to reuse compilation across rows
    analyzers: List[Optional[AdaptiveSensitivityAnalyzer]] = [None] * cols
    baselines: List[Optional[Dict[str, np.ndarray]]] = [None] * cols

    size_labels: List[str] = []
    for c, (size_label, init_fn) in enumerate(sizes):
        size_labels.append(size_label)
        sdfg_spec = copy.deepcopy(base_sdfg)
        try:
            specialize_symbols, baseline_inputs = init_fn()
        except Exception as e:
            logger.error(f"Init function for size '{size_label}' failed: {e}")
            raise
        try:
            if specialize_symbols:
                sdfg_spec.specialize(specialize_symbols)
        except Exception as e:
            logger.warning(f"Could not specialize SDFG with symbols {specialize_symbols}: {e}")
        analyzers[c] = AdaptiveSensitivityAnalyzer(sdfg_spec, compile_sdfg=True)
        baselines[c] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in baseline_inputs.items()}

    rmse_matrix = np.full((rows, cols), np.nan, dtype=float)

    for r, (noise_label, nspec) in enumerate(noise_specs):
        for c, _ in enumerate(sizes):
            try:
                analyzer = analyzers[c]
                baseline_inputs = baselines[c]
                assert analyzer is not None and baseline_inputs is not None

                results = analyzer.perform_single_input_sensitivity_analysis(
                    target_input_name=target_input_name,
                    noise_spec=nspec,
                    baseline_inputs=baseline_inputs,
                    num_trials=1,
                )

                # Choose meaningful outputs
                filtered_outputs = {
                    name: arr for name, arr in results.baseline_outputs.items()
                    if not (name.startswith('__tmp') or 'tmp' in name.lower() or
                            name.startswith('__s') or (name.startswith('_') and len(name) > 2))
                }
                if not filtered_outputs:
                    rmse_matrix[r, c] = np.nan
                    continue

                # Pick output
                if preferred_output_name and preferred_output_name in filtered_outputs:
                    out_name = preferred_output_name
                else:
                    out_name = list(filtered_outputs.keys())[0]

                # Prefer metric if available, else compute directly
                rmse_val: Optional[float] = None
                if out_name in results.array_wise_metrics:
                    rmse_val = results.array_wise_metrics[out_name].get('rms_error')

                if rmse_val is None:
                    b = results.baseline_outputs[out_name].astype(float).ravel()
                    p = results.perturbed_outputs[out_name].astype(float).ravel()
                    if b.size == 0:
                        rmse_val = float('nan')
                    else:
                        rmse_val = float(np.sqrt(np.mean((p - b) ** 2)))

                rmse_matrix[r, c] = rmse_val

            except Exception as e:
                logger.error(f"RMSE grid cell ({r},{c}) failed: {e}")
                rmse_matrix[r, c] = np.nan

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(1.6 * cols + 2, 1.0 * rows + 2))
    im = ax.imshow(rmse_matrix, aspect='auto', cmap='magma')
    ax.set_title(f"RMSE Sensitivity Grid (target: {target_input_name})", fontsize=12)
    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels(size_labels)
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels([lbl for (lbl, _) in noise_specs])
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    # Annotate cells with values (scientific notation for readability)
    # Improve readability by adapting text color to background luminance and adding an outline
    norm = im.norm
    cmap = im.cmap
    for r in range(rows):
        for c in range(cols):
            val = rmse_matrix[r, c]
            if np.isnan(val):
                text = '—'
            else:
                # Compact formatting depending on magnitude
                if val == 0:
                    text = '0'
                elif val < 1e-3 or val >= 1e2:
                    text = f"{val:.1e}"
                else:
                    text = f"{val:.3f}"
            # Determine background RGBA at this cell and derive luminance
            try:
                rgba = cmap(norm(val)) if np.isfinite(val) else cmap(np.nan)
                r_ch, g_ch, b_ch = rgba[0], rgba[1], rgba[2]
                luminance = 0.2126 * r_ch + 0.7152 * g_ch + 0.0722 * b_ch
            except Exception:
                # Fallback if colormap/norm isn't available
                luminance = 0.0

            txt_color = 'black' if luminance > 0.55 else 'white'
            stroke_color = 'white' if txt_color == 'black' else 'black'
            ax.text(
                c,
                r,
                text,
                va='center',
                ha='center',
                color=txt_color,
                fontsize=annot_fontsize,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=stroke_color)],
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('RMSE', rotation=90)

    ax.set_xlabel('Model size')
    ax.set_ylabel('Noise spec')
    ax.grid(False)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"RMSE grid heatmap saved to {save_path}")
    try:
        plt.show()
    except Exception as e:
        logger.info(f"Could not display plot (headless environment): {e}")
    return fig
import numpy as np
import dace
import matplotlib.pyplot as plt

from heat_3d_dace import kernel
from cavity_flow_dace import cavity_flow
from enhanced_input_sensitivity_analysis import (
    NoiseType,
    NoiseSpec,
    DistributionType,
    AdaptiveSensitivityAnalyzer
)

def cavity_initialize(ny, nx):
    u = np.zeros((ny, nx), dtype=np.float64)
    v = np.zeros((ny, nx), dtype=np.float64)
    p = np.zeros((ny, nx), dtype=np.float64)
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = .1 / ((nx - 1) * (ny - 1))
    return u, v, p, dx, dy, dt

def run_cavity_flow_simple():
    """
    Simply run the cavity flow SDFG with sample values.
    """
    print("Creating cavity flow SDFG...")
    nx, ny = 41, 41
    nit = 50  # Number of pressure iterations
    nt = 100  # Number of timesteps
    
    sdfg = cavity_flow.to_sdfg()
    sdfg.specialize(dict(nx=nx, ny=ny, nit=nit))
    # sdfg.save('cavity_flow.sdfg')

    u, v, p, dx, dy, dt = cavity_initialize(ny, nx)
    
    # Set boundary condition: lid velocity
    # u[:, -1] = 0.25  # Top wall moving right with u=1

    rho = 1.0            # Density
    nu = 0.1             # Viscosity
    
    print(f"Running cavity flow simulation for {nt} timesteps...")
    sdfg(u=u, v=v, p=p, dt=dt, dx=dx, dy=dy, rho=rho, nu=nu, nit=nit, nt=nt)
    
    print("Simulation completed!")
    print(f"Max u velocity: {np.max(u)}")
    print(f"Max v velocity: {np.max(v)}")
    print(f"Max pressure: {np.max(p)}")

def run_cavity_flow():
    """
    Simply run the cavity flow SDFG with sample values.
    """
    nx, ny = 41, 41
    nit = 50  # Number of pressure iterations
    nt = 100  # Number of timesteps
    
    sdfg = cavity_flow.to_sdfg()
    sdfg.specialize(dict(nx=nx, ny=ny, nit=nit))
    # sdfg.save('cavity_flow.sdfg')

    u, v, p, dx, dy, dt = cavity_initialize(ny, nx)
    
    # Set boundary condition: lid velocity
    # u[:, -1] = 0.25  # Top wall moving right with u=1

    rho = 1.0            # Density
    nu = 0.1             # Viscosity
    
    sdfg(u=u, v=v, p=p, dt=dt, dx=dx, dy=dy, rho=rho, nu=nu, nit=nit, nt=nt)
    
    return u, v, p

def heat_initialize(N, datatype=np.float64):
    A = np.fromfunction(lambda i, j, k: (i + j + (N - k)) * 10 / N, (N, N, N),
                        dtype=datatype)
    B = np.copy(A)

    return A, B

def run_heat_3d_simple():
    """
    Run the 3D heat equation SDFG with sample values.
    """
    print("Creating 3D heat equation SDFG...")
    N = 50  # Grid size (N x N x N)
    TSTEPS = 100  # Number of time steps
    
    sdfg = kernel.to_sdfg()
    sdfg.specialize(dict(N=N))
    # sdfg.save('heat_3d.sdfg')

    # Initialize input arrays
    print(f"Initializing arrays of size {N}x{N}x{N}...")
    A, B = heat_initialize(N)
    
    print(f"Running 3D heat equation simulation for {TSTEPS} timesteps...")
    sdfg(TSTEPS=TSTEPS, A=A, B=B)
    
    print("Simulation completed!")
    print(f"Max value in A: {np.max(A)}")
    print(f"Min value in A: {np.min(A)}")
    print(f"Mean value in A: {np.mean(A)}")

def run_heat_3d():
    """
    Run the 3D heat equation SDFG with sample values.
    """
    N = 50  # Grid size (N x N x N)
    TSTEPS = 100  # Number of time steps
    
    sdfg = kernel.to_sdfg()
    sdfg.specialize(dict(N=N))
    # sdfg.save('heat_3d.sdfg')

    A, B = heat_initialize(N)
    
    sdfg(TSTEPS=TSTEPS, A=A, B=B)
    
    return A

def cavity_flow_sensitivity_analysis(target_input: str = 'u', 
                                    noise_type: NoiseType = NoiseType.ADDITIVE,
                                    noise_level: float = 0.01,
                                    num_trials: int = 5):
    """
    Perform sensitivity analysis on cavity flow SDFG.
    
    Args:
        target_input: Input to analyze ('u', 'v', 'p', 'dt', 'dx', 'dy', 'rho', 'nu')
        noise_type: Type of noise to apply
        noise_level: Level/magnitude of noise
        num_trials: Number of trials to run
    """
    from cavity_flow_dace import cavity_flow
    
    # Create and specialize SDFG
    nx, ny = 41, 41
    nit = 50
    nt = 100
    
    sdfg = cavity_flow.to_sdfg()
    sdfg.specialize(dict(nx=nx, ny=ny, nit=nit))
    
    # Generate baseline inputs
    # from cavity_flow_analysis2 import cavity_initialize
    u, v, p, dx, dy, dt = cavity_initialize(ny, nx)
    
    baseline_inputs = {
        'u': u,
        'v': v, 
        'p': p,
        'dt': dt,        
        'dx': dx,        
        'dy': dy,        
        'rho': 1.0,      
        'nu': 0.1,       
        'nit': nit,      
        'nt': nt         
    }
    
    # Create noise specification
    noise_spec = NoiseSpec(
        noise_type=noise_type,
        noise_distribution=DistributionType.NORMAL,
        noise_level=noise_level
    )
    
    # Create analyzer and run analysis
    analyzer = AdaptiveSensitivityAnalyzer(sdfg)
    
    results = analyzer.perform_single_input_sensitivity_analysis(
        target_input_name=target_input,
        noise_spec=noise_spec,
        baseline_inputs=baseline_inputs,
        num_trials=num_trials
    )
    
    # Generate report and plots
    report = analyzer.generate_report(results, f'cavity_flow_{target_input}_sensitivity_report.txt')
    print(report)
    
    analyzer.plot_sensitivity_results(results, f'cavity_flow_{target_input}_sensitivity_plots.png')
    
    return results, analyzer

def heat_3d_sensitivity_analysis(target_input: str = 'A',
                                noise_type: NoiseType = NoiseType.ADDITIVE, 
                                noise_level: float = 0.01,
                                num_trials: int = 5):
    """
    Perform sensitivity analysis on 3D heat equation SDFG.
    
    Args:
        target_input: Input to analyze ('A', 'B', 'TSTEPS')
        noise_type: Type of noise to apply
        noise_level: Level/magnitude of noise
        num_trials: Number of trials to run
    """
    from heat_3d_dace import kernel
    
    # Create and specialize SDFG
    N = 50
    TSTEPS = 100
    
    sdfg = kernel.to_sdfg()
    sdfg.specialize(dict(N=N))
    
    # Generate baseline inputs
    # from cavity_flow_analysis2 import heat_initialize
    A, B = heat_initialize(N)
    
    baseline_inputs = {
        'A': A,
        'B': B,
        'TSTEPS': TSTEPS 
    }
    
    # Create noise specification
    noise_spec = NoiseSpec(
        noise_type=noise_type,
        noise_distribution=DistributionType.NORMAL,
        noise_level=noise_level
    )
    
    # Create analyzer and run analysis
    analyzer = AdaptiveSensitivityAnalyzer(sdfg)
    
    results = analyzer.perform_single_input_sensitivity_analysis(
        target_input_name=target_input,
        noise_spec=noise_spec,
        baseline_inputs=baseline_inputs,
        num_trials=num_trials
    )
    
    # Generate report and plots
    report = analyzer.generate_report(results, f'heat_3d_{target_input}_sensitivity_report.txt')
    print(report)
    
    analyzer.plot_sensitivity_results(results, f'heat_3d_{target_input}_sensitivity_plots.png')
    
    return results, analyzer


# if __name__ == "__main__":
#     # run_heat_3d_simple()
#     run_cavity_flow_simple()

# Example usage
if __name__ == "__main__":
    # Example 1: Cavity flow analysis
    print("Running cavity flow sensitivity analysis...")
    results, analyzer = cavity_flow_sensitivity_analysis(
        target_input='v', 
        noise_type=NoiseType.MULTIPLICATIVE,
        noise_level=0.01,
        num_trials=3
    )
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Heat 3D analysis  
    print("Running heat 3D sensitivity analysis...")
    results, analyzer = heat_3d_sensitivity_analysis(
        target_input='A',
        noise_type=NoiseType.MULTIPLICATIVE, 
        noise_level=0.01,
        num_trials=3
    )
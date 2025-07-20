
import dace
import numpy as np
#import cupy as cp

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import copy

import numpy as np
from scipy import stats
from real_lowering import InputSpec, analyze_sdfg_precision


def initialize(N, datatype=np.float64):
    u = np.fromfunction(lambda i, j: (i + N - j) / N, (N, N), dtype=datatype)

    return u

sdfg = dace.SDFG.from_file("../sdfgs/adi_auto_opt_cpu.sdfg")

def run_S():
    TSTEPS, N = (5, 100)
    u = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, u=u)
    print("Result: ", u)

def run_M():
    TSTEPS, N = (20, 200)
    u = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, u=u)

def run_L():
    TSTEPS, N = (50, 500)
    u = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, u=u)

def run_paper():
    TSTEPS, N = (100, 200)
    u = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, u=u)

def run_analysis():
    TSTEPS, N = (5, 100)
    u = initialize(N=N)
    class ConstantDistribution:
        def __init__(self, value):
            self.value = value
        def rvs(self, size):
            return np.repeat(self.value[None, ...], size[0], axis=0)
    const_dist = ConstantDistribution(u)
    input_spec = InputSpec('u', const_dist)
    extra_args = {'N': N, 'TSTEPS': TSTEPS}
    final_sdfg, analysis_results = analyze_sdfg_precision(
        sdfg, [input_spec], max_error_abs=1e-5, num_samples=1, extra_args=extra_args
    )
    # print("Analysis results:", analysis_results)

def use_self_defined_fptype(sdfg: dace.SDFG, src_fptype: dace.dtypes.typeclass, dace_internal_fptype: dace.dtypes.typeclass):
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

def test_change_fptype_fp64_noop():
    TSTEPS, N = (5, 100)
    u = initialize(N=N)
    # Make a deep copy of the SDFG
    sdfg_copy = copy.deepcopy(sdfg)
    # Call change_fptype with both src and internal type as FP64
    use_self_defined_fptype(sdfg_copy, dace.float64, dace.rational)
    # Run the SDFG as in run_S
    u2 = initialize(N=N)
    sdfg_copy(TSTEPS=TSTEPS, N=N, _u=u2)
    print("Result after change_fptype (FP64->rational):", u2)
    # Optionally, compare to original run_S result
    u_ref = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, u=u_ref)
    print("Original result:", u_ref)
    print("Difference:", np.max(np.abs(u2 - u_ref)))

if __name__ == "__main__":
    # run_analysis()
    test_change_fptype_fp64_noop()
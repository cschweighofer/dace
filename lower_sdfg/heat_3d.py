import dace
import numpy as np
#import cupy as cp

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import copy

import numpy as np
from scipy import stats

# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float64):
    A = np.fromfunction(lambda i, j, k: (i + j + (N - k)) * 10 / N, (N, N, N),
                        dtype=datatype)
    B = np.copy(A)

    return A, B


sdfg = dace.SDFG.from_file("../sdfgs/heat_3d_auto_opt_cpu.sdfg")

def run_S():
    TSTEPS, N = (25, 25)
    A, B = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, A=A, B=B)
    print("Result: ", A, B)

def run_M():
    TSTEPS, N = (50, 40)
    A, B = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, A=A, B=B)

def run_L():
    TSTEPS, N = (100, 70)
    A, B = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, A=A, B=B)

def run_paper():
    TSTEPS, N = (500, 120)
    A, B = initialize(N=N)
    sdfg(TSTEPS=TSTEPS, N=N, A=A, B=B)


def change_fptype(sdfg: dace.SDFG, src_fptype: dace.dtypes.typeclass, dace_internal_fptype: dace.dtypes.typeclass):

    def sdfg_types_to_mpf(sdfg, src_fptype, mpf_fptype):
        """
        Recursively transform all float64/double types in the SDFG to dace.mpf, including arrays, scalars, memlets, and connectors. This leverages DaCe's type system for correct C++ code generation.
        """
        # Transform array and scalar types
        for _, arr in sdfg.arrays.items():
            if arr.dtype == src_fptype:
                arr.dtype = mpf_fptype

        # Transform connectors and tasklet input/output types
        for state in sdfg.nodes():
            if hasattr(state, 'nodes'):
                for node in state.nodes():
                    if hasattr(node, 'inputs') and isinstance(node.inputs, dict):
                        for k, v in node.inputs.items():
                            if v == src_fptype:
                                node.inputs[k] = mpf_fptype
                    if hasattr(node, 'outputs') and isinstance(node.outputs, dict):
                        for k, v in node.outputs.items():
                            if v == src_fptype:
                                node.outputs[k] = mpf_fptype
                    if hasattr(node, 'local_variables') and isinstance(node.local_variables, dict):
                        for k, v in node.local_variables.items():
                            if v == src_fptype:
                                node.local_variables[k] = mpf_fptype

        # Transform memlets
        for state in sdfg.nodes():
            if hasattr(state, 'edges'):
                for edge in state.edges():
                    memlet = getattr(edge, 'data', None)
                    if memlet is not None and hasattr(memlet, 'dtype') and memlet.dtype == src_fptype:
                        memlet.dtype = mpf_fptype

    sdfg_types_to_mpf(sdfg, src_fptype, dace_internal_fptype)

    # Add simulated double transient arrays to the SDFG
    arrays_that_become_transient = set()
    for arr_name, arr in sdfg.arrays.items():
        if arr.dtype == dace_internal_fptype:
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

        # Add a tasklet that performs the type cast (no C-style cast)
        tasklet_code = "out = in;" if dst_arr.dtype == dace_internal_fptype else f"out = in;"
        tasklet = state.add_tasklet(
            name=f"copy_{src_arr_name}_to_{dst_arr_name}",
            inputs={"in"},
            outputs={"out"},
            code=tasklet_code,
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


def compare_to_high_precision():
    TSTEPS, N = (5, 5)
    A_initial, B_initial = initialize(N=N)

    # Run original SDFG for reference
    A_ref, B_ref = np.copy(A_initial), np.copy(B_initial)
    sdfg(TSTEPS=TSTEPS, N=N, A=A_ref, B=B_ref)
    print("Original executed")
    # Make a deep copy of the SDFG and change floating point type to higher precision
    sdfg_copy = copy.deepcopy(sdfg)
    change_fptype(sdfg_copy, dace.float64, dace.mpf)

    # Run the high precision SDFG
    A2, B2 = np.copy(A_initial), np.copy(B_initial)
    sdfg_copy(TSTEPS=TSTEPS, N=N, _A=A2, _B=B2)
    print("Rational executed")
    # Compare results
    diff = max(np.max(np.abs(A2 - A_ref)), np.max(np.abs(B2 - B_ref)))
    # print("Difference:", diff)
    diff_A = np.sqrt(np.mean((A2 - A_ref)**2))
    diff_B = np.sqrt(np.mean((B2 - B_ref)**2))
    # diff = max(diff_A, diff_B)
    print("RMS Difference:", diff)
    return diff


if __name__ == "__main__":
    # old_cpu_args = dace.config.Config.get('compiler', 'cpu', 'args')
    # dace.config.Config.set('compiler', 'cpu', 'args', value=f'{old_cpu_args} -fsanitize=address -g')
    # old_linker_args = dace.config.Config.get('compiler', 'linker', 'args')
    # dace.config.Config.set('compiler', 'linker', 'args', value=f'{old_linker_args} -fsanitize=address')
    # import shutil
    # shutil.rmtree('.dacecache', ignore_errors=True)
    compare_to_high_precision()

#     Original executed
# free(): double free detected in tcache 2
# free(): double free detected in tcache 2
# free(): double free detected in tcache 2
# Aborted (core dumped)

# echo $(gcc -print-file-name=libasan.so)


# export LD_PRELOAD="$(gcc -print-file-name=libasan.so)"


# fsanitize=address'

# gdb --args python your_script.py





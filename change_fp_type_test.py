import dace
import copy
import numpy as np

@dace.program
def test_program(A: dace.float64[10, 10], B: dace.float64[10, 10], C: dace.float64[10, 10]):
    C = A + B

sdfg = test_program.to_sdfg()
sdfg.save("test_sdfg.sdfg")


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

# use_self_defined_fptype(modified_sdfg, dace.float64, dace.simulated_double)
# modified_sdfg.save("test_sdfg_with_transients.sdfg")

def test_with_random_input(
    original_sdfg: dace.SDFG,
    shape=(10, 10),
    num_executions=5
):
    modified_sdfg = copy.deepcopy(original_sdfg)
    use_self_defined_fptype(modified_sdfg, dace.float64, dace.simulated_double)
    modified_sdfg.save("test_sdfg_with_transients.sdfg")

    errors = []

    for _ in range(num_executions):
        A = dace.ndarray(shape, dtype=dace.float64)
        B = dace.ndarray(shape, dtype=dace.float64)
        # A[:] = np.random.rand(*shape) 
        # B[:] = np.random.rand(*shape)
        A[:] = np.random.uniform(1.0, 3.0, size=shape)  #TODO: Allow for different distributions
        B[:] = np.random.uniform(1.0, 3.0, size=shape)

        C_orig = dace.ndarray(shape, dtype=dace.float64)
        C_orig[:] = 0
        original_sdfg(A=A, B=B, C=C_orig)

        C_mod = dace.ndarray(shape, dtype=dace.float64)
        C_mod[:] = 0
        modified_sdfg(_A=A, _B=B, _C=C_mod)

        diff = np.abs(C_orig - C_mod)
        errors.append(np.amax(diff))

    # print("Mean absolute differences per execution:", errors)
    # print("Overall mean difference:", np.mean(errors))
    print("Overall max difference", np.amax(errors))

if __name__ == "__main__":
    # A = dace.ndarray([10, 10], dtype=dace.float64)
    # B = dace.ndarray([10, 10], dtype=dace.float64)
    # C = dace.ndarray([10, 10], dtype=dace.float64)
    # A[:] = 1.0
    # B[:] = 2.0
    # C[:] = 0.0

    # test_program(A=A, B=B, C=C)
    # print("Original Result C:")
    # print(C)

    # A = dace.ndarray([10, 10], dtype=dace.float64)
    # B = dace.ndarray([10, 10], dtype=dace.float64)
    # C = dace.ndarray([10, 10], dtype=dace.float64)
    # A[:] = 1.0
    # B[:] = 2.0
    # C[:] = 0.0

    # modified_sdfg(_A=A, _B=B, _C=C) # Non-transient arrays are program inputs, mind the _ prefix

    # print("Modified Result C:")
    # print(C)
    test_with_random_input(sdfg)
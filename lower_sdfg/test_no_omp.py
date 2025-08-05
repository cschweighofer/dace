import dace
import numpy as np
import copy
import shutil

import os
os.environ['OMP_NUM_THREADS'] = '1'

def test_without_openmp():
    """Test heat_3d without OpenMP parallelization"""
    
    import heat_3d
    return heat_3d.compare_to_high_precision()

if __name__ == "__main__":
    shutil.rmtree('.dacecache', ignore_errors=True)
    try:
        result = test_without_openmp()
        print(f"Test result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_TYPES_H
#define __DACE_TYPES_H

#include <cstdint>
#include <complex>
#include "definitions.h"
#include "fp_types/simulated_double.h"
#include "fp_types/rational.h"

// GPU support
#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <thrust/complex.h>
    #include "cuda/multidim_gbar.cuh"

    // Workaround so that half is defined as a scalar (for reductions)
    namespace std {
        template <>
        struct is_scalar<half> : std::integral_constant<bool, true> {};
        template <>
        struct is_fundamental<half> : std::integral_constant<bool, true> {};
    }  // namespace std
#elif defined(__HIPCC__)
    #include <hip/hip_runtime.h>
    #include <hip/hip_fp16.h>
#endif

namespace dace
{
    typedef bool bool_;
    typedef int8_t  int8;
    typedef int16_t int16;
    typedef int32_t int32;
    typedef int64_t int64;
    typedef uint8_t  uint8;
    typedef uint16_t uint16;
    typedef uint32_t uint32;
    typedef uint64_t uint64;
    typedef unsigned int uint;
    typedef float float32;
    typedef double float64;

    #ifdef __CUDACC__
    typedef thrust::complex<float> complex64;
    typedef thrust::complex<double> complex128;
    typedef half float16;
    #elif defined(__HIPCC__)
    typedef half float16;
    #else
    typedef std::complex<float> complex64;
    typedef std::complex<double> complex128;
    struct half {
        // source: https://stackoverflow.com/a/26779139/15853075
        half(float f) {
            uint32_t x = *((uint32_t*)&f);
            h = ((x>>16)&0x8000)|((((x&0x7f800000)-0x38000000)>>13)&0x7c00)|((x>>13)&0x03ff);
        }
        operator float() {
            float f = ((h&0x8000)<<16) | (((h&0x7c00)+0x1C000)<<13) | ((h&0x03FF)<<13);
            return f;
        }
        uint16_t h;
    };
    typedef half float16;
    #endif

    enum NumAccesses
    {
        NA_RUNTIME = 0, // Given at runtime
    };

    template <int DIM, int... OTHER_DIMS>
    struct TotalNDSize
    {
	enum
	{
	    value = DIM * TotalNDSize<OTHER_DIMS...>::value,
	};
    };

    template <int DIM>
    struct TotalNDSize<DIM>
    {
	enum
	{
	    value = DIM,
	};
    };

    // Mirror of dace.dtypes.ReductionType
    enum class ReductionType {
        Custom = 0,
        Min = 1,
        Max = 2,
        Sum = 3,
        Product = 4,
        Logical_And = 5,
        Bitwise_And = 6,
        Logical_Or = 7,
        Bitwise_Or = 8,
        Logical_Xor = 9,
        Bitwise_Xor = 10,
        Min_Location = 11,
        Max_Location = 12,
        Exchange = 13
    };
}

#endif  // __DACE_TYPES_H

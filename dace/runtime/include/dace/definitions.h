#pragma once

#ifdef _MSC_VER
    //#define DACE_ALIGN(N) __declspec( align(N) )
    #define DACE_ALIGN(N)
    #undef __in
    #undef __inout
    #undef __out
    #define DACE_EXPORTED extern "C" __declspec(dllexport)
    #define DACE_PRAGMA(x) __pragma(x)
#else
    #define DACE_ALIGN(N) __attribute__((aligned(N)))
    #define DACE_EXPORTED extern "C"
    #define DACE_PRAGMA(x) _Pragma(#x)
#endif

// Visual Studio (<=2017) + CUDA support
#if defined(_MSC_VER) && _MSC_VER <= 1999
#define DACE_CONSTEXPR
#else
#define DACE_CONSTEXPR constexpr
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
    #define DACE_HDFI __host__ __device__ __forceinline__
    #define DACE_HFI __host__ __forceinline__
    #define DACE_DFI __device__ __forceinline__
#else
    #define DACE_HDFI inline
    #define DACE_HFI inline
    #define DACE_DFI inline
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
    #define __DACE_UNROLL DACE_PRAGMA(unroll)
#else
    #define __DACE_UNROLL
#endif

// If CUDA version is 11.4 or higher, __device__ variables can be declared as constexpr
#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 4))
    #define DACE_CONSTEXPR_HOSTDEV constexpr __host__ __device__
#elif defined(__CUDACC__) || defined(__HIPCC__)
    #define DACE_CONSTEXPR_HOSTDEV const __host__ __device__
#else
    #define DACE_CONSTEXPR_HOSTDEV const
#endif

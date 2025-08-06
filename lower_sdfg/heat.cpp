#include <dace/dace.h>
#include ".dacecache/auto_opt/include/hash.h"
#include <iostream>
#include <cmath>
#include <chrono>

struct auto_opt_state_t {
    dace::mpf_class * __restrict__ __0___tmp19;
    dace::mpf_class * __restrict__ __0___tmp24;
};

void __program_auto_opt_internal(auto_opt_state_t*__state, double * __restrict__ _A, double * __restrict__ _B, int64_t N, int64_t TSTEPS)
{
    dace::mpf_class *A;
    A = new dace::mpf_class DACE_ALIGN(64)[((N * N) * N)];
    dace::mpf_class *B;
    B = new dace::mpf_class DACE_ALIGN(64)[((N * N) * N)];
    int64_t t;

    {

        {
            #pragma omp parallel for
            for (auto i0 = 0; i0 < N; i0 += 1) {
                for (auto i1 = 0; i1 < N; i1 += 1) {
                    for (auto i2 = 0; i2 < N; i2 += 1) {
                        {
                            double in = _B[((((N * N) * i0) + (N * i1)) + i2)];
                            dace::mpf_class out;

                            ///////////////////
                            out = static_cast<dace::mpf_class>(in);
                            ///////////////////

                            B[((((N * N) * i0) + (N * i1)) + i2)] = out;
                        }
                    }
                }
            }
        }
        {
            #pragma omp parallel for
            for (auto i0 = 0; i0 < N; i0 += 1) {
                for (auto i1 = 0; i1 < N; i1 += 1) {
                    for (auto i2 = 0; i2 < N; i2 += 1) {
                        {
                            double in = _A[((((N * N) * i0) + (N * i1)) + i2)];
                            dace::mpf_class out;

                            ///////////////////
                            out = static_cast<dace::mpf_class>(in);
                            ///////////////////

                            A[((((N * N) * i0) + (N * i1)) + i2)] = out;
                        }
                    }
                }
            }
        }

    }
    for (t = 1; (t < TSTEPS); t = (t + 1)) {
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < (N - 2); __i0 += 1) {
                    for (auto __i1 = 0; __i1 < (N - 2); __i1 += 1) {
                        for (auto __i2 = 0; __i2 < (N - 2); __i2 += 1) {
                            dace::mpf_class __s0_n2OUT___tmp0_n3None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n4OUT___tmp1_n5None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n5OUT___tmp2_n6None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n6OUT___tmp3_n7None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n8OUT___tmp4_n9None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n9OUT___tmp5_n10None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n10OUT___tmp6_n11None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n11OUT___tmp7_n12None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n11OUT___tmp8_n12None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n13OUT___tmp9_n14None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n14OUT___tmp10_n15None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n15OUT___tmp11_n16None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n16OUT___tmp12_n17None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n16OUT___tmp13_n17None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n17OUT___tmp14_n18None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n24OUT_1_n17None[1]  DACE_ALIGN(64);
                            {
                                dace::mpf_class __in2 = A[(((((N * N) * (__i0 + 1)) + (N * (__i1 + 1))) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (2.0 * __in2);
                                ///////////////////

                                __s0_n13OUT___tmp9_n14None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = __s0_n13OUT___tmp9_n14None[0];
                                dace::mpf_class __in1 = A[(((((N * N) * (__i0 + 1)) + (N * (__i1 + 1))) + __i2) + 2)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Sub_)
                                __out = (__in1 - __in2);
                                ///////////////////

                                __s0_n14OUT___tmp10_n15None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = __s0_n14OUT___tmp10_n15None[0];
                                dace::mpf_class __in2 = A[((((N * N) * (__i0 + 1)) + (N * (__i1 + 1))) + __i2)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __s0_n15OUT___tmp11_n16None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = __s0_n15OUT___tmp11_n16None[0];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (0.125 * __in2);
                                ///////////////////

                                __s0_n16OUT___tmp12_n17None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = A[(((((N * N) * (__i0 + 1)) + (N * (__i1 + 1))) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (2.0 * __in2);
                                ///////////////////

                                __s0_n8OUT___tmp4_n9None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = __s0_n8OUT___tmp4_n9None[0];
                                dace::mpf_class __in1 = A[(((((N * N) * (__i0 + 1)) + (N * (__i1 + 2))) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Sub_)
                                __out = (__in1 - __in2);
                                ///////////////////

                                __s0_n9OUT___tmp5_n10None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = __s0_n9OUT___tmp5_n10None[0];
                                dace::mpf_class __in2 = A[(((((N * N) * (__i0 + 1)) + (N * __i1)) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __s0_n10OUT___tmp6_n11None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = __s0_n10OUT___tmp6_n11None[0];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (0.125 * __in2);
                                ///////////////////

                                __s0_n11OUT___tmp7_n12None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = A[(((((N * N) * (__i0 + 1)) + (N * (__i1 + 1))) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (2.0 * __in2);
                                ///////////////////

                                __s0_n2OUT___tmp0_n3None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = __s0_n2OUT___tmp0_n3None[0];
                                dace::mpf_class __in1 = A[(((((N * N) * (__i0 + 2)) + (N * (__i1 + 1))) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Sub_)
                                __out = (__in1 - __in2);
                                ///////////////////

                                __s0_n4OUT___tmp1_n5None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = __s0_n4OUT___tmp1_n5None[0];
                                dace::mpf_class __in2 = A[(((((N * N) * __i0) + (N * (__i1 + 1))) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __s0_n5OUT___tmp2_n6None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = __s0_n5OUT___tmp2_n6None[0];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (0.125 * __in2);
                                ///////////////////

                                __s0_n6OUT___tmp3_n7None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = __s0_n6OUT___tmp3_n7None[0];
                                dace::mpf_class __in2 = __s0_n11OUT___tmp7_n12None[0];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __s0_n11OUT___tmp8_n12None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = __s0_n11OUT___tmp8_n12None[0];
                                dace::mpf_class __in2 = __s0_n16OUT___tmp12_n17None[0];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __s0_n16OUT___tmp13_n17None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = __s0_n16OUT___tmp13_n17None[0];
                                dace::mpf_class __in2 = A[(((((N * N) * (__i0 + 1)) + (N * (__i1 + 1))) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __s0_n17OUT___tmp14_n18None[0] = __out;
                            }

                            dace::CopyND<dace::mpf_class, 1, false, 1>::template ConstDst<1>::Copy(
                            __s0_n17OUT___tmp14_n18None, __s0_n24OUT_1_n17None, 1);

                            dace::CopyND<dace::mpf_class, 1, false, 1>::template ConstDst<1>::Copy(
                            __s0_n24OUT_1_n17None, B + (((((N * N) * (__i0 + 1)) + (N * (__i1 + 1))) + __i2) + 1), 1);
                            {
                                dace::mpf_class __in2 = __s0_n24OUT_1_n17None[0];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (2.0 * __in2);
                                ///////////////////

                                __state->__0___tmp24[((((__i0 * (N - 2)) * (N - 2)) + (__i1 * (N - 2))) + __i2)] = __out;
                            }
                            {
                                dace::mpf_class __in2 = __s0_n17OUT___tmp14_n18None[0];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (2.0 * __in2);
                                ///////////////////

                                __state->__0___tmp19[((((__i0 * (N - 2)) * (N - 2)) + (__i1 * (N - 2))) + __i2)] = __out;
                            }
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < (N - 2); __i0 += 1) {
                    for (auto __i1 = 0; __i1 < (N - 2); __i1 += 1) {
                        for (auto __i2 = 0; __i2 < (N - 2); __i2 += 1) {
                            dace::mpf_class __s0_n20OUT___tmp15_n21None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n21OUT___tmp16_n22None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n22OUT___tmp17_n23None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n23OUT___tmp18_n24None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n27OUT___tmp20_n28None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n28OUT___tmp21_n29None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n29OUT___tmp22_n30None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n29OUT___tmp23_n30None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n34OUT___tmp25_n35None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n35OUT___tmp26_n36None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n36OUT___tmp27_n37None[1]  DACE_ALIGN(64);
                            dace::mpf_class __s0_n36OUT___tmp28_n37None[1]  DACE_ALIGN(64);
                            {
                                dace::mpf_class __in1 = B[(((((N * N) * (__i0 + 1)) + (N * (__i1 + 1))) + __i2) + 2)];
                                dace::mpf_class __in2 = __state->__0___tmp24[((((__i0 * (N - 2)) * (N - 2)) + (__i1 * (N - 2))) + __i2)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Sub_)
                                __out = (__in1 - __in2);
                                ///////////////////

                                __s0_n34OUT___tmp25_n35None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = __s0_n34OUT___tmp25_n35None[0];
                                dace::mpf_class __in2 = B[((((N * N) * (__i0 + 1)) + (N * (__i1 + 1))) + __i2)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __s0_n35OUT___tmp26_n36None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = __s0_n35OUT___tmp26_n36None[0];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (0.125 * __in2);
                                ///////////////////

                                __s0_n36OUT___tmp27_n37None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = B[(((((N * N) * (__i0 + 1)) + (N * (__i1 + 2))) + __i2) + 1)];
                                dace::mpf_class __in2 = __state->__0___tmp19[((((__i0 * (N - 2)) * (N - 2)) + (__i1 * (N - 2))) + __i2)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Sub_)
                                __out = (__in1 - __in2);
                                ///////////////////

                                __s0_n27OUT___tmp20_n28None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = __s0_n27OUT___tmp20_n28None[0];
                                dace::mpf_class __in2 = B[(((((N * N) * (__i0 + 1)) + (N * __i1)) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __s0_n28OUT___tmp21_n29None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = __s0_n28OUT___tmp21_n29None[0];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (0.125 * __in2);
                                ///////////////////

                                __s0_n29OUT___tmp22_n30None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = B[(((((N * N) * (__i0 + 1)) + (N * (__i1 + 1))) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (2.0 * __in2);
                                ///////////////////

                                __s0_n20OUT___tmp15_n21None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = __s0_n20OUT___tmp15_n21None[0];
                                dace::mpf_class __in1 = B[(((((N * N) * (__i0 + 2)) + (N * (__i1 + 1))) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Sub_)
                                __out = (__in1 - __in2);
                                ///////////////////

                                __s0_n21OUT___tmp16_n22None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = __s0_n21OUT___tmp16_n22None[0];
                                dace::mpf_class __in2 = B[(((((N * N) * __i0) + (N * (__i1 + 1))) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __s0_n22OUT___tmp17_n23None[0] = __out;
                            }
                            {
                                dace::mpf_class __in2 = __s0_n22OUT___tmp17_n23None[0];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (0.125 * __in2);
                                ///////////////////

                                __s0_n23OUT___tmp18_n24None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = __s0_n23OUT___tmp18_n24None[0];
                                dace::mpf_class __in2 = __s0_n29OUT___tmp22_n30None[0];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __s0_n29OUT___tmp23_n30None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = __s0_n29OUT___tmp23_n30None[0];
                                dace::mpf_class __in2 = __s0_n36OUT___tmp27_n37None[0];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __s0_n36OUT___tmp28_n37None[0] = __out;
                            }
                            {
                                dace::mpf_class __in1 = __s0_n36OUT___tmp28_n37None[0];
                                dace::mpf_class __in2 = B[(((((N * N) * (__i0 + 1)) + (N * (__i1 + 1))) + __i2) + 1)];
                                dace::mpf_class __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                A[(((((N * N) * (__i0 + 1)) + (N * (__i1 + 1))) + __i2) + 1)] = __out;
                            }
                        }
                    }
                }
            }

        }

    }
    {

        {
            #pragma omp parallel for
            for (auto i0 = 0; i0 < N; i0 += 1) {
                for (auto i1 = 0; i1 < N; i1 += 1) {
                    for (auto i2 = 0; i2 < N; i2 += 1) {
                        {
                            dace::mpf_class in = B[((((N * N) * i0) + (N * i1)) + i2)];
                            double out;

                            ///////////////////
                            out = static_cast<double>(in);
                            ///////////////////

                            _B[((((N * N) * i0) + (N * i1)) + i2)] = out;
                        }
                    }
                }
            }
        }
        {
            #pragma omp parallel for
            for (auto i0 = 0; i0 < N; i0 += 1) {
                for (auto i1 = 0; i1 < N; i1 += 1) {
                    for (auto i2 = 0; i2 < N; i2 += 1) {
                        {
                            dace::mpf_class in = A[((((N * N) * i0) + (N * i1)) + i2)];
                            double out;

                            ///////////////////
                            out = static_cast<double>(in);
                            ///////////////////

                            _A[((((N * N) * i0) + (N * i1)) + i2)] = out;
                        }
                    }
                }
            }
        }

    }
    delete[] A;
    delete[] B;
}

DACE_EXPORTED void __program_auto_opt(auto_opt_state_t *__state, double * __restrict__ _A, double * __restrict__ _B, int64_t N, int64_t TSTEPS)
{
    __program_auto_opt_internal(__state, _A, _B, N, TSTEPS);
}

DACE_EXPORTED auto_opt_state_t *__dace_init_auto_opt(int64_t N)
{
    int __result = 0;
    auto_opt_state_t *__state = new auto_opt_state_t;


    __state->__0___tmp19 = new dace::mpf_class DACE_ALIGN(64)[(((N - 2) * (N - 2)) * (N - 2))];
    __state->__0___tmp24 = new dace::mpf_class DACE_ALIGN(64)[(((N - 2) * (N - 2)) * (N - 2))];

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED int __dace_exit_auto_opt(auto_opt_state_t *__state)
{
    int __err = 0;
    delete[] __state->__0___tmp19;
    delete[] __state->__0___tmp24;
    delete __state;
    return __err;
}

// Function to initialize arrays based on the Python initialization
void initialize(double* A, double* B, int64_t N) {
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < N; j++) {
            for (int64_t k = 0; k < N; k++) {
                int64_t idx = ((N * N) * i) + (N * j) + k;
                double value = (i + j + (N - k)) * 10.0 / N;
                A[idx] = value;
                B[idx] = value;
            }
        }
    }
}

// Print a slice of the 3D array for verification
void print_slice(const double* arr, int64_t N, int64_t slice_i, const char* name) {
    std::cout << name << " slice [" << slice_i << ",:,:] (first 5x5):" << std::endl;
    for (int64_t j = 0; j < std::min((int64_t)5, N); j++) {
        for (int64_t k = 0; k < std::min((int64_t)5, N); k++) {
            int64_t idx = ((N * N) * slice_i) + (N * j) + k;
            std::cout << arr[idx] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    const int64_t N = 5;  // Reduced size for debugging
    const int64_t TSTEPS = 3;  // Reduced steps for debugging
    
    std::cout << "Initializing 3D Heat Equation Solver (Debug Mode)" << std::endl;
    std::cout << "Grid size: " << N << "x" << N << "x" << N << std::endl;
    std::cout << "Time steps: " << TSTEPS << std::endl;
    
    // Allocate arrays
    double* A = new double[N * N * N];
    double* B = new double[N * N * N];
    
    // Initialize arrays
    initialize(A, B, N);
    
    std::cout << "Arrays initialized." << std::endl;
    
    // Print initial state
    print_slice(A, N, 0, "Initial A");
    print_slice(B, N, 0, "Initial B");
    
    // Initialize DaCe state
    std::cout << "Initializing DaCe state..." << std::endl;
    auto_opt_state_t* state = __dace_init_auto_opt(N);
    if (!state) {
        std::cerr << "Failed to initialize DaCe state!" << std::endl;
        delete[] A;
        delete[] B;
        return 1;
    }
    std::cout << "DaCe state initialized successfully." << std::endl;
    
    std::cout << "Starting computation..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run the computation
    std::cout << "Calling DaCe computation..." << std::endl;
    __program_auto_opt(state, A, B, N, TSTEPS);
    std::cout << "DaCe computation completed." << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Computation completed in " << duration.count() << " ms" << std::endl;
    
    // Print final state
    print_slice(A, N, 0, "Final A");
    print_slice(B, N, 0, "Final B");
    
    // Calculate some statistics
    double sum_A = 0.0, sum_B = 0.0;
    double min_A = A[0], max_A = A[0];
    double min_B = B[0], max_B = B[0];
    
    for (int64_t i = 0; i < N * N * N; i++) {
        sum_A += A[i];
        sum_B += B[i];
        
        if (A[i] < min_A) min_A = A[i];
        if (A[i] > max_A) max_A = A[i];
        if (B[i] < min_B) min_B = B[i];
        if (B[i] > max_B) max_B = B[i];
    }
    
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "A: sum=" << sum_A << ", min=" << min_A << ", max=" << max_A << std::endl;
    std::cout << "B: sum=" << sum_B << ", min=" << min_B << ", max=" << max_B << std::endl;
    
    // Clean up
    std::cout << "Cleaning up DaCe state..." << std::endl;
    __dace_exit_auto_opt(state);
    std::cout << "DaCe state cleaned up." << std::endl;
    
    std::cout << "Cleaning up arrays..." << std::endl;
    delete[] A;
    delete[] B;
    std::cout << "Arrays cleaned up." << std::endl;
    
    std::cout << "Program completed successfully." << std::endl;
    return 0;
}

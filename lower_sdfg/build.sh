#!/bin/bash

# Build script for heat.cpp with DaCe runtime from source tree
# Usage:
#   ./build.sh               - Normal build
#   ./build.sh run           - Build and run
#   ./build.sh --scan-build  - Build with Clang Static Analyzer
#   USE_SCAN_BUILD=true ./build.sh - Use scan-build via environment variable

set -e

# Use the DaCe runtime from source tree
RUNTIME_DIR="../dace/runtime/include"

echo "Using DaCe runtime from source tree: $RUNTIME_DIR"

# Check if the DaCe header exists
if [ ! -f "$RUNTIME_DIR/dace/dace.h" ]; then
    echo "Error: Could not find dace.h at $RUNTIME_DIR/dace/dace.h"
    echo "Current directory: $(pwd)"
    echo "Please ensure you're running from the lower_sdfg directory in the DaCe source tree"
    exit 1
fi

# Compiler settings
CXX=${CXX:-clang++}
CXXFLAGS="-std=c++17 -O0 -g -march=native"  # Debug build without ASAN during compilation
INCLUDES="-I$RUNTIME_DIR -I."

# Check if scan-build should be used
USE_SCAN_BUILD=${USE_SCAN_BUILD:-false}
SCAN_BUILD_CMD=""
if [ "$USE_SCAN_BUILD" = "true" ] || [ "$1" = "--scan-build" ]; then
    # Check if scan-build is available
    if command -v scan-build >/dev/null 2>&1; then
        SCAN_BUILD_CMD="scan-build -o scan-results --use-analyzer=/usr/local/bin/clang"
        echo "Using scan-build for static analysis"
        echo "Results will be saved to scan-results/"
        # Remove AddressSanitizer when using scan-build as it can interfere
        CXXFLAGS="-std=c++17 -O0 -g -march=native"
    else
        echo "Warning: scan-build not found, falling back to regular compilation"
        echo "Install clang-tools package to use scan-build"
    fi
fi

# Try to detect OpenMP library and add GMP library
# AddressSanitizer will be added at linking stage only
# ASAN_FLAGS="-fsanitize=address"
ASAN_FLAGS=""

if [ -f "/usr/lib/x86_64-linux-gnu/libomp.so" ] || [ -f "/usr/lib64/libomp.so" ]; then
    LIBS="$ASAN_FLAGS -lgmp -lm"  # Removed OpenMP for debugging
elif [ -f "/usr/lib/x86_64-linux-gnu/libgomp.so" ] || [ -f "/usr/lib64/libgomp.so" ]; then
    LIBS="$ASAN_FLAGS -lgmp -lm"  # Removed OpenMP for debugging
else
    echo "Warning: OpenMP library not found, compiling without OpenMP"
    LIBS="$ASAN_FLAGS -lgmp -lm"
fi

echo "Compiling heat.cpp..."
echo "Compiler: $CXX"
echo "Flags: $CXXFLAGS"
echo "Includes: $INCLUDES"
echo "Libraries: $LIBS"
if [ -n "$SCAN_BUILD_CMD" ]; then
    echo "Using scan-build: $SCAN_BUILD_CMD"
fi

# Compile with or without scan-build
if [ -n "$SCAN_BUILD_CMD" ]; then
    $SCAN_BUILD_CMD $CXX $CXXFLAGS $INCLUDES -o heat_solver heat.cpp $LIBS
else
    $CXX $CXXFLAGS $INCLUDES -o heat_solver heat.cpp $LIBS
fi

echo "Build successful! Executable: heat_solver"

# Optionally run if requested
if [ "$1" = "run" ] || [ "$1" = "--run" ]; then
    echo ""
    echo "Running heat_solver..."
    ./heat_solver
elif [ "$1" = "--scan-build" ]; then
    echo ""
    echo "Static analysis completed. Check scan-results/ directory for reports."
    echo "To view the results, open scan-results/*/index.html in a web browser."
fi


# #!/bin/bash

# # Build script for heat.cpp with DaCe runtime from source tree

# set -e

# # Use the DaCe runtime from source tree
# RUNTIME_DIR="../dace/runtime/include"

# echo "Using DaCe runtime from source tree: $RUNTIME_DIR"

# # Check if the DaCe header exists
# if [ ! -f "$RUNTIME_DIR/dace/dace.h" ]; then
#     echo "Error: Could not find dace.h at $RUNTIME_DIR/dace/dace.h"
#     echo "Current directory: $(pwd)"
#     echo "Please ensure you're running from the lower_sdfg directory in the DaCe source tree"
#     exit 1
# fi

# # Compiler settings
# CXX=${CXX:-clang++}
# CXXFLAGS="-std=c++17 -O3 -fopenmp -march=native"
# INCLUDES="-I$RUNTIME_DIR -I."

# # Try to detect OpenMP library and add GMP library
# if [ -f "/usr/lib/x86_64-linux-gnu/libomp.so" ] || [ -f "/usr/lib64/libomp.so" ]; then
#     LIBS="-fopenmp -lgmp -lm"
# elif [ -f "/usr/lib/x86_64-linux-gnu/libgomp.so" ] || [ -f "/usr/lib64/libgomp.so" ]; then
#     LIBS="-lgomp -lgmp -lm"
#     CXXFLAGS="-std=c++17 -O3 -fopenmp -march=native"
# else
#     echo "Warning: OpenMP library not found, compiling without OpenMP"
#     CXXFLAGS="-std=c++17 -O3 -march=native"
#     LIBS="-lgmp -lm"
# fi

# echo "Compiling heat.cpp..."
# echo "Compiler: $CXX"
# echo "Flags: $CXXFLAGS"
# echo "Includes: $INCLUDES"

# # Compile
# $CXX $CXXFLAGS $INCLUDES -o heat_solver heat.cpp $LIBS

# echo "Build successful! Executable: heat_solver"

# # Optionally run if requested
# if [ "$1" = "run" ] || [ "$1" = "--run" ]; then
#     echo ""
#     echo "Running heat_solver..."
#     ./heat_solver
# fi


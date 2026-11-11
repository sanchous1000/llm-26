# CUDA build environment for lab2 (variant sort).
# Source this file in each new terminal before running cmake:
#     source scripts/cuda_env.sh
#
# Environment facts this fixes:
#   * nvcc (CUDA 12.2) is installed at /usr/local/cuda-12.2 but is NOT on PATH.
#   * CUDA 12.2 supports host g++ up to 12.2; system g++ is 13.3 (unsupported).
#   * A compatible gcc/g++ 12.2.0 is installed in the conda env `cudabuild`.
#
# Because the lab cmake command does not pass -DCMAKE_CUDA_HOST_COMPILER,
# a `cmake` shim is placed early on PATH that injects that flag automatically,
# so nvcc uses g++ 12.2 instead of the unsupported system g++ 13.3.

basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. CUDA toolkit: nvcc, CUDA runtime, Thrust.
export CUDA_HOME=/usr/local/cuda-12.2
export PATH="$CUDA_HOME/bin:$PATH"

# 2. Compatible host compiler (gcc/g++ 12.2.0) from the conda env.
CUDABUILD_HOME=/opt/conda/envs/cudabuild
GCC12_BIN="$CUDABUILD_HOME/bin"
if [ ! -x "$GCC12_BIN/x86_64-conda-linux-gnu-g++" ]; then
    echo "ERROR: gcc 12.2 not found at $GCC12_BIN." >&2
    echo "Install it once with:" >&2
    echo "  conda create -y -n cudabuild -c conda-forge gcc_linux-64=12.2.0 gxx_linux-64=12.2.0" >&2
    return 1 2>/dev/null || exit 1
fi
export CMAKE_CUDA_HOST_COMPILER="$GCC12_BIN/x86_64-conda-linux-gnu-g++"
# Also make plain gcc/g++ resolve to 12.2 for any non-CUDA host C++ step.
export CC="$GCC12_BIN/x86_64-conda-linux-gnu-gcc"
export CXX="$GCC12_BIN/x86_64-conda-linux-gnu-g++"

# 3. cmake shim: inject CMAKE_CUDA_HOST_COMPILER so the lab's plain
#    `cmake -S nvidia -B build ...` command uses the compatible compiler.
#    Capture the REAL cmake before the shim goes onto PATH to avoid recursion.
REAL_CMAKE="$(which cmake)"
SHIM_DIR="$basedir/scripts/.cmake-shim"
mkdir -p "$SHIM_DIR"
CC_ARG="-DCMAKE_CUDA_HOST_COMPILER=$CMAKE_CUDA_HOST_COMPILER"
cat > "$SHIM_DIR/cmake" <<EOF
#!/usr/bin/env bash
real_cmake="$REAL_CMAKE"
for a in "\$@"; do
  # Already has the flag -> pass through untouched.
  if [ "\$a" = "-DCMAKE_CUDA_HOST_COMPILER" ] || [[ "\$a" == -DCMAKE_CUDA_HOST_COMPILER=* ]]; then
    exec "\$real_cmake" "\$@"
  fi
  # Build/install/preset steps are configured already -> pass through,
  # injecting -D before --build breaks cmake 4.x.
  if [ "\$a" = "--build" ] || [ "\$a" = "--install" ] || [ "\$a" = "--preset" ] \
     || [[ "\$a" == --build=* ]] || [[ "\$a" == --install=* ]]; then
    exec "\$real_cmake" "\$@"
  fi
done
exec "\$real_cmake" "$CC_ARG" "\$@"
EOF
chmod +x "$SHIM_DIR/cmake"
export PATH="$SHIM_DIR:$PATH"

# 4. Lab variables for the sort variant and L40S (compute capability 8.9).
export ALG=sort
export ARCH=89

echo "CUDA env ready:"
echo "  nvcc : $(command -v nvcc) ($(nvcc --version | sed -n '4s/^ *release /*/p'))"
echo "  g++  : $CXX"
echo "  cmake: $(command -v cmake) (shim injects CMAKE_CUDA_HOST_COMPILER)"
echo "  ALG=$ALG ARCH=$ARCH"
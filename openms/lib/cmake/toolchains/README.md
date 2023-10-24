# Summary
This is a collection of toolchains for generic and specific high-end platform instances.

# Generic platforms

The following toolchains are provided:
- `clang-mpi-mkl-tbb`: generic clang + generic MPI + Intel MKL + Intel TBB
- `gcc-mpi-mkl-tbb`: gcc + generic MPI + Intel MKL + Intel TBB
- `intel-parallel-studio`: Intel Parallel Studio
- `intel-oneapi`: Intel OneAPI kit (base + HPC)
- `macos-clang-mpi-accelerate`: MacOS Clang + generic MPI + Accelerate
- `macos-gcc-mpi-accelerate`: MacOS Clang + generic MPI + Accelerate

# Specific Platforms

## OLCF Summit

See `olcf-summit-gcc-essl.cmake`. Recommended configure scripts are can be found in `<project_src_dir>/cmake/toolchains/README.md`.

## ALCF Theta

See instructions in the toolchain file `alcf-theta-mkl-tbb.cmake` (contributed by @victor-anisimov ). This should work for other generic x86-based platforms with Cray compiler wrappers.

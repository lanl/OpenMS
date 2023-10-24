#!/bin/sh

if [ ! -d "build" ]; then
	mkdir build
fi

export EIGEN3_PATH=/path/to/eigen-3.3.9

cmake -B build \
        -DCMAKE_PREFIX_PATH=/path/to/libs/other/than/normal/ones/ \
	-DENABLE_MPI=ON \
	-DENABLE_MEEP=ON \
	-DENABLE_TA=OFF \
	-DENABLE_TASCALAPACK=ON \
	-DENABLE_TACUDA=OFF \
	-DENABLE_TAPYTHON=ON \
	-DENABLE_TBB=ON \
	-DCMAKE_TOOLCHAIN_FILE=./cmake/toolchains/gcc-mpi-mkl-tbb.cmak

cd build
make 
cd ..


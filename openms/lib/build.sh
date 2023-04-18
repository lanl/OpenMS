#!/bin/sh

if [ ! -d "build" ]; then

	mkdir build
fi

cmake -DPython3_EXECUTABLE=/path/to/python/
	-DEIGEN3_PATH=/path/to/eigen3/ \
	-Dtoolchainpath=/path/to/toolchian/ \
	-B build

cd build
make 
cd ..


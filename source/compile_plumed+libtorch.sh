#!/bin/bash

FOLDER=${PWD}

# download PLUMED
wget -q --show-progress https://github.com/plumed/plumed2/archive/refs/tags/v2.8.1.zip && unzip -q v2.8.1.zip && rm v2.8.1.zip && mv plumed* plumed
# newest commits on master branch break compatibility with lammps: use older branch

# download libtorch
wget -q --show-progress https://download.pytorch.org/libtorch/lts/1.8/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.2%2Bcpu.zip && unzip -q libtorch* && rm libtorch-cxx11*

# path
export LIBTORCH=${FOLDER}/libtorch
export CPATH=${LIBTORCH}/include/torch/csrc/api/include/:${LIBTORCH}/include/:${LIBTORCH}/include/torch:$CPATH
export INCLUDE=${LIBTORCH}/include/torch/csrc/api/include/:${LIBTORCH}/include/:${LIBTORCH}/include/torch:$INCLUDE
export LIBRARY_PATH=${LIBTORCH}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# Replace Custom.cpp
cd plumed
if [ ! -f "./origCustom.cpp" ]
then
cp ./src/function/Custom.cpp ./origCustom.cpp
fi
cp ${FOLDER}/myCustom.cpp    ./src/function/Custom.cpp

mkdir build
./configure --enable-libtorch --enable-modules=all CXX=mpicxx --prefix=${FOLDER}/plumed/build

# The following is needed because LD_RO does not change in Makefile.conf
sed -i 's/LD_RO=ld -r -o/LD_RO=\/usr\/bin\/ld -r -o/' Makefile.conf

make --silent -j 8
make --silent -j 8 install

cd $FOLDER

# Libtorch
export LIBTORCH=./libtorch

export CPATH=${LIBTORCH}/include/torch/csrc/api/include/:${LIBTORCH}/include/:${LIBTORCH}/include/torch:$CPATH
export INCLUDE=${LIBTORCH}/include/torch/csrc/api/include/:${LIBTORCH}/include/:${LIBTORCH}/include/torch:$INCLUDE
export LIBRARY_PATH=${LIBTORCH}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# Plumed
export PLUMED_ROOT=./plumed
source ${PLUMED_ROOT}sourceme.sh

export PATH=${PLUMED_ROOT}/src/lib/:$PATH
export CPATH=${PLUMED_ROOT}/include:$CPATH
export INCLUDE=${PLUMED_ROOT}/include:$INCLUDE
export LIBRARY_PATH=${PLUMED_ROOT}/src/lib/:$LIBRARY_PATH
export LD_LIBRARY_PATH=${PLUMED_ROOT}/src/lib/:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=${PLUMED_ROOT}/lib/:$DYLD_LIBRARY_PATH
export PKG_CONFIG_PATH=${PLUMED_ROOT}/lib/pkgconfig:$PKG_CONFIG_PATH
export PLUMED_KERNEL=${PLUMED_ROOT}/src/lib/libplumedKernel.so
export PLUMED_VIMPATH=${PLUMED_ROOT}/vim
export PYTHONPATH=${PLUMED_ROOT}/python:$PYTHONPATH

# Lammps
export LAMMPS_SRC=./lammps/src/
export PATH=$LAMMPS_SRC:$PATH

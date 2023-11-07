#!/bin/bash

export FOLDER=${PWD}

# store custom fixes
#cp lammps/src/fix_path_dynamics.* ./
#cp lammps/src/fix_path_dynamics_mb.* ./
#cp lammps/src/fix_tps.* ./

# opt: remove old version
# rm -r lammps

# download LAMMPS
wget -q --show-progress https://download.lammps.org/tars/lammps-stable.tar.gz && tar -xzf lammps* && rm lammps-stable.tar.gz && mv lammps* lammps

source lammps+plumed+libtorch.sh

cd lammps/src

# MetaD of Paths patch by Davide Mandelli
#cp ${FOLDER}/fix_path_dynamics.* ./MISC/
#sed -i 's/forward_comm_fix/forward_comm/g' ./MISC/fix_path_dynamics.cpp
cp $FOLDER/fix_path_dynamics.* ./MISC/
cp $FOLDER/fix_path_dynamics_mb.* ./MISC/

# openmp support
sed -i '/^CCFLAGS/ s/$/ -fopenmp/' ./MAKE/OPTIONS/Makefile.g++_openmpi
sed -i '/^LINKFLAGS/ s/$/ -fopenmp/' ./MAKE/OPTIONS/Makefile.g++_openmpi
make yes-openmp

make yes-extra-fix
make yes-misc
make yes-reaxff
make yes-molecule #needed to enable "atom_style full" (for graphene h-BN)
make yes-manybody #enables use of manybody potentials (e.g., REBO)
make yes-kspace
make yes-phonon
make yes-rigid
make yes-replica #needed for NEB

make clean-all
make -j 8 g++_openmpi

cd ../lib/plumed
python3 Install.py -p ${FOLDER}/plumed/build -m shared
cd -
make yes-plumed
make -j 8 g++_openmpi

/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(path_dynamics,FixPATH_DYNAMICS)

#else

#ifndef FIX_PATH_DYNAMICS_H
#define FIX_PATH_DYNAMICS_H

#include "fix.h"

namespace LAMMPS_NS {

class FixPATH_DYNAMICS : public Fix {
 public:
  FixPATH_DYNAMICS(class LAMMPS *, int, char **);
  //virtual ~FixPATH_DYNAMICS();

  int setmask();

  void init();
  void setup(int);
  void post_force(int);
  void initial_integrate(int);
  void final_integrate();

  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int,int,int);
  int pack_exchange(int,double*);
  int unpack_exchange(int,double*);
  int pack_restart(int,double*);
  void unpack_restart(int,int);
  int maxsize_restart();
  int size_restart(int);
  double compute_vector(int);
  double compute_scalar();

  int pack_forward_comm(int, int*, double *, int, int*);
  void unpack_forward_comm(int, int, double *);

  int nbeads,ibead,np_u,np_w,me_u,me_w;
  MPI_Comm uworld;

  /* run type */
  int run;
  int min_style; //minimization style of neb

  /* finite difference */
  double epsilon;

  /* Brownian level */
  double dt_path;
  double damp_path;
  double *md_2dt;
  double *dt_md;

  /* Polymer model */

  void pimd_fill(double**);

  double spring_energy;
  double kinetic_energy;
  double potential_energy; //force-field
  int x_last, x_next;

  void spring_force();
  
  //NEB
  void neb_force();
  void neb_test_force();
  double veng,plen,nlen,tlen,dotpath,dot;
  double lentot,lenuntilIm,meanDist;
  double **tangent,**SpringFPerp;
  int climber;
  MPI_Comm rootworld;

  double kspringPerp,kspringPara;
  double kspringIni,kspringFinal;
  double EIniIni,EFinalIni;
  bool FreeEndIni,FreeEndFinal;
  bool FreeEndFinalWithRespToEIni,FinalAndInterWithRespToEIni;
  bool PerpSpring,ParaSpring;
  //NEB

  /* fictious mass */

  double *fmass;

  /* inter-partition communication */

  int max_nsend;
  int* tag_send;
  double *buf_send;

  int max_nlocal;
  double *buf_recv, **buf_beads, **buf_force;

  int size_plan;
  int *plan_send, *plan_recv;
  double **comm_ptr;

  void comm_init();
  void comm_exec(double **);

  int *mode_index;

  /* Langevin thermostat */

  double lange_tau;
  double *gfactor1,*gfactor2;
  class RanMars *random;
  int seed;  

  /* Nose-hoover chain integration */

  int nhc_offset_one_1, nhc_offset_one_2;
  int nhc_size_one_1, nhc_size_one_2;
  int nhc_nchain;
  bool nhc_ready;
  double target_temp, dtv, dtf, t_sys;

  double **nhc_eta;        /* coordinates of NH chains for ring-polymer beads */
  double **nhc_eta_dot;    /* velocities of NH chains                         */
  double **nhc_eta_dotdot; /* acceleration of NH chains                       */
  double **nhc_eta_mass;   /* mass of NH chains                               */

  void nhc_init();
  void nhc_update_v();
  void neb_update_v();

  /* Verlet */
  void update_x();
  void update_v();

  private:

  //NEB
  char *id_pe;
  class Compute *pe;
  //NEB

  virtual void force_clear();
  void compute_force();
  void compute_kinetic_energy();

  //Flags for force evaluation
  int torqueflag;
  int extraflag;
  int n_pre_reverse;
  int n_pre_force;
  int kspace_compute_flag;
  int pair_compute_flag;
  //int n_post_force; //no fixes with post_force allowed
  //Aux vectors
  double **x_bead, **f_bead, **f_fwd, **f_bwd;
  double **eta_l,**eta_n;

};


}

#endif
#endif

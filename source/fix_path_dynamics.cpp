/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Package      FixPATH_DYNAMICS
   Purpose      Classical Path Dynamics Algorithm
   Copyright    Parrinello Group @  IIT, Genova
   Authors      Davide Mandelli (davide.mandelli at iit.it)
   Updated      Jun-23-2020
   Version      1.0

Inter-partition communicator: 'universe->uworld' (see "universe.*").
universe->nworlds; (number of worlds in universe)
universe->iworld; (which world I am)
universe->me; (my rank in universe)
universe->nprocs; (size of universe)

Intra-partition communicator: 'world' (see "lammps.*").
comm->nprocs; (size of world)
comm->me; (my rank in world)

ESEMPIO: using -partition NxM

universe->nworlds = N
universe->iworld  = 0,1,2,..., N-1
universe->nprocs  = N*M
universe->me      = 0,1,2,..., N*M-1

comm->nprocs      = M
comm->me          = 0,1,2,..., M-1

For NEB a new group is created that contains only the root procs of each world.
The communicator of this group is 'rootworld' and can be used only by root process of each world (comm->me == 0).

-------------------------------------------------------------------------

fix <fix-ID> all path_dynamics <dt> <damp> <eps> [keyword <values> ...]

dt    = time step at Brownian level
damp  = damping at Brownian level
eps   = for finite difference computation of second derivatives

OSS: "dt" and "damp" define the ~spring constants~ and ~equilibrium spring lengths~ appearing in the OM action.

keywords: langevin, damped, nhc, neb

1. 'langevin' performs NVT langevin dynamics, compulsory values: temperature tau seed.
    
    'tau' is the relaxation time.
    
    es: fix 1 all path_dynamics 0.1 1.0 0.0001 langevin 300.0 1.0 2375

2) 'damp' performs damped dynamics, compulsory value: tau.
   
   es: fix 1 all path_dynamics 0.1 1.0 0.0001 damped 1.0

3) 'nhc' performs NVT run using Nosee-Hoover chains thermostat, compulsory values: temperature nchains.

   es: fix 1 all path_dynamics 0.1 1.0 0.0001 nhc 300.0 5

4) 'neb' performs minimization using forces as in nudged elastic band.
   'neb_test' same as 'neb' but I neglect the hessian part in the spring forces and I use the absolute values.

   Compulsory value: minimization_style.

   "minimization_style" can be specified as: 'sd' or 'asd' or 'damp'.
   sd     -> steepest descent.
   asd    -> accellerated sd.
   damped -> damped dynamics

   es: fix 1 all path_dynamics 0.1 1.0 0.0001 neb asd

   Optional args: perp parallel end climber

   -'climber ibead' switches on climbing image. ibead=1,..,N is the replica that will climb up.
   
   es: fix 1 all path_dynamics 0.1 1.0 0.0001 neb asd climber 6  
 
   -'perp' allows to define a spring constant different from the natural one appearing in the OM action.
    This value will be used to compute a spring force perpendicular to the path. Units: force/length. The same k_perp is used for all atoms.

    es: fix 1 all path_dynamics 0.1 1.0 0.0001 neb asd perp 1.0

    If k_perp is set to zero then the natural spring constant is used.
    If 'perp' keyword is not specified, then no perp force is applied to the band, this can lead to kinks in the path especially if total number of beads is small.

   -'parallel' allows to define a spring constant different from the natural one appearing in the OM action.
    This value will be used to compute a spring force tangent to the path. Units: force/length. The same k_paral is used for all atoms.

    es: fix 1 all path_dynamics 0.1 1.0 0.0001 neb asd parallel 1.0

    If 'parallel' keyword is not specified, then the natural spring constant appearing in the OM action will be used.

   -'end'
    In standard neb, first and last bead simply relax to local minimum. This keyword allow to modify the forces acting on the beads.

    Usage: end [key] [values] 
   
    Possible keys are: first, last, last/efirst, last/efirst/middle

    - end first [k_first]
     Modifies force on first bead to keep its energy close to that of the configuration at the very beginning of simulation.
 
    - end last [k_last]
     Modifies force on last bead to keep its energy close to that of the configuration at the very beginning of simulation.

    - end last/efirst [k_last]
     Modifies force on last bead to keep its energy above that of the first bead, as the latter during simulation.

    - end last/efirst/middle [k_last]
     Modifies force on last and middle beads to keep their energy above that of the first bead, as the latter evolves during simulation.
   
     k_last,k_first units are force/energy.
     You can specify both "end first" and one of the "end last*" if you want.
    
     es. fix 1 all path_dynamics 0.1 1.0 0.0001 neb asd parallel 1.0 perp 1.0 end first 1.0 end last 2.0
         fix 1 all path_dynamics 0.1 1.0 0.0001 neb asd parallel 1.0 perp 1.0 end first 1.0 end last/efirst/middle 2.0

DEFAULT: If no keyword is specified, an NVE run is performed.

-------------------------------------------------------------------------

TODO:
0. Only act on group of atoms (right now acts on all atoms);
1. NEB:
  -improve minimization algorithm
2. Langevin Action with Inertia
3. NPT
------------------------------------------------------------------------- */

#include <cmath>
#include <cstring>
#include <cstdlib>
#include "fix_path_dynamics.h"
#include "universe.h"
#include "comm.h"
#include "force.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "memory.h"
#include "error.h"
#include "neighbor.h"
#include "atom_vec.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "output.h"
#include "modify.h"
#include "random_mars.h"
#include "compute.h"
#include "math_const.h"
//#include "group.h"  See "Da fare"


using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{NHC,LANGEVIN,DAMPED,NVE,NEB,NEB_TEST};
enum{MIN_SD,MIN_DAMPED,MIN_ASD};
#define BUFSIZE 5

/* ---------------------------------------------------------------------- */

FixPATH_DYNAMICS::FixPATH_DYNAMICS(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg),id_pe(NULL),pe(NULL)
{

  // fix <fix-ID> all path_dynamics <dt> <damp> <eps> [keyword <values> ...]
  if (narg < 6) error->universe_all(FLERR,"Illegal fix path_dynamics command");

  dt_path   = utils::numeric(FLERR,arg[3],false,lmp);
  damp_path = utils::numeric(FLERR,arg[4],false,lmp);
  epsilon   = utils::numeric(FLERR,arg[5],false,lmp);

  if (dt_path   <= 0.0) error->universe_all(FLERR,"Fix path_dynamics dt_path must be > 0");
  if (damp_path <= 0.0) error->universe_all(FLERR,"Fix path_dynamics damp_path must be > 0");
  if (epsilon   <= 0.0) error->universe_all(FLERR,"Fix path_dynamics epsilon must be > 0");
  
  int iarg;
  iarg=6;
  run = NVE; //Default: NVE run

  //optional arguments
  seed = 1765;
  lange_tau=1.;
  target_temp = 0;
  nhc_nchain = 1;

  if (narg>iarg){
	  if (strcmp(arg[iarg],"langevin") == 0) {
	    if ( iarg+4 != narg ) error->universe_all(FLERR,"Illegal fix path_dynamics command");
	    run = LANGEVIN;
	    target_temp = utils::numeric(FLERR,arg[iarg+1],false,lmp);
	    lange_tau   = utils::numeric(FLERR,arg[iarg+2],false,lmp);
	    seed        = utils::inumeric(FLERR,arg[iarg+3],false,lmp);
	    if (target_temp <= 0.0) error->universe_all(FLERR,"Fix path_dynamics temp must be > 0");
	    if (lange_tau   <= 0.0) error->universe_all(FLERR,"Fix path_dynamics tau must be > 0");
	    if (seed        <= 0  ) error->universe_all(FLERR,"Fix path_dynamics seed must be > 0");
	    iarg+=3;
	  }
	  else if (strcmp(arg[iarg],"nhc") == 0) {
	    if ( iarg+3 != narg ) error->universe_all(FLERR,"Illegal fix path_dynamics command");
	    run = NHC;
	    target_temp = utils::numeric(FLERR,arg[iarg+1],false,lmp);
	    nhc_nchain = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
	    if ( target_temp <= 0.0 ) error->universe_all(FLERR,"Fix path_dynamics temp must be > 0");
	    if ( nhc_nchain  <= 0   ) error->universe_all(FLERR,"Fix path_dynamics nchain must be > 0");
            iarg+=2;
	  }
	  else if (strcmp(arg[iarg],"damped") == 0) {
	    if ( iarg+2 != narg ) error->universe_all(FLERR,"Illegal fix path_dynamics command");
	    run = DAMPED;
	    lange_tau = utils::numeric(FLERR,arg[iarg+1],false,lmp);
	    if (lange_tau <= 0.0) error->universe_all(FLERR,"Fix path_dynamics tau must be > 0");
            iarg+=1;
	  }
          else if (strcmp(arg[iarg],"neb") == 0) {
            if ( iarg+2 > narg ) error->universe_all(FLERR,"Illegal fix path_dynamics command");
            run = NEB;
            if (strcmp(arg[iarg+1],"sd") == 0) {
              min_style = MIN_SD;
            }
            if (strcmp(arg[iarg+1],"asd") == 0) {
              min_style = MIN_ASD;
            }
            if (strcmp(arg[iarg+1],"damped") == 0) {
              min_style = MIN_DAMPED;
            }
            iarg+=1;
          }
	  else if (strcmp(arg[iarg],"neb_test") == 0) {
            if ( iarg+2 > narg ) error->universe_all(FLERR,"Illegal fix path_dynamics command");
            run = NEB_TEST;
            if (strcmp(arg[iarg+1],"sd") == 0) {
              min_style = MIN_SD;
            }
            if (strcmp(arg[iarg+1],"asd") == 0) {
              min_style = MIN_ASD;
            }
            if (strcmp(arg[iarg+1],"damped") == 0) {
              min_style = MIN_DAMPED;
            }
            iarg+=1;
          }
  }

  iarg+=1;
  ParaSpring = PerpSpring = FreeEndIni = FreeEndFinal = false;
  FreeEndFinalWithRespToEIni = FinalAndInterWithRespToEIni = false;
  kspringPerp = 0.0;
  kspringPara = 1.0;
  kspringIni = 1.0;
  kspringFinal = 1.0;
  climber = -1;

  while (iarg < narg) {
      if (strcmp(arg[iarg],"parallel") == 0) {
        if (iarg+2 > narg) error->universe_all(FLERR,"Illegal fix neb command");
        ParaSpring = true;
        kspringPara = utils::numeric(FLERR,arg[iarg+1],false,lmp);
        if (kspringPara <= 0.0) error->universe_all(FLERR,"Illegal fix neb command");
        iarg += 2;
      }
      else if (strcmp(arg[iarg],"perp") == 0) {
        if (iarg+2 > narg) error->universe_all(FLERR,"Illegal fix neb command");
        PerpSpring = true;
        kspringPerp = utils::numeric(FLERR,arg[iarg+1],false,lmp);
        if (kspringPerp == 0.0) PerpSpring = false;
        iarg += 2;
      }
      else if (strcmp(arg[iarg],"climber") == 0) {
        if (iarg+2 > narg) error->universe_all(FLERR,"Illegal fix neb command");
        climber = utils::numeric(FLERR,arg[iarg+1],false,lmp);
        if ( (climber < 1) || (climber > universe->nworlds) ) error->universe_all(FLERR,"Illegal fix neb command");
        climber -= 1;
        iarg += 2;
      }
      else if (strcmp (arg[iarg],"end") == 0) {
        if (iarg+3 > narg) error->universe_all(FLERR,"Illegal fix neb command");
        if (strcmp(arg[iarg+1],"first") == 0) {
          FreeEndIni = true;
          kspringIni = utils::numeric(FLERR,arg[iarg+2],false,lmp);
        } else if (strcmp(arg[iarg+1],"last") == 0) {
          FreeEndFinal = true;
          kspringFinal = utils::numeric(FLERR,arg[iarg+2],false,lmp);
        } else if (strcmp(arg[iarg+1],"last/efirst") == 0) {
          FreeEndFinal = false;
          FinalAndInterWithRespToEIni = false;
          FreeEndFinalWithRespToEIni = true;
          kspringFinal = utils::numeric(FLERR,arg[iarg+2],false,lmp);
        } else if (strcmp(arg[iarg+1],"last/efirst/middle") == 0) {
          FreeEndFinal = false;
          FinalAndInterWithRespToEIni = true;
          FreeEndFinalWithRespToEIni = true;
          kspringFinal = utils::numeric(FLERR,arg[iarg+2],false,lmp);
        } else error->universe_all(FLERR,"Illegal fix neb command");
        iarg += 3;
      }
      else error->universe_all(FLERR,"Illegal fix neb command");
  }

  //partition NxM
  np_u   = universe->nprocs; // N*M
  np_w   = comm->nprocs;     // M
  nbeads = universe->nworlds;// N
  me_u   = universe->me;     // 0,1,2,...,N*M-1
  me_w   = comm->me;         // 0,1,2,...,M-1
  ibead  = universe->iworld; // 0,1,2,...,N-1
  uworld = universe->uworld; // Inter-partition communicator

  // initialize Marsaglia RNG with processor-unique seed
  random = new RanMars(lmp,seed + me_u);

  // allocate per-type arrays for random and drag force prefactors 
  gfactor1 = new double[atom->ntypes+1];
  gfactor2 = new double[atom->ntypes+1];

  /* Initiation */
  spring_energy=0.0;
  kinetic_energy=0.0;
  potential_energy=0.0;
  lenuntilIm=0.0;
  plen=0.0;
  lentot=0.0;
  meanDist=0.0;

  max_nsend = 0;
  tag_send = NULL;
  buf_send = NULL;

  max_nlocal = 0;
  buf_recv = NULL;
  buf_beads = NULL;
  buf_force = NULL;

  size_plan = 0;
  plan_send = plan_recv = NULL;

  mode_index = NULL;

  fmass = NULL;
  md_2dt = NULL;
  dt_md = NULL;

  array_atom = NULL;
  nhc_eta = NULL;
  nhc_eta_dot = NULL;
  nhc_eta_dotdot = NULL;
  nhc_eta_mass = NULL;

  eta_l=NULL;
  eta_n=NULL;
  x_bead=NULL;
  f_bead=NULL;
  f_fwd=NULL;
  f_bwd=NULL;

  size_peratom_cols = 12 * nhc_nchain + 3;

  nhc_offset_one_1 = 3 * nhc_nchain;
  nhc_offset_one_2 = 3 * nhc_nchain +3;
  nhc_size_one_1 = sizeof(double) * nhc_offset_one_1;
  nhc_size_one_2 = sizeof(double) * nhc_offset_one_2;

  restart_peratom = 1; // 1 if Fix saves peratom state, 0 if not
  peratom_flag    = 1; // 0/1 if per-atom data is stored
  peratom_freq    = 1; // frequency per-atom data is available at

  global_freq   = 1; // frequency s/v data is available at
  thermo_energy = 1; // 1 if fix_modify energy enabled, 0 if not
  energy_global_flag  = 1; // 1 if contributes to global eng
  energy_peratom_flag = 0; // 1 if contributes to peratom eng
  vector_flag   = 1; // 0/1 if compute_vector() function exists ---> accessible as f_{fixid}[j], j=1,size_vector
  scalar_flag   = 1; // 0/1 if compute_scalar() function exists ---> accessible as f_{fixid}
  size_vector   = 7; // length of global vector
  extvector     = 1; // 0/1/-1 if global vector is all int/ext/extlist
  extscalar     = 1; // 0/1 if global scalar is intensive/extensive
  comm_forward  = 3; // size of forward communication (0 if none)

  atom->add_callback(0); // Call LAMMPS to allocate memory for per-atom array
  atom->add_callback(1); // Call LAMMPS to re-assign restart-data for per-atom array

  grow_arrays(atom->nmax);

  // some initilizations

  nhc_ready = false;

  n_pre_reverse = modify->n_pre_reverse;
  n_pre_force = modify->n_pre_force;
  //n_post_force = modify->n_post_force;//no fixes with post_force allowed
  if (force->pair && force->pair->compute_flag) pair_compute_flag = 1;
  else pair_compute_flag = 0;
  if (force->kspace && force->kspace->compute_flag) kspace_compute_flag = 1;
  else kspace_compute_flag = 0;
  if (atom->torque_flag) torqueflag = 1;
  else torqueflag = 0;
  if (atom->avec->forceclearflag) extraflag = 1;
  else extraflag = 0;

  double **v = atom->v;
  if ( ( run == NEB ) || (run == NEB_TEST) || ( run == DAMPED ) ) for (int i=0;i<atom->nlocal;i++) v[i][0]=v[i][1]=v[i][2]=0.0;

  // NEB
  if ( (run == NEB) || (run == NEB_TEST) ) {
     //Create a new compute pe style
     id_pe = utils::strdup("pathmd_pe");
     pe = modify->add_compute(std::string(id_pe) + " all pe");
     //Create new group
     int *iroots = new int[nbeads];
     MPI_Group uworldgroup,rootgroup;
     for (int i=0; i<nbeads; i++) iroots[i] = universe->root_proc[i];
     MPI_Comm_group(uworld, &uworldgroup);
     MPI_Group_incl(uworldgroup, nbeads, iroots, &rootgroup);
     MPI_Comm_create(uworld, rootgroup, &rootworld);
     delete [] iroots;
  }
  else {
     //Create a new compute pe style
     id_pe = utils::strdup("pathmd_pe");
     pe = modify->add_compute(std::string(id_pe) + " all pe");
  }
  //NEB

}

/* ---------------------------------------------------------------------- */

// If I include the destructor, simulations run smoothly till the end but application stops with seg fault
// NOTE: Also in fix_pimd.* the destructor is not defined
/*
FixPATH_DYNAMICS::~FixPATH_DYNAMICS()
{
  if (random)   delete [] random;
  if (gfactor1) delete [] gfactor1;
  if (gfactor2) delete [] gfactor2;
}
*/

/* ---------------------------------------------------------------------- */

int FixPATH_DYNAMICS::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::init()
{
  int icompute = modify->find_compute(id_pe);
  if (icompute < 0) error->universe_all(FLERR,"Potential energy ID for fix pathmd does not exist");
  pe = modify->compute[icompute];

  if (atom->map_style == 0)
    error->universe_all(FLERR,"Fix path_dynamics requires an atom map, see atom_modify");

  if(me_u==0 && screen) fprintf(screen,"Fix path_dynamics initializing Classical Path Dynamics\n");

  // time steps

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  //setup comm

  comm_init();

  // Open polymer model

  //Fictitious masses
  fmass = new double [atom->ntypes+1];
  for(int i=1; i<=atom->ntypes; i++) fmass[i]=atom->mass[i]; //Room for improvement here

  //Coeffs appearing in spring energy
  md_2dt = new double [atom->ntypes+1];
  dt_md = new double [atom->ntypes+1];
  for(int i=1; i<=atom->ntypes; i++) {
    //Use atomic masses 
    md_2dt[i] = atom->mass[i] * damp_path / dt_path / 2.0;
    dt_md[i] = dt_path / atom->mass[i] / damp_path;
    //
  }
  
  //Random and drag force prefactors. Use fictitious masses
  for (int i = 1; i <= atom->ntypes; i++) { 
      gfactor1[i] = -fmass[i] / lange_tau / force->ftm2v;
      gfactor2[i] = sqrt(fmass[i]) * sqrt(target_temp) *
          sqrt(24.0*force->boltz/lange_tau/update->dt/force->mvv2e) / force->ftm2v;
  }

  //Initialize nhc
  if ( run == NHC ) if(!nhc_ready) nhc_init();
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::setup(int vflag)
{
  if(me_u==0 && screen) fprintf(screen,"Setting up Classical Path Dynamics\n");
  if(me_u==0 && screen && (run==NVE)) fprintf(screen,"NVE run\n");
  if(me_u==0 && screen && (run==DAMPED)) fprintf(screen,"Damped dynamics\n");
  if(me_u==0 && screen && (run==LANGEVIN)) fprintf(screen,"NVT run using Langevin thermostat\n");
  if(me_u==0 && screen && (run==NHC)) fprintf(screen,"NVT run using Nose-Hoover chains thermostat\n");
  if(me_u==0 && screen && (run==NEB)) {
      if ( min_style == MIN_SD )     fprintf(screen,"NEB-like action minimization using steepest descent\n");
      if ( min_style == MIN_ASD )    fprintf(screen,"NEB-like action minimization using accellerated steepest descent\n");
      if ( min_style == MIN_DAMPED ) fprintf(screen,"NEB-like action minimization using damped dynamics\n");
  }
  if(me_u==0 && screen && (run==NEB_TEST)) {
      if ( min_style == MIN_SD )     fprintf(screen,"NEB-like action minimization using steepest descent\n");
      if ( min_style == MIN_ASD )    fprintf(screen,"NEB-like action minimization using accellerated steepest descent\n");
      if ( min_style == MIN_DAMPED ) fprintf(screen,"NEB-like action minimization using damped dynamics\n");
      fprintf(screen,"The hessian contribution to the spring forces will be neglected.\n");
  }

  // trigger potential energy computation on next timestep
  pe->addstep(update->ntimestep+1);

  post_force(vflag);//Compute forces and spring energy at the very beginning of simulation
  compute_kinetic_energy(); //Compute kinetic energy
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::initial_integrate(int /*vflag*/)
{
  //Half time step in v
  if (run == NHC) nhc_update_v();
  else if ( (run == NEB) || (run == NEB_TEST) ) neb_update_v();
  else update_v();
  //Full time-step in x
  update_x();
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::final_integrate()
{
  //After initial_integrate, the time-step proceeds with a new calculation of all forces (see developer manual)
  //Then the post_force is called, which here recomputes the forces via 'spring_force()' or 'neb_force()'
  //NOTE: 'spring_force()' also computes the spring_energy at the updated positions
  //After this post_force is called, the post_force of any other fix defined in input is called that further modify the forces.
  //Then this final_integrate is called that performs the last half-step of Verlet
  if (run == NHC) nhc_update_v();
  else if ( (run == NEB) || (run == NEB_TEST) ) neb_update_v();
  else update_v();
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::post_force(int flag)
{
  
  //Open polymer model 
  pimd_fill(atom->f);  //forward comm for the force on ghost atoms
  comm_exec(atom->f);  //Communicates forces and stores them in buf_beads
  memcpy(buf_force[x_last],buf_beads[x_last],sizeof(double)*atom->nlocal*3); //store forces in buf_force
  memcpy(buf_force[x_next],buf_beads[x_next],sizeof(double)*atom->nlocal*3); //store forces in buf_force
  comm_exec(atom->x);  //Communicates positions and stores them in buf_beads

  if ( run == NEB ) neb_force();
  else if ( run == NEB_TEST ) neb_test_force();
  else spring_force();

  //Drag and stochastic forces
  if ( ( run == DAMPED ) || ( (( run == NEB ) || (run == NEB_TEST)) && ( min_style == MIN_DAMPED ) ) ) { //Damped dynamics

      int *type = atom->type;
      double **v = atom->v;
      double **f = atom->f;
      double fdrag[3];
      double gamma1;

      for (int i=0;i<atom->nlocal;i++) {

          gamma1 = gfactor1[type[i]];

          fdrag[0] = gamma1*v[i][0];
          fdrag[1] = gamma1*v[i][1];
          fdrag[2] = gamma1*v[i][2];
       
          f[i][0] += fdrag[0];
          f[i][1] += fdrag[1];
          f[i][2] += fdrag[2];
      }
  }
  else if ( run == LANGEVIN ) { //Langevin thermostat

      int *type = atom->type;
      double **v = atom->v;
      double **f = atom->f;

      double fran[3],fdrag[3];
      double gamma1, gamma2;

      for (int i=0;i<atom->nlocal;i++) {

          gamma1 = gfactor1[type[i]];
          gamma2 = gfactor2[type[i]];

          fran[0] = gamma2*(random->uniform()-0.5);
          fran[1] = gamma2*(random->uniform()-0.5);
          fran[2] = gamma2*(random->uniform()-0.5);

          fdrag[0] = gamma1*v[i][0];
          fdrag[1] = gamma1*v[i][1];
          fdrag[2] = gamma1*v[i][2];

          f[i][0] += fdrag[0] + fran[0];
          f[i][1] += fdrag[1] + fran[1];
          f[i][2] += fdrag[2] + fran[2];
    }
  }
 
}

/* ----------------------------------------------------------------------
   Nose-Hoover Chains
------------------------------------------------------------------------- */

void FixPATH_DYNAMICS::nhc_init()
{
  double KT  = force->boltz * target_temp;
  int max = 3 * atom->nlocal;

  for(int i=0; i<max; i++)
  {
    for(int ichain=0; ichain<nhc_nchain; ichain++)
    {
      nhc_eta[i][ichain]        = 0.0;
      nhc_eta_dot[i][ichain]    = 0.0;
      nhc_eta_dot[i][ichain]    = 0.0;
      nhc_eta_dotdot[i][ichain] = 0.0;
      nhc_eta_mass[i][ichain]   = 1.0; //Room for improvement here
    }

    nhc_eta_dot[i][nhc_nchain]    = 0.0;

    for(int ichain=1; ichain<nhc_nchain; ichain++)
      nhc_eta_dotdot[i][ichain] = (nhc_eta_mass[i][ichain-1] * nhc_eta_dot[i][ichain-1]
        * nhc_eta_dot[i][ichain-1] * force->mvv2e - KT) / nhc_eta_mass[i][ichain];
  }

  nhc_ready = true;
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::update_x()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **v = atom->v;
  for(int i=0; i<nlocal; i++)
  {
    x[i][0] += dtv * v[i][0];
    x[i][1] += dtv * v[i][1];
    x[i][2] += dtv * v[i][2];
  }
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::neb_update_v()
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double **v = atom->v;
  double **f = atom->f;
  if ( min_style == MIN_ASD )
  {
        double vdotf,fdotf,vdotfall,fdotfall;
        vdotf=fdotf=vdotfall=fdotfall=0.0;
        for(int i=0; i<nlocal; i++) {
         vdotf += v[i][0]*f[i][0] + v[i][1]*f[i][1] + v[i][2]*f[i][2];
         fdotf += f[i][0]*f[i][0] + f[i][1]*f[i][1] + f[i][2]*f[i][2];
        }
        //Sum over all procs in the universe
        MPI_Allreduce(&vdotf,&vdotfall,1,MPI_DOUBLE,MPI_SUM,uworld);
        MPI_Allreduce(&fdotf,&fdotfall,1,MPI_DOUBLE,MPI_SUM,uworld);
        vdotf=vdotfall;
        fdotf=fdotfall;
        if (vdotfall > 0.0) // v_new = dot_product(v_old,f)*f/|f|**2 + f*dt; accellerated steepest descent
        {
           double prefactor=vdotf/fdotf;
           for(int i=0; i<nlocal; i++)
           {
             double dtfm = dtf / fmass[type[i]];
             v[i][0] = (prefactor+dtfm)*f[i][0];
             v[i][1] = (prefactor+dtfm)*f[i][1];
             v[i][2] = (prefactor+dtfm)*f[i][2];
           }
        }
        else // v_new = f*dt;  steepest descent
        {
           for(int i=0; i<nlocal; i++)
           {
             double dtfm = dtf / fmass[type[i]];
             v[i][0] = dtfm * f[i][0];
             v[i][1] = dtfm * f[i][1];
             v[i][2] = dtfm * f[i][2];
           }
        }
  }
  else if ( min_style == MIN_SD )
  {
           for(int i=0; i<nlocal; i++)
           {
             double dtfm = dtf / fmass[type[i]];
             v[i][0] = dtfm * f[i][0];
             v[i][1] = dtfm * f[i][1];
             v[i][2] = dtfm * f[i][2];
           }
  }
  else if ( min_style == MIN_DAMPED )
  {
           for(int i=0; i<nlocal; i++)
           {
             double dtfm = dtf / fmass[type[i]];
             v[i][0] += dtfm * f[i][0];
             v[i][1] += dtfm * f[i][1];
             v[i][2] += dtfm * f[i][2];
           }
  }
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::update_v()
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double **v = atom->v;
  double **f = atom->f;
  for(int i=0; i<nlocal; i++)
  {
    double dtfm = dtf / fmass[type[i]];
    v[i][0] += dtfm * f[i][0];
    v[i][1] += dtfm * f[i][1];
    v[i][2] += dtfm * f[i][2];
  }
}

/* ---------------------------------------------------------------------- */
/*                       Nose-Hoover chains thermostat                    */
/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::nhc_update_v()
{
  int n = atom->nlocal;
  int *type = atom->type;
  double **v = atom->v;
  double **f = atom->f;

  for(int i=0; i<n; i++)
  {
    double dtfm = dtf / fmass[type[i]];
    v[i][0] += dtfm * f[i][0];
    v[i][1] += dtfm * f[i][1];
    v[i][2] += dtfm * f[i][2];
  }

  t_sys = 0.0;

  double expfac;
  int nmax = 3 * atom->nlocal;
  double KT = force->boltz * target_temp;
  double kecurrent, t_current;

  double dthalf = 0.5   * update->dt;
  double dt4    = 0.25  * update->dt;
  double dt8    = 0.125 * update->dt;

  for(int i=0; i<nmax; i++)
  {
    int iatm = i/3;
    int idim = i%3;

    double *vv = v[iatm];

    kecurrent = fmass[type[iatm]] * vv[idim]* vv[idim] * force->mvv2e;
    t_current = kecurrent / force->boltz;

    double *eta = nhc_eta[i];
    double *eta_dot = nhc_eta_dot[i];
    double *eta_dotdot = nhc_eta_dotdot[i];

    eta_dotdot[0] = (kecurrent - KT) / nhc_eta_mass[i][0];

    for(int ichain=nhc_nchain-1; ichain>0; ichain--)
    {
      expfac = exp(-dt8 * eta_dot[ichain+1]);
      eta_dot[ichain] *= expfac;
      eta_dot[ichain] += eta_dotdot[ichain] * dt4;
      eta_dot[ichain] *= expfac;
    }

    expfac = exp(-dt8 * eta_dot[1]);
    eta_dot[0] *= expfac;
    eta_dot[0] += eta_dotdot[0] * dt4;
    eta_dot[0] *= expfac;

    // Update particle velocities half-step

    double factor_eta = exp(-dthalf * eta_dot[0]);
    vv[idim] *= factor_eta;

    t_current *= (factor_eta * factor_eta);
    kecurrent = force->boltz * t_current;
    eta_dotdot[0] = (kecurrent - KT) / nhc_eta_mass[i][0];

    for(int ichain=0; ichain<nhc_nchain; ichain++)
      eta[ichain] += dthalf * eta_dot[ichain];

    eta_dot[0] *= expfac;
    eta_dot[0] += eta_dotdot[0] * dt4;
    eta_dot[0] *= expfac;

    for(int ichain=1; ichain<nhc_nchain; ichain++)
    {
      expfac = exp(-dt8 * eta_dot[ichain+1]);
      eta_dot[ichain] *= expfac;
      eta_dotdot[ichain] = (nhc_eta_mass[i][ichain-1] * eta_dot[ichain-1] * eta_dot[ichain-1]
                           - KT) / nhc_eta_mass[i][ichain];
      eta_dot[ichain] += eta_dotdot[ichain] * dt4;
      eta_dot[ichain] *= expfac;
    }

    t_sys += t_current;
  }

  t_sys /= nmax;
}

/* ----------------------------------------------------------------------
   Polymer forces
------------------------------------------------------------------------- */

void FixPATH_DYNAMICS::spring_force()
{
  spring_energy = 0.0;
  potential_energy = 0.0;

  //I want to use the OM action as CV.
  //To this end, I plan to use plumed's 'Energy.cpp' action.
  //'Energy.cpp' gets the potential energy from the MD engine.
  //Therefore, I need to modify the potential energy computed by LAMMPS in such a way that it is equal to 
  //the OM action.
  //HOW I ACHIEVE THIS:
  //LAMMPS allows a fix to 'modify' the potential energy of the system by adding to the potential energy the output
  //of its compute_scalar() method. Therefore, I need to design the compute_scalar() method so that its output added to
  //the potential energy of the bead becomes equal to the OM action.
  //In practice, I need to be able to reset the potential energy to zero and then add the OM action.
  //The following lines compute the physical potential energy of the bead:
  pe->compute_scalar();
  potential_energy = pe->scalar;
  pe->addstep(update->ntimestep+1); // trigger potential energy computation on next timestep
  //I then use 'potential_energy' in the compute_scalar() method so that the potential energy of the bead
  //plus the output os compute_scalar() is equal to the OM action (see bottom).

  double **x = atom->x;
  double **f = atom->f;
  double **v = atom->v;
  int* type = atom->type;
  int nlocal = atom->nlocal;

  //Positions of atoms of the two adjacent replicas
  double* xlast = buf_beads[x_last];
  double* xnext = buf_beads[x_next];

  //Forces acting on atoms of the two adjacent replicas
  double* flast = buf_force[x_last]; //We need only flast

  double maxeta2=0.0;
  double eps=0.0;
  double maxx=0.0;

  //nlocal can change. I need to reallocate at every time step (check)
  //Allocate spring length
  eta_l = new double* [nlocal];
  for (int i=0;i<nlocal;i++) eta_l[i] = new double [3];
  eta_n = new double* [nlocal];
  for (int i=0;i<nlocal;i++) eta_n[i] = new double [3];
  //Allocate aux x and f
  f_fwd = new double* [nlocal];
  for (int i=0;i<nlocal;i++) f_fwd[i] = new double [3];
  f_bwd = new double* [nlocal];
  for (int i=0;i<nlocal;i++) f_bwd[i] = new double [3];
  x_bead = new double* [nlocal];
  for (int i=0;i<nlocal;i++) x_bead[i] = new double [3];
  f_bead = new double* [nlocal];
  for (int i=0;i<nlocal;i++) f_bead[i] = new double [3];

  //Store current positions
  for (int i=0;i<nlocal;i++){ x_bead[i][0]=x[i][0]; x_bead[i][1]=x[i][1]; x_bead[i][2]=x[i][2]; }
  //Store current forces
  for (int i=0;i<nlocal;i++){ f_bead[i][0]=f[i][0]; f_bead[i][1]=f[i][1]; f_bead[i][2]=f[i][2]; }

  if(ibead==0)//first bead
  {
          
	  //Compute spring lengths, spring energy
	  for(int i=0; i<nlocal; i++)
	  {
                  double dx,dy,dz;
                  dx = xnext[0] - x_bead[i][0];
                  dy = xnext[1] - x_bead[i][1];
                  dz = xnext[2] - x_bead[i][2];
                  domain->minimum_image(dx,dy,dz);
		  eta_n[i][0] = dx - dt_md[type[i]] * f_bead[i][0] * force->ftm2v;
		  eta_n[i][1] = dy - dt_md[type[i]] * f_bead[i][1] * force->ftm2v;
		  eta_n[i][2] = dz - dt_md[type[i]] * f_bead[i][2] * force->ftm2v;
		  xnext += 3;
		  double msd = (eta_n[i][0]*eta_n[i][0]+eta_n[i][1]*eta_n[i][1]+eta_n[i][2]*eta_n[i][2]);
                  if ( msd > maxeta2 ) maxeta2=msd;
       		  spring_energy += 0.5 * md_2dt[type[i]] * msd * force->mvv2e;
	  }

          MPI_Allreduce(&maxeta2, &maxx, 1, MPI_DOUBLE, MPI_MAX, world);
          eps=epsilon/sqrt(maxx); //Room for improvement here

	  // Displace x fwd
	  for (int i=0;i<nlocal;i++){
		  x[i][0] = x_bead[i][0] + eps * eta_n[i][0];
		  x[i][1] = x_bead[i][1] + eps * eta_n[i][1];
		  x[i][2] = x_bead[i][2] + eps * eta_n[i][2];
	  }
	  // Compute f_fwd
          compute_force();
	  // Store f_fwd
	  for (int i=0;i<nlocal;i++){ f_fwd[i][0]=f[i][0]; f_fwd[i][1]=f[i][1]; f_fwd[i][2]=f[i][2]; }
	  // Displace x bwd
	  for (int i=0;i<nlocal;i++){
		  x[i][0] = x_bead[i][0] - eps * eta_n[i][0];
                  x[i][1] = x_bead[i][1] - eps * eta_n[i][1];
                  x[i][2] = x_bead[i][2] - eps * eta_n[i][2];
          }
	  // Compute f_bwd
          compute_force();
	  // Store f_bwd
	  for (int i=0;i<nlocal;i++){ f_bwd[i][0]=f[i][0]; f_bwd[i][1]=f[i][1]; f_bwd[i][2]=f[i][2]; }
	  // Update f
	  for (int i=0;i<nlocal;i++){//f = force-field + ~spring~ + ~hessian~
                  f[i][0] = f_bead[i][0] + md_2dt[type[i]] * eta_n[i][0] * force->mvv2e + 0.25 * ( f_fwd[i][0] - f_bwd[i][0] ) / eps;
                  f[i][1] = f_bead[i][1] + md_2dt[type[i]] * eta_n[i][1] * force->mvv2e + 0.25 * ( f_fwd[i][1] - f_bwd[i][1] ) / eps;
                  f[i][2] = f_bead[i][2] + md_2dt[type[i]] * eta_n[i][2] * force->mvv2e + 0.25 * ( f_fwd[i][2] - f_bwd[i][2] ) / eps;
          }
  }
  else if (ibead == nbeads-1)//last bead: no contrib to spring_energy to avoid double counting, f is only due to last spring.
  {
          //Compute spring length
          for(int i=0; i<nlocal; i++)
          {
                  double dx,dy,dz;
                  dx = x_bead[i][0] - xlast[0];
                  dy = x_bead[i][1] - xlast[1];
                  dz = x_bead[i][2] - xlast[2];
                  domain->minimum_image(dx,dy,dz);
                  eta_l[i][0] = dx - dt_md[type[i]] * flast[0] * force->ftm2v;
                  eta_l[i][1] = dy - dt_md[type[i]] * flast[1] * force->ftm2v;
                  eta_l[i][2] = dz - dt_md[type[i]] * flast[2] * force->ftm2v;
                  flast += 3;
                  xlast += 3;
          }
          // Update f
          for (int i=0;i<nlocal;i++){//f = springs
                  f[i][0] = md_2dt[type[i]] * ( - eta_l[i][0] ) * force->mvv2e;
                  f[i][1] = md_2dt[type[i]] * ( - eta_l[i][1] ) * force->mvv2e;
                  f[i][2] = md_2dt[type[i]] * ( - eta_l[i][2] ) * force->mvv2e;
          }
  }
  else //bulk beads
  {
	  //Compute spring length, spring energy
	  for(int i=0; i<nlocal; i++)
          {
                  double dx,dy,dz;
                  dx = xnext[0] - x_bead[i][0];
                  dy = xnext[1] - x_bead[i][1];
                  dz = xnext[2] - x_bead[i][2];
                  domain->minimum_image(dx,dy,dz);
                  eta_n[i][0] = dx - dt_md[type[i]] * f_bead[i][0] * force->ftm2v;
                  eta_n[i][1] = dy - dt_md[type[i]] * f_bead[i][1] * force->ftm2v;
                  eta_n[i][2] = dz - dt_md[type[i]] * f_bead[i][2] * force->ftm2v;
                  dx = x_bead[i][0] - xlast[0];
                  dy = x_bead[i][1] - xlast[1];
                  dz = x_bead[i][2] - xlast[2];
                  domain->minimum_image(dx,dy,dz);
		  eta_l[i][0] = dx - dt_md[type[i]] * flast[0] * force->ftm2v;
                  eta_l[i][1] = dy - dt_md[type[i]] * flast[1] * force->ftm2v;
                  eta_l[i][2] = dz - dt_md[type[i]] * flast[2] * force->ftm2v;
		  flast += 3;
		  xlast += 3;
                  xnext += 3;
                  double msd = (eta_n[i][0]*eta_n[i][0]+eta_n[i][1]*eta_n[i][1]+eta_n[i][2]*eta_n[i][2]);
                  if (msd>maxeta2) maxeta2=msd;
                  spring_energy += 0.5 * md_2dt[type[i]] * msd * force->mvv2e;
          }

          MPI_Allreduce(&maxeta2, &maxx, 1, MPI_DOUBLE, MPI_MAX, world);
          eps=epsilon/sqrt(maxx); //Room for improvement here

          // Displace x fwd
          for (int i=0;i<nlocal;i++){
                  x[i][0] = x_bead[i][0] + eps * eta_n[i][0];
                  x[i][1] = x_bead[i][1] + eps * eta_n[i][1];
                  x[i][2] = x_bead[i][2] + eps * eta_n[i][2];
          }
          // Compute f_fwd
          compute_force();
          // Store f_fwd
          for (int i=0;i<nlocal;i++){ f_fwd[i][0]=f[i][0]; f_fwd[i][1]=f[i][1]; f_fwd[i][2]=f[i][2]; }
          // Displace x bwd
          for (int i=0;i<nlocal;i++){
                  x[i][0] = x_bead[i][0] - eps * eta_n[i][0];
                  x[i][1] = x_bead[i][1] - eps * eta_n[i][1];
                  x[i][2] = x_bead[i][2] - eps * eta_n[i][2];
          }
          // Compute f_bwd
          compute_force();
          // Store f_bwd
          for (int i=0;i<nlocal;i++){ f_bwd[i][0]=f[i][0]; f_bwd[i][1]=f[i][1]; f_bwd[i][2]=f[i][2]; }
          // Update f Finite Diff
          for (int i=0;i<nlocal;i++){//f = ~springs~ + ~hessian~
                  f[i][0] = md_2dt[type[i]] * ( eta_n[i][0] - eta_l[i][0] ) * force->mvv2e + 0.25 * ( f_fwd[i][0] - f_bwd[i][0] ) / eps;
                  f[i][1] = md_2dt[type[i]] * ( eta_n[i][1] - eta_l[i][1] ) * force->mvv2e + 0.25 * ( f_fwd[i][1] - f_bwd[i][1] ) / eps;
                  f[i][2] = md_2dt[type[i]] * ( eta_n[i][2] - eta_l[i][2] ) * force->mvv2e + 0.25 * ( f_fwd[i][2] - f_bwd[i][2] ) / eps;
          }
  }
  
  // Restore x
  for (int i=0;i<nlocal;i++){ x[i][0]=x_bead[i][0]; x[i][1]=x_bead[i][1]; x[i][2]=x_bead[i][2]; }

  //Sum spring_energy from all procs in this world
  double aux_E;
  aux_E=0.0;
  MPI_Allreduce(&spring_energy, &aux_E, 1, MPI_DOUBLE, MPI_SUM, world);
  spring_energy=aux_E;

  //Free aux arrays
  if (f_fwd){ for(int i=0; i<nlocal; i++) {if (f_fwd[i]) delete [] f_fwd[i];}
              delete [] f_fwd;}
  if (f_bwd){ for(int i=0; i<nlocal; i++) {if (f_bwd[i]) delete [] f_bwd[i];}
              delete [] f_bwd;}
  if (x_bead){ for(int i=0; i<nlocal; i++) {if (x_bead[i]) delete [] x_bead[i];}
               delete [] x_bead;}
  if (f_bead){ for(int i=0; i<nlocal; i++) {if (f_bead[i]) delete [] f_bead[i];}
               delete [] f_bead;}
  if (eta_n){ for(int i=0; i<nlocal; i++) {if (eta_n[i]) delete [] eta_n[i];}
              delete [] eta_n;}
  if (eta_l){ for(int i=0; i<nlocal; i++) {if (eta_l[i]) delete [] eta_l[i];}
              delete [] eta_l;}

}//END spring_force

/* ----------------------------------------------------------------------
   NEB-like force without hessian
------------------------------------------------------------------------- */

void FixPATH_DYNAMICS::neb_test_force()
{
  spring_energy = 0.0;

  //Communicate epot for smart tangent and 'end' option
  double vprev,vnext; //Potential energy of previous and next bead
  int procprev,procnext;
  double vIni=0.0;

  pe->compute_scalar();
  vprev = vnext = veng = pe->scalar;
  // trigger potential energy computation on next timestep
  pe->addstep(update->ntimestep+1);

  if (ibead > 0) procprev = universe->root_proc[ibead-1];
  else procprev = -1;

  if (ibead < nbeads-1) procnext = universe->root_proc[ibead+1];
  else procnext = -1;

  if (ibead < nbeads-1 && me_w == 0) MPI_Send(&veng,1,MPI_DOUBLE,procnext,0,uworld);

  if (ibead > 0 && me_w == 0) MPI_Recv(&vprev,1,MPI_DOUBLE,procprev,0,uworld,MPI_STATUS_IGNORE);

  if (ibead > 0 && me_w == 0) MPI_Send(&veng,1,MPI_DOUBLE,procprev,0,uworld);

  if (ibead < nbeads-1 && me_w == 0) MPI_Recv(&vnext,1,MPI_DOUBLE,procnext,0,uworld,MPI_STATUS_IGNORE);

  MPI_Bcast(&vprev,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&vnext,1,MPI_DOUBLE,0,world);
   
  if ( ibead == 0 ) vIni=veng;
  if ( me_w == 0 ) MPI_Bcast(&vIni,1,MPI_DOUBLE,0,rootworld);
  MPI_Bcast(&vIni,1,MPI_DOUBLE,0,world);
  //End epot comm

  if ( (ibead == nbeads-1) && (update->ntimestep == 0)) EFinalIni = veng;
  if ( (ibead == 0 ) && (update->ntimestep == 0)) EIniIni = veng;

  //pos vel force owned by this proc
  double **x = atom->x;
  double **f = atom->f;
  double **v = atom->v;
  int* type = atom->type;
  int nlocal = atom->nlocal;
  //Positions of atoms of the two adjacent replicas
  double* xlast = buf_beads[x_last];
  double* xnext = buf_beads[x_next];
  //Forces acting on atoms of the two adjacent replicas
  double* flast = buf_force[x_last];
  //double* fnext = buf_force[x_next];
  //Some aux vars
  double maxeta2,eps,maxx;
  double dxn,dyn,dzn,dxp,dyp,dzp;
  maxeta2 = eps = maxx = 0.0;
  dxn = dyn = dzn = dxp = dyp = dzp = 0.0;
  dotpath = dot = tlen = plen = nlen = 0.0;

  //nlocal can change. I need to reallocate at every time step (check)
  SpringFPerp = new double* [nlocal];
  for (int i=0;i<nlocal;i++) SpringFPerp[i] = new double [3];
  tangent = new double* [nlocal];
  for (int i=0;i<nlocal;i++) tangent[i] = new double [3];
  eta_l = new double* [nlocal];
  for (int i=0;i<nlocal;i++) eta_l[i] = new double [3];
  eta_n = new double* [nlocal];
  for (int i=0;i<nlocal;i++) eta_n[i] = new double [3];
  //Allocate aux x and f
  x_bead = new double* [nlocal];
  for (int i=0;i<nlocal;i++) x_bead[i] = new double [3];
  f_bead = new double* [nlocal];
  for (int i=0;i<nlocal;i++) f_bead[i] = new double [3];

  //Store current positions
  for (int i=0;i<nlocal;i++){ x_bead[i][0]=x[i][0]; x_bead[i][1]=x[i][1]; x_bead[i][2]=x[i][2]; }
  //Store current force-field forces
  for (int i=0;i<nlocal;i++){ f_bead[i][0]=f[i][0]; f_bead[i][1]=f[i][1]; f_bead[i][2]=f[i][2]; }

  if ( ibead == 0 )//First bead moves according only to f_bead
  {                //To held it fixed use "partition yes 1 fix set_force 0 0 0" in the input, after fix path_dynamics.
	  for(int i=0; i<nlocal; i++)
	  {
             dxn = xnext[0] - x_bead[i][0];
             dyn = xnext[1] - x_bead[i][1];
             dzn = xnext[2] - x_bead[i][2];
             domain->minimum_image(dxn,dyn,dzn);
             //Needed for spring energy:
             eta_n[i][0] = dxn - dt_md[type[i]] * f_bead[i][0] * force->ftm2v;
             eta_n[i][1] = dyn - dt_md[type[i]] * f_bead[i][1] * force->ftm2v;
             eta_n[i][2] = dzn - dt_md[type[i]] * f_bead[i][2] * force->ftm2v;
             //NEB
             nlen += dxn*dxn + dyn*dyn + dzn*dzn;
             tangent[i][0]=dxn; tangent[i][1]=dyn; tangent[i][2]=dzn;
             tlen += dxn*dxn + dyn*dyn + dzn*dzn;
             dot += f_bead[i][0]*tangent[i][0] + f_bead[i][1]*tangent[i][1] + f_bead[i][2]*tangent[i][2];
             //NEB
             double msd = eta_n[i][0]*eta_n[i][0] + eta_n[i][1]*eta_n[i][1] + eta_n[i][2]*eta_n[i][2];
       	     if (ParaSpring) spring_energy += 0.5 * kspringPara * msd * force->mvv2e;
             else spring_energy += 0.5 * md_2dt[type[i]] * msd * force->mvv2e;
             xnext += 3;
	  }
  }

  else if ( ibead == nbeads-1 ) //Last bead moves according only to f_bead
  {                             //To held it fixed use "partition yes N fix set_force 0 0 0" in the input, after fix path_dynamics.
          for(int i=0; i<nlocal; i++)
          {
             dxp = x_bead[i][0] - xlast[0];
             dyp = x_bead[i][1] - xlast[1];
             dzp = x_bead[i][2] - xlast[2];
             domain->minimum_image(dxp,dyp,dzp);
             //NEB
             plen += dxp*dxp + dyp*dyp + dzp*dzp;
             tangent[i][0]=dxp; tangent[i][1]=dyp; tangent[i][2]=dzp;
             tlen += dxp*dxp + dyp*dyp + dzp*dzp;
             dot += f_bead[i][0]*tangent[i][0] + f_bead[i][1]*tangent[i][1] + f_bead[i][2]*tangent[i][2];
             //NEB
             xlast += 3;
          }
  }
  else // bulk beads: I need to compute also the spring forces
  {
    double vmax = MAX(fabs(vnext-veng),fabs(vprev-veng));
    double vmin = MIN(fabs(vnext-veng),fabs(vprev-veng));
	  for(int i=0; i<nlocal; i++)
          {
            dxn = xnext[0] - x_bead[i][0];
            dyn = xnext[1] - x_bead[i][1];
            dzn = xnext[2] - x_bead[i][2];
            domain->minimum_image(dxn,dyn,dzn);
            eta_n[i][0] = dxn - dt_md[type[i]] * (f_bead[i][0]) * force->ftm2v;
            eta_n[i][1] = dyn - dt_md[type[i]] * (f_bead[i][1]) * force->ftm2v;
            eta_n[i][2] = dzn - dt_md[type[i]] * (f_bead[i][2]) * force->ftm2v;
            dxp = x_bead[i][0] - xlast[0];
            dyp = x_bead[i][1] - xlast[1];
            dzp = x_bead[i][2] - xlast[2];
            domain->minimum_image(dxp,dyp,dzp);
	    eta_l[i][0] = dxp - dt_md[type[i]] * (flast[0]) * force->ftm2v;
            eta_l[i][1] = dyp - dt_md[type[i]] * (flast[1]) * force->ftm2v;
            eta_l[i][2] = dzp - dt_md[type[i]] * (flast[2]) * force->ftm2v;
            //NEB
	    if (vnext > veng && veng > vprev) {
	      tangent[i][0] = dxn;
	      tangent[i][1] = dyn;
	      tangent[i][2] = dzn;
	    } else if (vnext < veng && veng < vprev) {
	      tangent[i][0] = dxp;
	      tangent[i][1] = dyp;
	      tangent[i][2] = dzp;
	    } else {
	        if (vnext > vprev) {
	          tangent[i][0] = vmax*dxn + vmin*dxp;
	          tangent[i][1] = vmax*dyn + vmin*dyp;
	          tangent[i][2] = vmax*dzn + vmin*dzp;
	        } else if (vnext < vprev) {
	          tangent[i][0] = vmin*dxn + vmax*dxp;
	          tangent[i][1] = vmin*dyn + vmax*dyp;
	          tangent[i][2] = vmin*dzn + vmax*dzp;
	        } else { // vnext == vprev, e.g. for potentials that do not compute an energy
	          tangent[i][0] = dxn + dxp;
	          tangent[i][1] = dyn + dyp;
	          tangent[i][2] = dzn + dzp;
	        }
	    }
	    plen += dxp*dxp + dyp*dyp + dzp*dzp;
  	    nlen += dxn*dxn + dyn*dyn + dzn*dzn;
            tlen += tangent[i][0]*tangent[i][0] + tangent[i][1]*tangent[i][1] + tangent[i][2]*tangent[i][2];
            dotpath += dxp*dxn + dyp*dyn + dzp*dzn;
            dot += f_bead[i][0]*tangent[i][0] + f_bead[i][1]*tangent[i][1] + f_bead[i][2]*tangent[i][2];
            //NEB
            double msd = eta_n[i][0]*eta_n[i][0] + eta_n[i][1]*eta_n[i][1] + eta_n[i][2]*eta_n[i][2];
            if (ParaSpring) spring_energy += 0.5 * kspringPara * msd * force->mvv2e;
            else spring_energy += 0.5 * md_2dt[type[i]] * msd * force->mvv2e;
            flast += 3;
            xlast += 3;
            xnext += 3;
          }

          // Update spring forces
          for (int i=0;i<nlocal;i++)
          {   //f = ~springs~ + ~hessian~
              if (ParaSpring) {
                f[i][0] = kspringPara * ( eta_n[i][0] - eta_l[i][0] ) * force->mvv2e;
                f[i][1] = kspringPara * ( eta_n[i][1] - eta_l[i][1] ) * force->mvv2e;
                f[i][2] = kspringPara * ( eta_n[i][2] - eta_l[i][2] ) * force->mvv2e;
              }
              else{
                f[i][0] = md_2dt[type[i]] * ( eta_n[i][0] - eta_l[i][0] ) * force->mvv2e;
                f[i][1] = md_2dt[type[i]] * ( eta_n[i][1] - eta_l[i][1] ) * force->mvv2e;
                f[i][2] = md_2dt[type[i]] * ( eta_n[i][2] - eta_l[i][2] ) * force->mvv2e;
              }
              if (PerpSpring) {
                SpringFPerp[i][0] = kspringPerp * ( eta_n[i][0] - eta_l[i][0] ) * force->mvv2e;
                SpringFPerp[i][1] = kspringPerp * ( eta_n[i][1] - eta_l[i][1] ) * force->mvv2e;
                SpringFPerp[i][2] = kspringPerp * ( eta_n[i][2] - eta_l[i][2] ) * force->mvv2e;
              }
          }
  }
  //At this point, f_bead[][] contains the forces due to the force-field, f[][] contains the spring forces and optionally SpringFPerp contains the spring forces used only in perp direction

  //Restore positions
  for (int i=0;i<nlocal;i++){ x[i][0]=x_bead[i][0]; x[i][1]=x_bead[i][1]; x[i][2]=x_bead[i][2]; }

  //Sum spring_energy from all procs in this world
  double aux_E;
  aux_E=0.0;
  MPI_Allreduce(&spring_energy, &aux_E, 1, MPI_DOUBLE, MPI_SUM, world);
  spring_energy=aux_E;
  //Potential energy
  potential_energy=veng;

  //Reduce
  double bufin[BUFSIZE], bufout[BUFSIZE];
  bufin[0] = nlen;
  bufin[1] = plen;
  bufin[2] = tlen;
  bufin[3] = dotpath;
  bufin[4] = dot;
  MPI_Allreduce(bufin,bufout,BUFSIZE,MPI_DOUBLE,MPI_SUM,world);
  nlen = sqrt(bufout[0]);
  plen = sqrt(bufout[1]);
  tlen = sqrt(bufout[2]);
  dotpath = bufout[3];
  dot = bufout[4];
 
  //Compute path related stuff
  lentot = lenuntilIm = meanDist = 0.0;
  double *aux_plen, *plenall;
  aux_plen = new double [nbeads];
  plenall  = new double [nbeads];
  for (int i=0;i<nbeads;i++) aux_plen[i] = plenall[i] = 0.0;
  aux_plen[ibead]=plen; //Length of this segment
  if ( me_w == 0 ) MPI_Allreduce(aux_plen, plenall, nbeads, MPI_DOUBLE, MPI_SUM, rootworld); //Gather all segments, store in array
  MPI_Bcast(plenall,nbeads,MPI_DOUBLE,0,world); //Communicate to all procs
  for (int i = 0; i < ibead+1; i++) lenuntilIm += plenall[i]; // (ibead+1) goes from 1 to nbeads -> i goes from 0 to up to nbeads-1
  for (int i = 0; i < nbeads; i++)  lentot += plenall[i];
  meanDist = lentot/(nbeads -1);
  delete [] aux_plen;
  delete [] plenall;
 
  //Normalize tangent and projection along tangent
  if (tlen>0) {
     double tlen_=1./tlen;
     dot*=tlen_;
     for (int i=0;i<nlocal;i++){
        tangent[i][0]*=tlen_;
        tangent[i][1]*=tlen_;
        tangent[i][2]*=tlen_;
     }
  }

  // Update forces
  if ( ( ibead == 0 ) || ( ibead == nbeads -1  ) ) { //Just relax to local minimum
    for (int i = 0; i < nlocal; i++){
         f[i][0] = f_bead[i][0];
         f[i][1] = f_bead[i][1];
         f[i][2] = f_bead[i][2];
    }
  }

  if ( ( ibead == 0 ) && FreeEndIni  ) { //Keep energy close to starting point
    double prefactor = 0.0;
    if (dot<0) prefactor = -dot - kspringIni*(veng-EIniIni);
    else prefactor = -dot + kspringIni*(veng-EIniIni);
    for (int i = 0; i < nlocal; i++){ 
         f[i][0] += prefactor*tangent[i][0];
         f[i][1] += prefactor*tangent[i][1];
         f[i][2] += prefactor*tangent[i][2];
    }
  }

  if ( ( ibead == nbeads-1 ) && FreeEndFinal ) { //Keep energy close to starting point
    double prefactor = 0.0;
    if (veng<EFinalIni) {
        if (dot<0) prefactor = -dot - kspringFinal*(veng-EFinalIni);
        else prefactor = -dot + kspringFinal*(veng-EFinalIni);
    }
    for (int i = 0; i < nlocal; i++){
         f[i][0] += prefactor*tangent[i][0];
         f[i][1] += prefactor*tangent[i][1];
         f[i][2] += prefactor*tangent[i][2];
    }
  }
 
  if ( ( ibead == nbeads-1 ) && FreeEndFinalWithRespToEIni ) { //Keep energy above that of first image
    double prefactor = 0.0;
    if (veng<vIni) {
        if (dot<0) prefactor = -dot - kspringFinal*(veng-vIni);
        else prefactor = -dot + kspringFinal*(veng-vIni);
    }
    for (int i = 0; i < nlocal; i++){
         f[i][0] += prefactor*tangent[i][0];
         f[i][1] += prefactor*tangent[i][1];
         f[i][2] += prefactor*tangent[i][2];
    }
  }

  if ( ( ibead > 0 ) && ( ibead < nbeads-1 ) ) { //BULK BEADS
    double wgt;
    dotpath = dotpath/(plen*nlen); // cosine of angle between (x_i-x_{i-1}) and (x_{i+1}-x_i)
    wgt = 0.5 *(1+cos(MY_PI * dotpath)); // weight to keep part of perpendicular spring force in case of kinks

    double dotSprTan=0.0; // dot_product(spring_force,tangent)
    for (int i = 0; i < nlocal; i++) dotSprTan += f[i][0]*tangent[i][0] + f[i][1]*tangent[i][1] + f[i][2]*tangent[i][2];
    double aux=0.0;
    MPI_Allreduce(&dotSprTan,&aux,1,MPI_DOUBLE,MPI_SUM,world);
    dotSprTan=aux;
 
    double dotSprTanPerp=0.0; // dot_product(spring_force,tangent)
    if (PerpSpring && (kspringPerp > 0.0)) {
      for (int i = 0; i < nlocal; i++) 
          dotSprTanPerp += SpringFPerp[i][0]*tangent[i][0] + SpringFPerp[i][1]*tangent[i][1] + SpringFPerp[i][2]*tangent[i][2];
      double aux1=0.0;
      MPI_Allreduce(&dotSprTanPerp,&aux1,1,MPI_DOUBLE,MPI_SUM,world);
      dotSprTanPerp=aux1;
    }

    if (FinalAndInterWithRespToEIni && veng<vIni) {
      for (int i = 0; i < nlocal; i++)
        {
          f[i][0] = dotSprTan*tangent[i][0];
          f[i][1] = dotSprTan*tangent[i][1];
          f[i][2] = dotSprTan*tangent[i][2];
        }
    }
    else {

        double pref = 0.0;
        if ( ibead == climber )
           { 
              pref = -2.0*dot;
              dotSprTan = 0.0;
              wgt = 0.0;
           }
        else pref = -dot;

        if (PerpSpring && ( kspringPerp == 0.0 )){
          for (int i = 0; i < nlocal; i++){
            f[i][0] = f_bead[i][0] + pref*tangent[i][0] + dotSprTan*tangent[i][0] + wgt * ( f[i][0] - dotSprTan*tangent[i][0] );
            f[i][1] = f_bead[i][1] + pref*tangent[i][1] + dotSprTan*tangent[i][1] + wgt * ( f[i][1] - dotSprTan*tangent[i][1] );
            f[i][2] = f_bead[i][2] + pref*tangent[i][2] + dotSprTan*tangent[i][2] + wgt * ( f[i][2] - dotSprTan*tangent[i][2] );
          }
        }
        else if (PerpSpring && ( kspringPerp > 0.0 )) {
          for (int i = 0; i < nlocal; i++){
            f[i][0] = f_bead[i][0] + pref*tangent[i][0] + dotSprTan*tangent[i][0] + wgt * ( SpringFPerp[i][0] - dotSprTanPerp*tangent[i][0] );
            f[i][1] = f_bead[i][1] + pref*tangent[i][1] + dotSprTan*tangent[i][1] + wgt * ( SpringFPerp[i][1] - dotSprTanPerp*tangent[i][1] );
            f[i][2] = f_bead[i][2] + pref*tangent[i][2] + dotSprTan*tangent[i][2] + wgt * ( SpringFPerp[i][2] - dotSprTanPerp*tangent[i][2] );
          }
        }
        else {
          for (int i = 0; i < nlocal; i++){
            f[i][0] = f_bead[i][0] + pref*tangent[i][0] + dotSprTan*tangent[i][0];
            f[i][1] = f_bead[i][1] + pref*tangent[i][1] + dotSprTan*tangent[i][1];
            f[i][2] = f_bead[i][2] + pref*tangent[i][2] + dotSprTan*tangent[i][2];
          }
        }
    }

  }//BULK BEADS

  //Free aux arrays
  if (SpringFPerp) { for(int i=0; i<nlocal; i++) {if (SpringFPerp[i]) delete [] SpringFPerp[i];}
              delete [] SpringFPerp;}
  if (tangent) { for(int i=0; i<nlocal; i++) {if (tangent[i]) delete [] tangent[i];}
              delete [] tangent;}
  if (f_fwd){ for(int i=0; i<nlocal; i++) {if (f_fwd[i]) delete [] f_fwd[i];}
              delete [] f_fwd;}
  if (f_bwd){ for(int i=0; i<nlocal; i++) {if (f_bwd[i]) delete [] f_bwd[i];}
              delete [] f_bwd;}
  if (x_bead){ for(int i=0; i<nlocal; i++) {if (x_bead[i]) delete [] x_bead[i];}
               delete [] x_bead;}
  if (f_bead){ for(int i=0; i<nlocal; i++) {if (f_bead[i]) delete [] f_bead[i];}
               delete [] f_bead;}
  if (eta_n){ for(int i=0; i<nlocal; i++) {if (eta_n[i]) delete [] eta_n[i];}
              delete [] eta_n;}
  if (eta_l){ for(int i=0; i<nlocal; i++) {if (eta_l[i]) delete [] eta_l[i];}
              delete [] eta_l;}

}//END neb_test_force


/* ----------------------------------------------------------------------
   NEB-like force
------------------------------------------------------------------------- */

void FixPATH_DYNAMICS::neb_force()
{
  spring_energy = 0.0;

  //Communicate epot for smart tangent and 'end' option
  double vprev,vnext; //Potential energy of previous and next bead
  int procprev,procnext;
  double vIni=0.0;

  pe->compute_scalar();
  vprev = vnext = veng = pe->scalar;
  // trigger potential energy computation on next timestep
  pe->addstep(update->ntimestep+1);

  if (ibead > 0) procprev = universe->root_proc[ibead-1];
  else procprev = -1;

  if (ibead < nbeads-1) procnext = universe->root_proc[ibead+1];
  else procnext = -1;

  if (ibead < nbeads-1 && me_w == 0) MPI_Send(&veng,1,MPI_DOUBLE,procnext,0,uworld);

  if (ibead > 0 && me_w == 0) MPI_Recv(&vprev,1,MPI_DOUBLE,procprev,0,uworld,MPI_STATUS_IGNORE);

  if (ibead > 0 && me_w == 0) MPI_Send(&veng,1,MPI_DOUBLE,procprev,0,uworld);

  if (ibead < nbeads-1 && me_w == 0) MPI_Recv(&vnext,1,MPI_DOUBLE,procnext,0,uworld,MPI_STATUS_IGNORE);

  MPI_Bcast(&vprev,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&vnext,1,MPI_DOUBLE,0,world);
   
  if ( ibead == 0 ) vIni=veng;
  if ( me_w == 0 ) MPI_Bcast(&vIni,1,MPI_DOUBLE,0,rootworld);
  MPI_Bcast(&vIni,1,MPI_DOUBLE,0,world);
  //End epot comm

  if ( (ibead == nbeads-1) && (update->ntimestep == 0)) EFinalIni = veng;
  if ( (ibead == 0 ) && (update->ntimestep == 0)) EIniIni = veng;

  //pos vel force owned by this proc
  double **x = atom->x;
  double **f = atom->f;
  double **v = atom->v;
  int* type = atom->type;
  int nlocal = atom->nlocal;
  //Positions of atoms of the two adjacent replicas
  double* xlast = buf_beads[x_last];
  double* xnext = buf_beads[x_next];
  //Forces acting on atoms of the two adjacent replicas
  double* flast = buf_force[x_last];
  //double* fnext = buf_force[x_next];
  //Some aux vars
  double maxeta2,eps,maxx;
  double dxn,dyn,dzn,dxp,dyp,dzp;
  maxeta2 = eps = maxx = 0.0;
  dxn = dyn = dzn = dxp = dyp = dzp = 0.0;
  dotpath = dot = tlen = plen = nlen = 0.0;

  //nlocal can change. I need to reallocate at every time step (check)
  SpringFPerp = new double* [nlocal];
  for (int i=0;i<nlocal;i++) SpringFPerp[i] = new double [3];
  tangent = new double* [nlocal];
  for (int i=0;i<nlocal;i++) tangent[i] = new double [3];
  eta_l = new double* [nlocal];
  for (int i=0;i<nlocal;i++) eta_l[i] = new double [3];
  eta_n = new double* [nlocal];
  for (int i=0;i<nlocal;i++) eta_n[i] = new double [3];
  //Allocate aux x and f
  f_fwd = new double* [nlocal];
  for (int i=0;i<nlocal;i++) f_fwd[i] = new double [3];
  f_bwd = new double* [nlocal];
  for (int i=0;i<nlocal;i++) f_bwd[i] = new double [3];
  x_bead = new double* [nlocal];
  for (int i=0;i<nlocal;i++) x_bead[i] = new double [3];
  f_bead = new double* [nlocal];
  for (int i=0;i<nlocal;i++) f_bead[i] = new double [3];

  //Store current positions
  for (int i=0;i<nlocal;i++){ x_bead[i][0]=x[i][0]; x_bead[i][1]=x[i][1]; x_bead[i][2]=x[i][2]; }
  //Store current force-field forces
  for (int i=0;i<nlocal;i++){ f_bead[i][0]=f[i][0]; f_bead[i][1]=f[i][1]; f_bead[i][2]=f[i][2]; }

  if ( ibead == 0 )//First bead moves according only to f_bead
  {                //To held it fixed use "partition yes 1 fix set_force 0 0 0" in the input, after fix path_dynamics.
	  for(int i=0; i<nlocal; i++)
	  {
             dxn = xnext[0] - x_bead[i][0];
             dyn = xnext[1] - x_bead[i][1];
             dzn = xnext[2] - x_bead[i][2];
             domain->minimum_image(dxn,dyn,dzn);
             //Needed for spring energy:
             eta_n[i][0] = dxn - dt_md[type[i]] * f_bead[i][0] * force->ftm2v;
             eta_n[i][1] = dyn - dt_md[type[i]] * f_bead[i][1] * force->ftm2v;
             eta_n[i][2] = dzn - dt_md[type[i]] * f_bead[i][2] * force->ftm2v;
             //NEB
             nlen += dxn*dxn + dyn*dyn + dzn*dzn;
             tangent[i][0]=dxn; tangent[i][1]=dyn; tangent[i][2]=dzn;
             tlen += dxn*dxn + dyn*dyn + dzn*dzn;
             dot += f_bead[i][0]*tangent[i][0] + f_bead[i][1]*tangent[i][1] + f_bead[i][2]*tangent[i][2];
             //NEB
             double msd = eta_n[i][0]*eta_n[i][0] + eta_n[i][1]*eta_n[i][1] + eta_n[i][2]*eta_n[i][2];
       	     if (ParaSpring) spring_energy += 0.5 * kspringPara * msd * force->mvv2e;
             else spring_energy += 0.5 * md_2dt[type[i]] * msd * force->mvv2e;
             xnext += 3;
	  }
  }

  else if ( ibead == nbeads-1 ) //Last bead moves according only to f_bead
  {                             //To held it fixed use "partition yes N fix set_force 0 0 0" in the input, after fix path_dynamics.
          for(int i=0; i<nlocal; i++)
          {
             dxp = x_bead[i][0] - xlast[0];
             dyp = x_bead[i][1] - xlast[1];
             dzp = x_bead[i][2] - xlast[2];
             domain->minimum_image(dxp,dyp,dzp);
             //NEB
             plen += dxp*dxp + dyp*dyp + dzp*dzp;
             tangent[i][0]=dxp; tangent[i][1]=dyp; tangent[i][2]=dzp;
             tlen += dxp*dxp + dyp*dyp + dzp*dzp;
             dot += f_bead[i][0]*tangent[i][0] + f_bead[i][1]*tangent[i][1] + f_bead[i][2]*tangent[i][2];
             //NEB
             xlast += 3;
          }
  }
  else // bulk beads: I need to compute also the spring forces
  {
    double vmax = MAX(fabs(vnext-veng),fabs(vprev-veng));
    double vmin = MIN(fabs(vnext-veng),fabs(vprev-veng));
	  for(int i=0; i<nlocal; i++)
          {
            dxn = xnext[0] - x_bead[i][0];
            dyn = xnext[1] - x_bead[i][1];
            dzn = xnext[2] - x_bead[i][2];
            domain->minimum_image(dxn,dyn,dzn);
            eta_n[i][0] = dxn - dt_md[type[i]] * f_bead[i][0] * force->ftm2v;
            eta_n[i][1] = dyn - dt_md[type[i]] * f_bead[i][1] * force->ftm2v;
            eta_n[i][2] = dzn - dt_md[type[i]] * f_bead[i][2] * force->ftm2v;
            dxp = x_bead[i][0] - xlast[0];
            dyp = x_bead[i][1] - xlast[1];
            dzp = x_bead[i][2] - xlast[2];
            domain->minimum_image(dxp,dyp,dzp);
	    eta_l[i][0] = dxp - dt_md[type[i]] * flast[0] * force->ftm2v;
            eta_l[i][1] = dyp - dt_md[type[i]] * flast[1] * force->ftm2v;
            eta_l[i][2] = dzp - dt_md[type[i]] * flast[2] * force->ftm2v;
            //NEB
	    if (vnext > veng && veng > vprev) {
	      tangent[i][0] = dxn;
	      tangent[i][1] = dyn;
	      tangent[i][2] = dzn;
	    } else if (vnext < veng && veng < vprev) {
	      tangent[i][0] = dxp;
	      tangent[i][1] = dyp;
	      tangent[i][2] = dzp;
	    } else {
	        if (vnext > vprev) {
	          tangent[i][0] = vmax*dxn + vmin*dxp;
	          tangent[i][1] = vmax*dyn + vmin*dyp;
	          tangent[i][2] = vmax*dzn + vmin*dzp;
	        } else if (vnext < vprev) {
	          tangent[i][0] = vmin*dxn + vmax*dxp;
	          tangent[i][1] = vmin*dyn + vmax*dyp;
	          tangent[i][2] = vmin*dzn + vmax*dzp;
	        } else { // vnext == vprev, e.g. for potentials that do not compute an energy
	          tangent[i][0] = dxn + dxp;
	          tangent[i][1] = dyn + dyp;
	          tangent[i][2] = dzn + dzp;
	        }
	    }
	    plen += dxp*dxp + dyp*dyp + dzp*dzp;
  	    nlen += dxn*dxn + dyn*dyn + dzn*dzn;
            tlen += tangent[i][0]*tangent[i][0] + tangent[i][1]*tangent[i][1] + tangent[i][2]*tangent[i][2];
            dotpath += dxp*dxn + dyp*dyn + dzp*dzn;
            dot += f_bead[i][0]*tangent[i][0] + f_bead[i][1]*tangent[i][1] + f_bead[i][2]*tangent[i][2];
            //NEB
            double msd = eta_n[i][0]*eta_n[i][0] + eta_n[i][1]*eta_n[i][1] + eta_n[i][2]*eta_n[i][2];
            if (msd>maxeta2) maxeta2=msd;
            if (ParaSpring) spring_energy += 0.5 * kspringPara * msd * force->mvv2e;
            else spring_energy += 0.5 * md_2dt[type[i]] * msd * force->mvv2e;
            flast += 3;
            xlast += 3;
            xnext += 3;
          }

          MPI_Allreduce(&maxeta2, &maxx, 1, MPI_DOUBLE, MPI_MAX, world);
          eps=epsilon/sqrt(maxx); //Room for improvement here

          // Displace fwd
          for (int i=0;i<nlocal;i++){
                  x[i][0] = x_bead[i][0] + eps * eta_n[i][0];
                  x[i][1] = x_bead[i][1] + eps * eta_n[i][1];
                  x[i][2] = x_bead[i][2] + eps * eta_n[i][2];
          }
          compute_force();
          // Store f_fwd
          for (int i=0;i<nlocal;i++){ f_fwd[i][0]=f[i][0]; f_fwd[i][1]=f[i][1]; f_fwd[i][2]=f[i][2]; }
          // Displace bwd
          for (int i=0;i<nlocal;i++){
                  x[i][0] = x_bead[i][0] - eps * eta_n[i][0];
                  x[i][1] = x_bead[i][1] - eps * eta_n[i][1];
                  x[i][2] = x_bead[i][2] - eps * eta_n[i][2];
          }
          compute_force();
          // Store f_bwd
          for (int i=0;i<nlocal;i++){ f_bwd[i][0]=f[i][0]; f_bwd[i][1]=f[i][1]; f_bwd[i][2]=f[i][2]; }
          // Update spring forces
          for (int i=0;i<nlocal;i++)
          {   //f = ~springs~ + ~hessian~
              if (ParaSpring) {
                f[i][0] = kspringPara * ( eta_n[i][0] - eta_l[i][0] ) * force->mvv2e + 0.25 * kspringPara * ( f_fwd[i][0] - f_bwd[i][0] ) / eps / md_2dt[type[i]];
                f[i][1] = kspringPara * ( eta_n[i][1] - eta_l[i][1] ) * force->mvv2e + 0.25 * kspringPara * ( f_fwd[i][1] - f_bwd[i][1] ) / eps / md_2dt[type[i]];
                f[i][2] = kspringPara * ( eta_n[i][2] - eta_l[i][2] ) * force->mvv2e + 0.25 * kspringPara * ( f_fwd[i][2] - f_bwd[i][2] ) / eps / md_2dt[type[i]];
              }
              else{
                f[i][0] = md_2dt[type[i]] * ( eta_n[i][0] - eta_l[i][0] ) * force->mvv2e + 0.25 * ( f_fwd[i][0] - f_bwd[i][0] ) / eps;
                f[i][1] = md_2dt[type[i]] * ( eta_n[i][1] - eta_l[i][1] ) * force->mvv2e + 0.25 * ( f_fwd[i][1] - f_bwd[i][1] ) / eps;
                f[i][2] = md_2dt[type[i]] * ( eta_n[i][2] - eta_l[i][2] ) * force->mvv2e + 0.25 * ( f_fwd[i][2] - f_bwd[i][2] ) / eps;
              }
              if (PerpSpring) {
                SpringFPerp[i][0] = kspringPerp * ( eta_n[i][0] - eta_l[i][0] ) * force->mvv2e + 0.25 * kspringPerp * ( f_fwd[i][0] - f_bwd[i][0] ) / eps / md_2dt[type[i]];
                SpringFPerp[i][1] = kspringPerp * ( eta_n[i][1] - eta_l[i][1] ) * force->mvv2e + 0.25 * kspringPerp * ( f_fwd[i][1] - f_bwd[i][1] ) / eps / md_2dt[type[i]];
                SpringFPerp[i][2] = kspringPerp * ( eta_n[i][2] - eta_l[i][2] ) * force->mvv2e + 0.25 * kspringPerp * ( f_fwd[i][2] - f_bwd[i][2] ) / eps / md_2dt[type[i]];
              }
          }
  }
  //At this point, f_bead[][] contains the forces due to the force-field, f[][] contains the spring forces and optionally SpringFPerp contains the spring forces used only in perp direction

  //Restore positions
  for (int i=0;i<nlocal;i++){ x[i][0]=x_bead[i][0]; x[i][1]=x_bead[i][1]; x[i][2]=x_bead[i][2]; }

  //Sum spring_energy from all procs in this world
  double aux_E;
  aux_E=0.0;
  MPI_Allreduce(&spring_energy, &aux_E, 1, MPI_DOUBLE, MPI_SUM, world);
  spring_energy=aux_E;
  //Potential energy
  potential_energy=veng;

  //Reduce
  double bufin[BUFSIZE], bufout[BUFSIZE];
  bufin[0] = nlen;
  bufin[1] = plen;
  bufin[2] = tlen;
  bufin[3] = dotpath;
  bufin[4] = dot;
  MPI_Allreduce(bufin,bufout,BUFSIZE,MPI_DOUBLE,MPI_SUM,world);
  nlen = sqrt(bufout[0]);
  plen = sqrt(bufout[1]);
  tlen = sqrt(bufout[2]);
  dotpath = bufout[3];
  dot = bufout[4];
 
  //Compute path related stuff
  lentot = lenuntilIm = meanDist = 0.0;
  double *aux_plen, *plenall;
  aux_plen = new double [nbeads];
  plenall  = new double [nbeads];
  for (int i=0;i<nbeads;i++) aux_plen[i] = plenall[i] = 0.0;
  aux_plen[ibead]=plen; //Length of this segment
  if ( me_w == 0 ) MPI_Allreduce(aux_plen, plenall, nbeads, MPI_DOUBLE, MPI_SUM, rootworld); //Gather all segments, store in array
  MPI_Bcast(plenall,nbeads,MPI_DOUBLE,0,world); //Communicate to all procs
  for (int i = 0; i < ibead+1; i++) lenuntilIm += plenall[i]; // (ibead+1) goes from 1 to nbeads -> i goes from 0 to up to nbeads-1
  for (int i = 0; i < nbeads; i++)  lentot += plenall[i];
  meanDist = lentot/(nbeads -1);
  delete [] aux_plen;
  delete [] plenall;
 
  //Normalize tangent and projection along tangent
  if (tlen>0) {
     double tlen_=1./tlen;
     dot*=tlen_;
     for (int i=0;i<nlocal;i++){
        tangent[i][0]*=tlen_;
        tangent[i][1]*=tlen_;
        tangent[i][2]*=tlen_;
     }
  }

  // Update forces
  if ( ( ibead == 0 ) || ( ibead == nbeads -1  ) ) { //Just relax to local minimum
    for (int i = 0; i < nlocal; i++){
         f[i][0] = f_bead[i][0];
         f[i][1] = f_bead[i][1];
         f[i][2] = f_bead[i][2];
    }
  }

  if ( ( ibead == 0 ) && FreeEndIni  ) { //Keep energy close to starting point
    double prefactor = 0.0;
    if (dot<0) prefactor = -dot - kspringIni*(veng-EIniIni);
    else prefactor = -dot + kspringIni*(veng-EIniIni);
    for (int i = 0; i < nlocal; i++){ 
         f[i][0] += prefactor*tangent[i][0];
         f[i][1] += prefactor*tangent[i][1];
         f[i][2] += prefactor*tangent[i][2];
    }
  }

  if ( ( ibead == nbeads-1 ) && FreeEndFinal ) { //Keep energy close to starting point
    double prefactor = 0.0;
    if (veng<EFinalIni) {
        if (dot<0) prefactor = -dot - kspringFinal*(veng-EFinalIni);
        else prefactor = -dot + kspringFinal*(veng-EFinalIni);
    }
    for (int i = 0; i < nlocal; i++){
         f[i][0] += prefactor*tangent[i][0];
         f[i][1] += prefactor*tangent[i][1];
         f[i][2] += prefactor*tangent[i][2];
    }
  }
 
  if ( ( ibead == nbeads-1 ) && FreeEndFinalWithRespToEIni ) { //Keep energy above that of first image
    double prefactor = 0.0;
    if (veng<vIni) {
        if (dot<0) prefactor = -dot - kspringFinal*(veng-vIni);
        else prefactor = -dot + kspringFinal*(veng-vIni);
    }
    for (int i = 0; i < nlocal; i++){
         f[i][0] += prefactor*tangent[i][0];
         f[i][1] += prefactor*tangent[i][1];
         f[i][2] += prefactor*tangent[i][2];
    }
  }

  if ( ( ibead > 0 ) && ( ibead < nbeads-1 ) ) { //BULK BEADS
    double wgt;
    dotpath = dotpath/(plen*nlen); // cosine of angle between (x_i-x_{i-1}) and (x_{i+1}-x_i)
    wgt = 0.5 *(1+cos(MY_PI * dotpath)); // weight to keep part of perpendicular spring force in case of kinks

    double dotSprTan=0.0; // dot_product(spring_force,tangent)
    for (int i = 0; i < nlocal; i++) dotSprTan += f[i][0]*tangent[i][0] + f[i][1]*tangent[i][1] + f[i][2]*tangent[i][2];
    double aux=0.0;
    MPI_Allreduce(&dotSprTan,&aux,1,MPI_DOUBLE,MPI_SUM,world);
    dotSprTan=aux;
 
    double dotSprTanPerp=0.0; // dot_product(spring_force,tangent)
    if (PerpSpring && (kspringPerp > 0.0)) {
      for (int i = 0; i < nlocal; i++) 
          dotSprTanPerp += SpringFPerp[i][0]*tangent[i][0] + SpringFPerp[i][1]*tangent[i][1] + SpringFPerp[i][2]*tangent[i][2];
      double aux1=0.0;
      MPI_Allreduce(&dotSprTanPerp,&aux1,1,MPI_DOUBLE,MPI_SUM,world);
      dotSprTanPerp=aux1;
    }

    if (FinalAndInterWithRespToEIni && veng<vIni) {
      for (int i = 0; i < nlocal; i++)
        {
          f[i][0] = dotSprTan*tangent[i][0];
          f[i][1] = dotSprTan*tangent[i][1];
          f[i][2] = dotSprTan*tangent[i][2];
        }
    }
    else {

        double pref = 0.0;
 	if ( ibead == climber )
           {
              pref = -2.0*dot;
              dotSprTan = 0.0;
              wgt = 0.0;
           }
        else pref = -dot;

        if (PerpSpring && ( kspringPerp == 0.0 )){
          for (int i = 0; i < nlocal; i++){
            f[i][0] = f_bead[i][0] + pref*tangent[i][0] + dotSprTan*tangent[i][0] + wgt * ( f[i][0] - dotSprTan*tangent[i][0] );
            f[i][1] = f_bead[i][1] + pref*tangent[i][1] + dotSprTan*tangent[i][1] + wgt * ( f[i][1] - dotSprTan*tangent[i][1] );
            f[i][2] = f_bead[i][2] + pref*tangent[i][2] + dotSprTan*tangent[i][2] + wgt * ( f[i][2] - dotSprTan*tangent[i][2] );
          }
        }
        else if (PerpSpring && ( kspringPerp > 0.0 )) {
          for (int i = 0; i < nlocal; i++){
            f[i][0] = f_bead[i][0] + pref*tangent[i][0] + dotSprTan*tangent[i][0] + wgt * ( SpringFPerp[i][0] - dotSprTanPerp*tangent[i][0] );
            f[i][1] = f_bead[i][1] + pref*tangent[i][1] + dotSprTan*tangent[i][1] + wgt * ( SpringFPerp[i][1] - dotSprTanPerp*tangent[i][1] );
            f[i][2] = f_bead[i][2] + pref*tangent[i][2] + dotSprTan*tangent[i][2] + wgt * ( SpringFPerp[i][2] - dotSprTanPerp*tangent[i][2] );
          }
        }
        else {
          for (int i = 0; i < nlocal; i++){
            f[i][0] = f_bead[i][0] + pref*tangent[i][0] + dotSprTan*tangent[i][0];
            f[i][1] = f_bead[i][1] + pref*tangent[i][1] + dotSprTan*tangent[i][1];
            f[i][2] = f_bead[i][2] + pref*tangent[i][2] + dotSprTan*tangent[i][2];
          }
        }
    }

  }//BULK BEADS

  //Free aux arrays
  if (SpringFPerp) { for(int i=0; i<nlocal; i++) {if (SpringFPerp[i]) delete [] SpringFPerp[i];}
              delete [] SpringFPerp;}
  if (tangent) { for(int i=0; i<nlocal; i++) {if (tangent[i]) delete [] tangent[i];}
              delete [] tangent;}
  if (f_fwd){ for(int i=0; i<nlocal; i++) {if (f_fwd[i]) delete [] f_fwd[i];}
              delete [] f_fwd;}
  if (f_bwd){ for(int i=0; i<nlocal; i++) {if (f_bwd[i]) delete [] f_bwd[i];}
              delete [] f_bwd;}
  if (x_bead){ for(int i=0; i<nlocal; i++) {if (x_bead[i]) delete [] x_bead[i];}
               delete [] x_bead;}
  if (f_bead){ for(int i=0; i<nlocal; i++) {if (f_bead[i]) delete [] f_bead[i];}
               delete [] f_bead;}
  if (eta_n){ for(int i=0; i<nlocal; i++) {if (eta_n[i]) delete [] eta_n[i];}
              delete [] eta_n;}
  if (eta_l){ for(int i=0; i<nlocal; i++) {if (eta_l[i]) delete [] eta_l[i];}
              delete [] eta_l;}
}//END neb_force


/* ----------------------------------------------------------------------
   Comm operations
------------------------------------------------------------------------- */

void FixPATH_DYNAMICS::comm_init()
{
  if(size_plan)
  {
    delete [] plan_send;
    delete [] plan_recv;
  }

  size_plan = 2;
  plan_send = new int [2];
  plan_recv = new int [2];
  mode_index = new int [2];

  int rank_last = me_u - np_w;
  int rank_next = me_u + np_w;
  if(rank_last<0) rank_last += np_u;
  if(rank_next>=np_u) rank_next -= np_u;

  plan_send[0] = rank_next; plan_send[1] = rank_last;
  plan_recv[0] = rank_last; plan_recv[1] = rank_next;

  mode_index[0] = 0; mode_index[1] = 1;
  x_last = 1; x_next = 0;

  if(buf_beads)
  {
    for(int i=0; i<size_plan; i++) if(buf_beads[i]) delete [] buf_beads[i];
    delete [] buf_beads;
  }

  buf_beads = new double* [size_plan];
  for(int i=0; i<size_plan; i++) buf_beads[i] = NULL;

  if(buf_force)
  {
    for(int i=0; i<size_plan; i++) if(buf_force[i]) delete [] buf_force[i];
    delete [] buf_force;
  }

  buf_force = new double* [size_plan];
  for(int i=0; i<size_plan; i++) buf_force[i] = NULL;
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::comm_exec(double **ptr)
{
  int nlocal = atom->nlocal;

  if(nlocal > max_nlocal)
  {
    max_nlocal = nlocal+200;
    int size = sizeof(double) * max_nlocal * 3;
    buf_recv = (double*) memory->srealloc(buf_recv, size, "FixPATH_DYNAMICS:x_recv");

    for(int i=0; i<size_plan; i++){
      buf_beads[i] = (double*) memory->srealloc(buf_beads[i], size, "FixPATH_DYNAMICS:x_beads[i]");
      buf_force[i] = (double*) memory->srealloc(buf_force[i], size, "FixPATH_DYNAMICS:x_force[i]"); //Also allocate force buffer
    }
  }

  // go over comm plans

  for(int iplan = 0; iplan<size_plan; iplan++)
  {
    // sendrecv nlocal

    int nsend;

    MPI_Sendrecv( &(nlocal), 1, MPI_INT, plan_send[iplan], 0,
                  &(nsend),  1, MPI_INT, plan_recv[iplan], 0, uworld, MPI_STATUS_IGNORE);

    // allocate arrays

    if(nsend > max_nsend)
    {
      max_nsend = nsend+200;
      tag_send = (int*) memory->srealloc(tag_send, sizeof(int)*max_nsend, "FixPATH_DYNAMICS:tag_send");
      buf_send = (double*) memory->srealloc(buf_send, sizeof(double)*max_nsend*3, "FixPATH_DYNAMICS:x_send");
    }

    // send tags

    MPI_Sendrecv( atom->tag, nlocal, MPI_INT, plan_send[iplan], 0,
                  tag_send,  nsend,  MPI_INT, plan_recv[iplan], 0, uworld, MPI_STATUS_IGNORE);

    // wrap positions

    double *wrap_ptr = buf_send;
    int ncpy = sizeof(double)*3;

    for(int i=0; i<nsend; i++)
    {
      int index = atom->map(tag_send[i]);

      if(index<0)
      {
        char error_line[256];

        sprintf(error_line, "Atom " TAGINT_FORMAT " is missing at world [%d] "
                "rank [%d] required by  rank [%d] (" TAGINT_FORMAT ", "
                TAGINT_FORMAT ", " TAGINT_FORMAT ").\n",tag_send[i],
                ibead, me_w, plan_recv[iplan],
                atom->tag[0], atom->tag[1], atom->tag[2]);

        error->universe_one(FLERR,error_line);
      }

      memcpy(wrap_ptr, ptr[index], ncpy);
      wrap_ptr += 3;
    }

    // sendrecv x

    MPI_Sendrecv( buf_send, nsend*3,  MPI_DOUBLE, plan_recv[iplan], 0,
                  buf_recv, nlocal*3, MPI_DOUBLE, plan_send[iplan], 0, uworld, MPI_STATUS_IGNORE);

    // copy x

    memcpy(buf_beads[mode_index[iplan]], buf_recv, sizeof(double)*nlocal*3);
  }
}

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

int FixPATH_DYNAMICS::pack_forward_comm(int n, int *list, double *buf,
                             int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = comm_ptr[j][0];
    buf[m++] = comm_ptr[j][1];
    buf[m++] = comm_ptr[j][2];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    comm_ptr[i][0] = buf[m++];
    comm_ptr[i][1] = buf[m++];
    comm_ptr[i][2] = buf[m++];
  }
}

/* ----------------------------------------------------------------------
   Memory operations
------------------------------------------------------------------------- */

double FixPATH_DYNAMICS::memory_usage()
{
  double bytes = 0;
  bytes = atom->nmax * size_peratom_cols * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::grow_arrays(int nmax)
{
  if (nmax==0) return;
  int count = nmax*3;

  memory->grow(array_atom, nmax, size_peratom_cols, "FixPATH_DYNAMICS::array_atom");
  memory->grow(nhc_eta,        count, nhc_nchain,   "FixPATH_DYNAMICS::nh_eta");
  memory->grow(nhc_eta_dot,    count, nhc_nchain+1, "FixPATH_DYNAMICS::nh_eta_dot");
  memory->grow(nhc_eta_dotdot, count, nhc_nchain,   "FixPATH_DYNAMICS::nh_eta_dotdot");
  memory->grow(nhc_eta_mass,   count, nhc_nchain,   "FixPATH_DYNAMICS::nh_eta_mass");
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::copy_arrays(int i, int j, int delflag)
{
  int i_pos = i*3;
  int j_pos = j*3;

  memcpy(nhc_eta       [j_pos], nhc_eta       [i_pos], nhc_size_one_1);
  memcpy(nhc_eta_dot   [j_pos], nhc_eta_dot   [i_pos], nhc_size_one_2);
  memcpy(nhc_eta_dotdot[j_pos], nhc_eta_dotdot[i_pos], nhc_size_one_1);
  memcpy(nhc_eta_mass  [j_pos], nhc_eta_mass  [i_pos], nhc_size_one_1);
}

/* ---------------------------------------------------------------------- */

int FixPATH_DYNAMICS::pack_exchange(int i, double *buf)
{
  int offset=0;
  int pos = i * 3;

  memcpy(buf+offset, nhc_eta[pos],        nhc_size_one_1); offset += nhc_offset_one_1;
  memcpy(buf+offset, nhc_eta_dot[pos],    nhc_size_one_2); offset += nhc_offset_one_2;
  memcpy(buf+offset, nhc_eta_dotdot[pos], nhc_size_one_1); offset += nhc_offset_one_1;
  memcpy(buf+offset, nhc_eta_mass[pos],   nhc_size_one_1); offset += nhc_offset_one_1;

  return size_peratom_cols;
}

/* ---------------------------------------------------------------------- */

int FixPATH_DYNAMICS::unpack_exchange(int nlocal, double *buf)
{
  int offset=0;
  int pos = nlocal*3;

  memcpy(nhc_eta[pos],        buf+offset, nhc_size_one_1); offset += nhc_offset_one_1;
  memcpy(nhc_eta_dot[pos],    buf+offset, nhc_size_one_2); offset += nhc_offset_one_2;
  memcpy(nhc_eta_dotdot[pos], buf+offset, nhc_size_one_1); offset += nhc_offset_one_1;
  memcpy(nhc_eta_mass[pos],   buf+offset, nhc_size_one_1); offset += nhc_offset_one_1;

  return size_peratom_cols;
}

/* ---------------------------------------------------------------------- */

int FixPATH_DYNAMICS::pack_restart(int i, double *buf)
{
  int offset=0;
  int pos = i * 3;
  buf[offset++] = size_peratom_cols+1;

  memcpy(buf+offset, nhc_eta[pos],        nhc_size_one_1); offset += nhc_offset_one_1;
  memcpy(buf+offset, nhc_eta_dot[pos],    nhc_size_one_2); offset += nhc_offset_one_2;
  memcpy(buf+offset, nhc_eta_dotdot[pos], nhc_size_one_1); offset += nhc_offset_one_1;
  memcpy(buf+offset, nhc_eta_mass[pos],   nhc_size_one_1); offset += nhc_offset_one_1;

  return size_peratom_cols+1;
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values

  int m = 0;
  for (int i=0; i<nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  int pos = nlocal * 3;

  memcpy(nhc_eta[pos],        extra[nlocal]+m, nhc_size_one_1); m += nhc_offset_one_1;
  memcpy(nhc_eta_dot[pos],    extra[nlocal]+m, nhc_size_one_2); m += nhc_offset_one_2;
  memcpy(nhc_eta_dotdot[pos], extra[nlocal]+m, nhc_size_one_1); m += nhc_offset_one_1;
  memcpy(nhc_eta_mass[pos],   extra[nlocal]+m, nhc_size_one_1); m += nhc_offset_one_1;

  nhc_ready = true;
}

/* ---------------------------------------------------------------------- */

int FixPATH_DYNAMICS::maxsize_restart()
{
  return size_peratom_cols+1;
}

/* ---------------------------------------------------------------------- */

int FixPATH_DYNAMICS::size_restart(int nlocal)
{
  return size_peratom_cols+1;
}

/*----------------------------------------------------------------------- */

void FixPATH_DYNAMICS::compute_force()
{

comm->forward_comm();
force_clear();
if (n_pre_force) modify->pre_force(0);
if (pair_compute_flag) force->pair->compute(0,0);
if (atom->molecular) {
        if (force->bond) force->bond->compute(0,0);
        if (force->angle) force->angle->compute(0,0);
        if (force->dihedral) force->dihedral->compute(0,0);
        if (force->improper) force->improper->compute(0,0);
}
if (kspace_compute_flag) force->kspace->compute(0,0);
if (n_pre_reverse) modify->pre_reverse(0,0);
if (force->newton) comm->reverse_comm();
//if (n_post_force) modify->post_force(vflag);//no fixes with post force are allowed

}

/*----------------------------------------------------------------------- */
void FixPATH_DYNAMICS::force_clear()
{
  size_t nbytes;

  // clear force on all particles
  // if either newton flag is set, also include ghosts
  // when using threads always clear all forces.

  int nlocal = atom->nlocal;
  //int torqueflag = 0;
  //int extraflag = 0;
  //if (atom->torque_flag) torqueflag = 1;
  //if (atom->avec->forceclearflag) extraflag = 1;

  if (neighbor->includegroup == 0) {
    nbytes = sizeof(double) * nlocal;
    if (force->newton) nbytes += sizeof(double) * atom->nghost;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

  // neighbor includegroup flag is set
  // clear force only on initial nfirst particles
  // if either newton flag is set, also include ghosts
} else {
    nbytes = sizeof(double) * atom->nfirst;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

    if (force->newton) {
      nbytes = sizeof(double) * atom->nghost;

      if (nbytes) {
        memset(&atom->f[nlocal][0],0,3*nbytes);
        if (torqueflag) memset(&atom->torque[nlocal][0],0,3*nbytes);
        if (extraflag) atom->avec->force_clear(nlocal,nbytes);
      }
    }
  }
}

/*----------------------------------------------------------------------- */

void FixPATH_DYNAMICS::compute_kinetic_energy()
{
   int nlocal = atom->nlocal;
   double **v = atom->v;
   int* type = atom->type;
   kinetic_energy = 0.0;
   for(int i=0; i<nlocal; i++) {
   kinetic_energy += 0.5 * fmass[type[i]] * ( v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2] ) * force->mvv2e; 
                               }
   double aux_E;
   aux_E=0.0;
   MPI_Allreduce(&kinetic_energy, &aux_E, 1, MPI_DOUBLE, MPI_SUM, world);
   kinetic_energy=aux_E;
}

/* ---------------------------------------------------------------------- */

void FixPATH_DYNAMICS::pimd_fill(double **ptr)
{
  comm_ptr = ptr;
  comm->forward_comm(this);
}

/* ---------------------------------------------------------------------- */

double FixPATH_DYNAMICS::compute_vector(int n)
{

  compute_kinetic_energy();
  if(n==0) { return spring_energy; }
  else if(n==1) { return kinetic_energy; }
  else if(n==2) { return potential_energy; }
  else if(n==3) { return lenuntilIm; }
  else if(n==4) { return plen; }
  else if(n==5) { return lentot; }
  else if(n==6) { return meanDist; }
  else return 0.0;

}

double FixPATH_DYNAMICS::compute_scalar()
{
	double OM_action = 0.0;
	double OM_ibead  = 0.0;
	if (ibead==0){
		OM_ibead = spring_energy + potential_energy;
	}
	else
	{
		OM_ibead = spring_energy;
	}
	//Note: spring_energy contains the contribution to the elastic part of the OM action of one bead,
	//      already reduced over all the ranks within its world.
	//To compute the OM action, I need to reduce OM_ibead across the universe:
	MPI_Allreduce(&OM_ibead, &OM_action, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
	return OM_action - potential_energy;
	//Using 'fix_modify fix_path_md_id energy yes', this will be added to the potential energy.
	//Hence, the potential energy of every bead as stored in LAMMPS will be equal to the OM_action of the whole polymer.
	//This allows to use the ENERGY action of plumed (no need to modify it) to perform metadynamics of paths in the
	//well tempered ensemble.
}

/*----------------------------------------------------------------------- */

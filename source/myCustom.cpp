/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2019 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "ActionRegister.h"
#include "Function.h"
#include "../tools/Communicator.h"
#include "lepton/Lepton.h"

using namespace std;

namespace PLMD {
namespace function {

//+PLUMEDOC FUNCTION CUSTOM
/*
Calculate a combination of variables using a custom expression.

This action computes an  arbitrary function of one or more
collective variables. Arguments are chosen with the ARG keyword,
and the function is provided with the FUNC string. Notice that this
string should contain no space. Within FUNC, one can refer to the
arguments as x,y,z, and t (up to four variables provided as ARG).
This names can be customized using the VAR keyword (see examples below).

This function is implemented using the Lepton library, that allows to evaluate
algebraic expressions and to automatically differentiate them.

If you want a function that depends not only on collective variables
but also on time you can use the \subpage TIME action.

The optional flag MULTIREP can be used during a multireplica simulation to use as arguments 
quantities computed locally by each partition.
In the current implementation, the number of local arguments chosen with the ARG keyword must be the same in each partition.
When explicitely defined via the VAR keyword, the names of the corresponding variables must also be the same in each partition.
Note that, as long as the two necessary conditions above are satisfied, one is free to use as arguments different 
quantities in different partitions.
When defining the function in the FUNC string, variables from different partitions
can be specified using their local names, say X, labeled with the rank of the partition, i.e., X0,X1,X2...etc..
The label can go from 0 to N-1, where N is the total number of partitions.

\par Examples

The following input tells plumed to perform a metadynamics
using as a CV the difference between two distances.
\plumedfile
dAB: DISTANCE ATOMS=10,12
dAC: DISTANCE ATOMS=10,15
diff: CUSTOM ARG=dAB,dAC FUNC=y-x PERIODIC=NO
# notice: the previous line could be replaced with the following
# diff: COMBINE ARG=dAB,dAC COEFFICIENTS=-1,1
METAD ARG=diff WIDTH=0.1 HEIGHT=0.5 BIASFACTOR=10 PACE=100
\endplumedfile
(see also \ref DISTANCE, \ref COMBINE, and \ref METAD).
Notice that forces applied to diff will be correctly propagated
to atoms 10, 12, and 15.
Also notice that since CUSTOM is used without the VAR option
the two arguments should be referred to as x and y in the expression FUNC.
For simple functions
such as this one it is possible to use \ref COMBINE.

The following input tells plumed to print the angle between vectors
identified by atoms 1,2 and atoms 2,3
its square (as computed from the x,y,z components) and the distance
again as computed from the square root of the square.
\plumedfile
DISTANCE LABEL=d1 ATOMS=1,2 COMPONENTS
DISTANCE LABEL=d2 ATOMS=2,3 COMPONENTS
CUSTOM ...
  LABEL=theta
  ARG=d1.x,d1.y,d1.z,d2.x,d2.y,d2.z
  VAR=ax,ay,az,bx,by,bz
  FUNC=acos((ax*bx+ay*by+az*bz)/sqrt((ax*ax+ay*ay+az*az)*(bx*bx+by*by+bz*bz))
  PERIODIC=NO
... CUSTOM
PRINT ARG=theta
\endplumedfile
(See also \ref PRINT and \ref DISTANCE).

Notice that this action implements a large number of functions (trigovar_nametric, exp, log, etc).
Among the useful functions, have a look at the step function (that is the Heaviside function).
`step(x)` is defined as 1 when `x` is positive and `0` when x is negative. This allows for
a straightforward implementation of if clauses.

For example, imagine that you want to implement a restraint that only acts when a
distance is larger than 0.5. You can do it with
\plumedfile
d: DISTANCE ATOMS=10,15
m: CUSTOM ARG=d FUNC=0.5*step(0.5-x)+x*step(x-0.5) PERIODIC=NO
# check the function you are applying:
PRINT ARG=d,n FILE=checkme
RESTRAINT ARG=d AT=0.5 KAPPA=10.0
\endplumedfile
(see also \ref DISTANCE, \ref PRINT, and \ref RESTRAINT)

The meaning of the function `0.5*step(0.5-x)+x*step(x-0.5)` is:
- If x<0.5 (step(0.5-x)!=0) use 0.5
- If x>0.5 (step(x-0.5)!=0) use x
Notice that the same could have been obtained using an \ref UPPER_WALLS
However, with CUSTOM you can create way more complex definitions.

\warning If you apply forces on the variable (as in the previous example) you should
make sure that the variable is continuous!
Conversely, if you are just analyzing a trajectory you can safely use
discontinuous variables.

A possible continuity check with gnuplot is
\verbatim
# this allow to step function to be used in gnuplot:
gnuplot> step(x)=0.5*(erf(x*10000000)+1)
# here you can test your function
gnuplot> p 0.5*step(0.5-x)+x*step(x-0.5)
\endverbatim

Also notice that you can easily make logical operations on the conditions that you
create. The equivalent of the AND operator is the product: `step(1.0-x)*step(x-0.5)` is
only equal to 1 when x is between 0.5 and 1.0. By combining negation and AND you can obtain an OR. That is,
`1-step(1.0-x)*step(x-0.5)` is only equal to 1 when x is outside the 0.5-1.0 interval.

CUSTOM can be used in combination with \ref DISTANCE to implement variants of the
DISTANCE keyword that were present in PLUMED 1.3 and that allowed to compute
the distance of a point from a line defined by two other points, or the progression
along that line.
\plumedfile
# take center of atoms 1 to 10 as reference point 1
p1: CENTER ATOMS=1-10
# take center of atoms 11 to 20 as reference point 2
p2: CENTER ATOMS=11-20
# take center of atoms 21 to 30 as reference point 3
p3: CENTER ATOMS=21-30

# compute distances
d12: DISTANCE ATOMS=p1,p2
d13: DISTANCE ATOMS=p1,p3
d23: DISTANCE ATOMS=p2,p3

# compute progress variable of the projection of point p3
# along the vector joining p1 and p2
# notice that progress is measured from the middle point
onaxis: CUSTOM ARG=d13,d23,d12 FUNC=(0.5*(y^2-x^2)/z) PERIODIC=NO

# compute between point p3 and the vector joining p1 and p2
fromaxis: CUSTOM ARG=d13,d23,d12,onaxis VAR=x,y,z,o FUNC=(0.5*(y^2+x^2)-o^2-0.25*z^2) PERIODIC=NO

PRINT ARG=onaxis,fromaxis

\endplumedfile

Notice that these equations have been used to combine \ref RMSD
from different snapshots of a protein so as to define
progression (S) and distance (Z) variables \cite perez2015atp.

In the following example we describe the use of the MULTIREP flag during a multireplica simulation.
If the following input is used for each partition, PLUMED will compute the position p of atom 1 locally for each replica.
In the CUSTOM action the coordinate p.x is passed as argument and it is assigned the variable name x.
By setting the MULTIREP flag, PLUMED will automatically generate an extended set of variables with names x0,x1,x2...,x(N-1),
corresponding to the variable x evaluated in each of the N replicas. 
This set of variables can be used in the definition of the function.
In this example, the difference between the x position of atom 1 in the first replica (x0) and in the fifth replica (x4) will be computed.
\plumedfile
p: POSITION ATOM=1
CUSTOM ...
  LABEL=dx
  ARG=p.x
  VAR=x
  FUNC=x0-x4
  MULTIREP
  PERIODIC=NO
... CUSTOM
PRINT ARG=dX
\endplumedfile

*/
//+ENDPLUMEDOC

class Custom :
  public Function
{
  lepton::CompiledExpression expression;
  std::vector<lepton::CompiledExpression> expression_deriv;
  vector<string> var;
  string func;
  bool multirep;//new multirep flag
public:
  explicit Custom(const ActionOptions&);
  void calculate() override;
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(Custom,"CUSTOM")

//+PLUMEDOC FUNCTION MATHEVAL
/*
An alias to the \ref CUSTOM function.

This alias is kept in order to maintain compatibility with previous PLUMED versions.
However, notice that as of PLUMED 2.5 the libmatheval library is not linked anymore,
and the \ref MATHEVAL function is implemented using the Lepton library.

\par Examples

Just replace \ref CUSTOM with \ref MATHEVAL.

\plumedfile
d: DISTANCE ATOMS=10,15
m: MATHEVAL ARG=d FUNC=0.5*step(0.5-x)+x*step(x-0.5) PERIODIC=NO
# check the function you are applying:
PRINT ARG=d,n FILE=checkme
RESTRAINT ARG=d AT=0.5 KAPPA=10.0
\endplumedfile
(see also \ref DISTANCE, \ref PRINT, and \ref RESTRAINT)

*/
//+ENDPLUMEDOC

class Matheval :
  public Custom {
};

PLUMED_REGISTER_ACTION(Custom,"MATHEVAL")

void Custom::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);
  keys.use("ARG"); keys.use("PERIODIC");
  keys.add("compulsory","FUNC","the function you wish to evaluate");
  keys.add("optional","VAR","the names to give each of the arguments in the function.  If you have up to three arguments in your function you can use x, y and z to refer to them.  Otherwise you must use this flag to give your variables names.");
  keys.addFlag("MULTIREP",false,"If set, it allows the definition of a function whose arguments are quantities evaluated locally in different replicas.");
}

Custom::Custom(const ActionOptions&ao):
  Action(ao),
  Function(ao),
  expression_deriv(getNumberOfArguments())
{
  //
  multirep=false;
  parseFlag("MULTIREP",multirep);
  //
  parseVector("VAR",var);
  if(var.size()==0) {
    var.resize(getNumberOfArguments());
    if(getNumberOfArguments()>3)
      error("More than 3 arguments: you must define the variables names using VAR keyword");
    if(var.size()>0) var[0]="x";
    if(var.size()>1) var[1]="y";
    if(var.size()>2) var[2]="z";
  }
  if(var.size()!=getNumberOfArguments())
    error("Number of arguments must match the number of variables");
  parse("FUNC",func);
  addValueWithDerivatives();
  checkRead();

  unsigned bead_rank;
  unsigned nargs = getNumberOfArguments();
  if (multirep){ //Define the extended set of variables for the multirep case
          unsigned nbeads;
	  if (comm.Get_rank() == 0) // multi_sim_comm works only in thread 0
              {
                 nbeads    = multi_sim_comm.Get_size(); 
                 bead_rank = multi_sim_comm.Get_rank(); 
              }
          if (comm.Get_size()  > 1) // if more than one thread per partition, update all threads
              { 
                 comm.Bcast(nbeads,0);
                 comm.Bcast(bead_rank,0); 
              }
	  vector<string> multirep_var(nargs*nbeads); // Auxiliary string of the extended set of variables
	  for (unsigned i=0; i<nbeads; i++){
		  for (unsigned j=0;j<nargs;j++) {
			  multirep_var.at(i*nargs+j)=var[j]+std::to_string(i); // Attach a label with the rank of the partition.
		  }
	  }
	  var.resize(nargs*nbeads);
	  for (unsigned i=0; i<nargs*nbeads; i++) var.at(i)=multirep_var[i]; // Set the new variables names
  }
  else bead_rank=0;

  log.printf("  with function : %s\n",func.c_str());
  log.printf("  with variables :");
  for(unsigned i=0; i<var.size(); i++) log.printf(" %s",var[i].c_str());
  log.printf("\n");

  lepton::ParsedExpression pe=lepton::Parser::parse(func).optimize(lepton::Constants());
  log<<"  function as parsed by lepton: "<<pe<<"\n";
  expression=pe.createCompiledExpression();
  for(auto &p: expression.getVariables()) {
    if(std::find(var.begin(),var.end(),p)==var.end()) {
      error("variable " + p + " is not defined");
    }
  }
  log<<"  derivatives as computed by lepton:\n";

  for(unsigned i=0; i<nargs; i++) { // derivatives only with respect to the variables of this partition
    lepton::ParsedExpression pe=lepton::Parser::parse(func).differentiate(var[bead_rank*nargs+i]).optimize(lepton::Constants());
    log<< "    "<<pe<<"\n";
    expression_deriv[i]=pe.createCompiledExpression();
  }
}

void Custom::calculate() {

          unsigned nargs = getNumberOfArguments();
	  vector<double> args(nargs);

          if (!multirep) for(unsigned i=0; i<nargs; i++) args[i]=getArgument(i);
	  else {
                  unsigned nbeads, bead_rank;
                  if (comm.Get_rank() == 0) {nbeads=multi_sim_comm.Get_size(); bead_rank=multi_sim_comm.Get_rank();}
                  if (comm.Get_size()  > 1) {comm.Bcast(nbeads,0); comm.Bcast(bead_rank,0);}
		  args.resize(nargs*nbeads);
		  for(unsigned i=0; i<nargs*nbeads; i++) args[i]=0.0;// Clear
                  if (comm.Get_rank() == 0) { // multi_sim_comm works only in thread 0
		    for(unsigned i=0; i<nargs; i++) args[bead_rank*nargs+i]=getArgument(i); // Copy local args
		    multi_sim_comm.Sum(args); // Inter-partition comm
                  }
                  if (comm.Get_size()  > 1) comm.Bcast(args,0); // Broadcast to all threads
	  }

	  //Evaluate and set FUNC
          for(unsigned i=0; i<args.size(); i++) {
                  try {
                          expression.getVariableReference(var[i])=args[i];
                          } catch(const PLMD::lepton::Exception& exc) {
                                  // this is necessary since in some cases lepton things a variable is not present even though it is present
                                  // e.g. func=0*x
                          }
          }
          setValue(expression.evaluate());

          //Evaluate and set the local derivatives
          for(unsigned i=0; i<nargs; i++) {
                  for(unsigned j=0; j<args.size(); j++) {
                          try {
                                  expression_deriv[i].getVariableReference(var[j])=args[j];
                                  } catch(const PLMD::lepton::Exception& exc) {
                                          // this is necessary since in some cases lepton things a variable is not present even though it is present
                                          // e.g. func=0*x
                                  }
                          }
                  setDerivative(i,expression_deriv[i].evaluate());
          }
}

}
}

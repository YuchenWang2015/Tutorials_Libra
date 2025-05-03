import os
import numpy as np
import scipy.sparse as sp
import h5py
import warnings
import argparse

from liblibra_core import *
import util.libutil as comn
from libra_py import units, data_conv
import libra_py.dynamics.tsh.compute as tsh_dynamics

path = os.getcwd()
params = {}

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_save_Hvibs', type=str)
parser.add_argument('--basis_type', type=str)
parser.add_argument('--istep', type=int)
parser.add_argument('--fstep', type=int)
parser.add_argument('--istate', type=int)
parser.add_argument('--dt', type=float)
args = parser.parse_args()

# Read the vibronic Hamiltonian
path_to_save_Hvibs = args.path_to_save_Hvibs
basis_type = args.basis_type
istep = args.istep
fstep = args.fstep

NSTEPS = fstep - istep

#================== Read energies =====================
E = []
for step in range(istep, fstep):
    energy_filename = F"{path_to_save_Hvibs}/Hvib_{basis_type}_{step}_re.npz"
    energy_mat = sp.load_npz(energy_filename)
    # For data conversion we need to turn np.ndarray to np.array so that 
    # we can use data_conv.nparray2CMATRIX
    E.append( np.array( np.diag( energy_mat.todense() ) ) )
E = np.array(E)
NSTATES = E[0].shape[0]

#================== Read time-overlap =====================
St = []
for step in range(istep, fstep):        
    St_filename = F"{path_to_save_Hvibs}/St_{basis_type}_{step}_re.npz"
    St_mat = sp.load_npz(St_filename)
    St.append( np.array( St_mat.todense() ) )
St = np.array(St)

#================ Compute NACs and vibronic Hamiltonians along the trajectory ============    
NAC = []
Hvib = [] 
for c, step in enumerate(range(istep, fstep)):
    nac_filename = F"{path_to_save_Hvibs}/Hvib_{basis_type}_{step}_im.npz"
    nac_mat = sp.load_npz(nac_filename)
    NAC.append( np.array( nac_mat.todense() ) )
    Hvib.append( np.diag(E[c, :])*(1.0+1j*0.0)  - (0.0+1j)*nac_mat[:, :] )

NAC = np.array(NAC)
Hvib = np.array(Hvib)

# The interface function
class tmp:
    pass

def compute_model(q, params1, full_id):
    timestep = params1["timestep"]
    nst = params1["nstates"]
    obj = tmp()

    obj.ham_adi = data_conv.nparray2CMATRIX( np.diag(E[timestep, : ]) )
    obj.nac_adi = data_conv.nparray2CMATRIX( NAC[timestep, :, :] )
    obj.hvib_adi = data_conv.nparray2CMATRIX( Hvib[timestep, :, :] )
    obj.basis_transform = CMATRIX(nst,nst); obj.basis_transform.identity()  #basis_transform
    obj.time_overlap_adi = data_conv.nparray2CMATRIX( St[timestep, :, :] )
    
    return obj

# Define the parameters
#================== Model parameters ====================
model_params = { "timestep":0, "icond":0,  "model0":0, "nstates":NSTATES }
# Here, the icond needn't to be adjusted, since istep above already reflects it.

#=============== Some automatic variables, related to the settings above ===================

dyn_general = { "nsteps":NSTEPS, "ntraj":1, "nstates":NSTATES, "dt":args.dt, "nfiles": NSTEPS,
                "progress_frequency":1, "which_adi_states":range(NSTATES), "which_dia_states":range(NSTATES),
                "mem_output_level":2,
                "properties_to_save":["timestep", "time", "se_pop_adi"],
                "prefix":F"NBRA", "isNBRA":0
              }

#=========== Set the coherent Ehrenfest propagation for the patch dynamics ===========

dyn_general.update({"ham_update_method":2})  # read adiabatic properties from mthe files
dyn_general.update({"ham_transform_method":0})  # don't attempt to compute adiabatic properties from the diabatic ones, not to
                                                # override the read ones 
dyn_general.update({"time_overlap_method":0})  # don't attempt to compute those, not to override the read ones
dyn_general.update({"nac_update_method":0})    # don't attempt to recompute NACs, so that we don't override the read values
dyn_general.update({"hvib_update_method":0})   # don't attempt to recompute Hvib, so that we don't override the read values
dyn_general.update({"force_method":0, "rep_force":1}) # NBRA = don't compute forces, so rep_force actually doesn't matter
dyn_general.update({"hop_acceptance_algo":0, "momenta_rescaling_algo":0 })  # no hop, no velocity rescaling
dyn_general.update({"rep_tdse":1}) # the TDSE integration is conducted in adiabatic rep
dyn_general.update({"electronic_integrator":2})  # using the local diabatization approach to integrate TD-SE
dyn_general.update({"tsh_method":-1 }) # no hop
dyn_general.update({"decoherence_algo":-1}) # no decoherence
dyn_general.update({"decoherence_times_type":-1 }) # no decoherence times, infinite decoherence times
dyn_general.update({"do_ssy":0 }) # do no use Shenvi-Subotnik-Yang phase correction
dyn_general.update({"dephasing_informed":0}) # no dephasing-informed correction

#=================== Dynamics =======================
# Nuclear DOF - these parameters don't matter much in the NBRA calculations
nucl_params = {"ndof":1, "init_type":3, "q":[-10.0], "p":[0.0], "mass":[2000.0], "force_constant":[0.0], "verbosity":-1 }

# Amplitudes are sampled
elec_params = {"ndia":NSTATES, "nadi":NSTATES, "verbosity":-1, "init_dm_type":0}

elec_params.update( {"init_type":1,  "rep":1,  "istate":args.istate } )  # how to initialize: random phase, adiabatic representation, given initial state

rnd = Random()

dyn_general.update({"prefix":"out"})
res = tsh_dynamics.generic_recipe(dyn_general, compute_model, model_params, elec_params, nucl_params, rnd)


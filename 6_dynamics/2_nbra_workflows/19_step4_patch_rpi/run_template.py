import os
import numpy as np
import scipy.sparse as sp
import h5py
import warnings
import pickle
import argparse

from liblibra_core import *
import util.libutil as comn
from libra_py import units, data_conv
import libra_py.dynamics.tsh.compute as tsh_dynamics

parser = argparse.ArgumentParser()
parser.add_argument('--nsteps', type=int)
parser.add_argument('--icond', type=int)
parser.add_argument('--istate', type=int)
parser.add_argument('--dt', type=float)
args = parser.parse_args()

# Read the vibronic Hamiltonian
nsteps, icond, istate, dt = args.nsteps, args.icond, args.istate, args.dt

# load parameters
with open('../dyn_params.pkl', 'rb') as f:
    dyn_params = pickle.load(f)

with open('../model_params.pkl', 'rb') as f:
    model_params = pickle.load(f)

with open('../elec_params.pkl', 'rb') as f:
    elec_params = pickle.load(f)

with open('../nucl_params.pkl', 'rb') as f:
    nucl_params = pickle.load(f)

# Adjust patch dynamics
dyn_params.update({"nsteps": nsteps, "dt": dt, "icond":icond}) # icond gives where to read the precomputed file

nstates = model_params["nstates"]
istates = [0.0]*nstates; istates[istate] = 1.0
elec_params.update({"istate":istate, "istates": istates})

# Define the interface function
class tmp:
    pass

def compute_model(q, params, full_id):
    """
    This function serves as an interface function for a serial patch dynamics calculation.
    """

    timestep = params["timestep"]
    nst = params["nstates"]
    E = params["E"]
    NAC = params["NAC"]
    Hvib = params["Hvib"]
    St = params["St"]

    obj = tmp()

    obj.ham_adi = data_conv.nparray2CMATRIX( np.diag(E[timestep, : ]) )
    obj.nac_adi = data_conv.nparray2CMATRIX( NAC[timestep, :, :] )
    obj.hvib_adi = data_conv.nparray2CMATRIX( Hvib[timestep, :, :] )
    obj.basis_transform = CMATRIX(nst,nst); obj.basis_transform.identity()  #basis_transform
    obj.time_overlap_adi = data_conv.nparray2CMATRIX( St[timestep, :, :] )

    return obj

rnd = Random()

res = tsh_dynamics.generic_recipe(dyn_params, compute_model, model_params, elec_params, nucl_params, rnd)


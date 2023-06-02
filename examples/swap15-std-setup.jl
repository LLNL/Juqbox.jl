### Set up a test problem using one of the standard Hamiltonian models
using Juqbox
using Printf
using Plots

# Four qubit parameter setup
# include("four_sys_noguard.jl") # Jaynes-Cummings
include("five_sys_ibm.jl") # IBM Guadelope system

# assign the target gate
target_gate = get_swap_1d_gate(5)

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, init_amp_frac=init_amp_frac, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-3 # better than 99.9% fidelity

println("Setup complete")
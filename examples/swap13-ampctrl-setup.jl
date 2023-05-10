### Set up a test problem using one of the standard Hamiltonian models
using Juqbox
using Printf

# Three qubits, each with 2 essential + 0 guard levels
include("three_sys_noguard.jl") # Jaynes-Cummings

# reduce initial amplitude
#init_amp_frac = 0.5/5

cw_amp_thres = 6e-2 # Get nearest neighbor frequencies
#cw_amp_thres = 1.0 # Only get the carrier frequencies within each sub-system

# duration
T=300.0

# assign the target gate
target_gate = get_swap_1d_gate(3) # get_ident_kron_swap23() # get_swap_1d_gate(nSys) # get_Hd_gate(8) # get_CpNOT(2) #  get_swap_13_1()

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, init_amp_frac=init_amp_frac, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, splines_real_imag=false, randomize_init_ctrl=true)
# cw_prox_thres=5e-3, , cw_amp_thres=6e-2
params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-3 # better than 99.9% fidelity
params.tik0 = 0.0

println("Setup complete")

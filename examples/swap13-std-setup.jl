### Set up a test problem using one of the standard Hamiltonian models
using Juqbox
using Printf
using Plots

# Three qubits parameters settings
include("three_sys_noguard.jl") # Jaynes-Cummings coupling
#include("three_sys_1guard.jl") # 1 guard level per system

# assign the target gate
target_gate = get_swap_1d_gate(3)

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, init_amp_frac=init_amp_frac, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, wmatScale=wmatScale, use_carrier_waves=use_carrier_waves)
# cw_prox_thres=5e-3, , cw_amp_thres=6e-2
params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-4 # better than 99.99% fidelity

println("Setup complete")
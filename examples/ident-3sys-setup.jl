### Set up a test problem using one of the standard Hamiltonian models
using Juqbox
using LinearAlgebra

## Three qubits, each with 2 essential + 0 guard levels
include("three_sys_noguard.jl") # Jaynes-Cummings

# reduce initial amplitude
init_amp_frac = 1e-2/3.14
#randomize_init_ctrl = true

cw_amp_thres = 1.0 # Only get the carrier frequencies within each sub-system

# duration
T=400.0

# assign the target gate
Nsize = prod(Ne)
target_gate = Matrix{ComplexF64}(I,Nsize,Nsize)

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, init_amp_frac=init_amp_frac, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres)
# cw_prox_thres=5e-3, , cw_amp_thres=6e-2
params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-7 # really tight tolerance
params.tik0 = 0.0

println("Setup complete")

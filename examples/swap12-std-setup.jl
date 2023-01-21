### Set up a test problem using one of the standard Hamiltonian models
using Juqbox

## Two qubits

include("two_sys_jc.jl")
retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, rand_amp=rand_amp, Pmin=Pmin)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

# try calling the timestep routine
# nsteps = calculate_timestep(params, maxCoupled=maxAmp)
# println("nsteps = ", nsteps)
### Set up a test problem using one of the standard Hamiltonian models
using Juqbox
using Printf
using Plots

include("two_sys_noguard.jl")

# assign the target gate
target_gate0 = get_swap_1d_gate(2)

# rotate target so that it will agree with the final unitary 
theta = pi/4
target_gate = exp(-im*theta)*target_gate0

maxIter = 50
fidType = 2 # fidType = 1 for Frobenius norm^2, or fidType = 2 for Infidelity, or fidType = 3 for infid^2

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, fidType=fidType)
#true)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-5 # better than 99.9% fidelity

println("Setup complete")

println("Calling run_optimizer for derivative check")
pcof = run_optimizer(params, pcof0, maxAmp, maxIter=maxIter)
pl = plot_results(params, pcof)

println("IPOpt iteration completed")

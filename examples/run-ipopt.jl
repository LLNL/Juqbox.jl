### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots

include("two_sys_noguard.jl")

# assign the target gate, sqrt(Swap12)
Vtg = get_swap_1d_gate(2)
target_gate = sqrt(Vtg)

nTimeIntervals = 3 # 3 # 3 # 2 # 1

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, init_amp_frac=init_amp_frac, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-3 # better than 99.9% fidelity

Ntot = params.Ntot

# Test non-zero Lagrange multipliers
if params.nTimeIntervals > 1
    for q = 1:params.nTimeIntervals-1
        params.Lmult_r[q] = zeros(Ntot, Ntot) # rand(Ntot, Ntot) # zeros(Ntot, Ntot) # 
        params.Lmult_i[q] = zeros(Ntot, Ntot) # rand(Ntot, Ntot)
    end
end

println("Setup completed\n")

params.objThreshold = -9999.9
params.traceInfidelityThreshold = -9999.9

println("Calling run_optimizer for derivative check")
pcof = run_optimizer(params, pcof0, maxAmp, maxIter=100, derivative_test=true)

println("IPOpt completed")
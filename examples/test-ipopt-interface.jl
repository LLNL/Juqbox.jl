### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots

include("two_sys_noguard.jl")

# assign the target gate, sqrt(Swap12)
Vtg = get_swap_1d_gate(2)
target_gate = Vtg # sqrt(Vtg)

nTimeIntervals = 3 # 3 # 2 # 1

fidType = 3 # 2 # fidType = 1 for Frobenius norm^2, or fidType = 2 for Infidelity
useUniConstraints = false # set to true for fidType = 2
    

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC, fidType=fidType, useUniCons=useUniConstraints)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-3 # better than 99.9% fidelity

Ntot = params.Ntot

# Test non-zero Lagrange multipliers
if params.nTimeIntervals > 1
    for q = 1:params.nTimeIntervals-1
        params.Lmult_r[q] = rand(Ntot, Ntot) # zeros(Ntot, Ntot) # 
        params.Lmult_i[q] = rand(Ntot, Ntot)
    end
else
    # Only for testing the Lagrange multiplier term
    params.Lmult_r = Vector{Matrix{Float64}}(undef, 1)
    params.Lmult_i = Vector{Matrix{Float64}}(undef, 1)
    params.Lmult_r[1] = rand(Ntot, Ntot)
    params.Lmult_i[1] = rand(Ntot, Ntot)
end

println("Setup completed\n")

println("Calling run_optimizer for derivative check")
pcof = run_optimizer(params, pcof0, maxAmp, maxIter=0, derivative_test=true)

println("IPOpt test completed")
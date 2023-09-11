### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots

include("two_sys_noguard.jl")

# assign the target gate, sqrt(Swap12)
Vtg = get_swap_1d_gate(2)
target_gate = Vtg # sqrt(Vtg)
fidType = 2 # fidType = 1 for Frobenius norm^2, or fidType = 2 for Infidelity
gammaJump = 0.01 # initial value
useUniConstraints = true # set to true for fidType = 2
maxIter= 150 #100 # 200
nOuter = 3
use_multipliers = true
gammaFactor = 4.0

nTimeIntervals = 3 # 3 # 3 # 2 # 1

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, init_amp_frac=init_amp_frac, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC, gammaJump=gammaJump, fidType=fidType, useUniCons=useUniConstraints)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = -1.0 # 1e-3 # better than 99.9% fidelity
params.objThreshold = -1.0
Ntot = params.Ntot

# Test non-zero Lagrange multipliers
if params.nTimeIntervals > 1
    for q = 1:params.nTimeIntervals-1
        params.Lmult_r[q] = zeros(Ntot, Ntot) # rand(Ntot, Ntot) # zeros(Ntot, Ntot) # 
        params.Lmult_i[q] = zeros(Ntot, Ntot) # rand(Ntot, Ntot)
    end
end

println("Setup completed\n")

derivative_test = true

for outerIt in 1:nOuter
    global pcof0, derivative_test, use_multipliers
    println()
    println("Outer iteration # ", outerIt, " gammaJump = ", params.gammaJump, " Calling run_optimizer")
    global pcof = run_optimizer(params, pcof0, maxAmp, maxIter=maxIter, derivative_test=derivative_test)
    global pl = plot_results(params,pcof)
    println("IPOpt completed")

    if use_multipliers
        println("Updating Lagrange multipliers")
        update_multipliers(pcof, params)

        # if params.nTimeIntervals > 1
        #     println("New Lagrange multipliers:")
        #     for q = 1:params.nTimeIntervals-1
        #         println(params.Lmult_r[:])
        #         println(params.Lmult_i[:])
        #     end
        # end
    end
    # for next outer iteration, increase gammaJump
    params.gammaJump *= gammaFactor

    # run ipopt with updated gammaJump coefficient (and Lagrange multiplier)
    pcof0 = pcof
    derivative_test = false
end

println("Outer iteration completed")
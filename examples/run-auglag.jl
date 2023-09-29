### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots

include("two_sys_noguard.jl")
#include("three_sys_noguard.jl")

# assign the target gate, sqrt(Swap12)
Vtg = get_swap_1d_gate(length(Ne))
target_gate = Vtg # sqrt(Vtg)
fidType = 3 # fidType=1 for Frobenius norm^2, fidType=2 for Infidelity, or fidType=3 for infidelity-squared

constraintType = 2 # 0: No constraints, 1: unitary constraints on initial conditions, 2: zero norm^2(jump) to make the state continuous across time intervals. Set to 1 for fidType = 2
maxIter= 5 # 200 #100 # 200
nOuter = 1 # Only the augmented Lagrangian method uses outer iters
use_multipliers = false # true # Lagrange multipliers
gammaJump = 5e-3 # initial value
gammaMax = 100.0
gammaFactor = 1.5 # 2.0
derivative_test = false # true

nTimeIntervals = 3 # 6 # 4 # 3 # 3 # 2 # 1

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC, gammaJump=gammaJump, fidType=fidType, constraintType=constraintType)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 0.0 # 1e-3 # better than 99.9% fidelity
params.objThreshold = -1.0e-1
Ntot = params.Ntot
params.tik0 = 1.0e-2 # 1.0 # Adjust Tikhonov coefficient

# Test non-zero Lagrange multipliers
if params.nTimeIntervals > 1
    for q = 1:params.nTimeIntervals-1
        params.Lmult_r[q] = zeros(Ntot, Ntot) # rand(Ntot, Ntot) # zeros(Ntot, Ntot) # 
        params.Lmult_i[q] = zeros(Ntot, Ntot) # rand(Ntot, Ntot)
    end
end

println("Setup completed\n")

for outerIt in 1:nOuter
    global pcof0, derivative_test, use_multipliers
    println()
    println("Outer iteration # ", outerIt, " gammaJump = ", params.gammaJump, " Calling run_optimizer")
    global pcof = run_optimizer(params, pcof0, maxAmp, maxIter=maxIter, derivative_test=derivative_test)
    global pl = plot_results(params,pcof)
    println("IPOpt completed")

    if outerIt < nOuter
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
        params.gammaJump = min(gammaMax, params.gammaJump*gammaFactor)
    end
    # run ipopt with updated gammaJump coefficient (and Lagrange multiplier)
    pcof0 = pcof
    derivative_test = false
end

println("Outer iteration completed")
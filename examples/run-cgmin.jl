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

# fidType      Objective
#     1        Frobenius norm^2, 
#     2        Infidelity
#     3        Infidelity-squared
#     4        Generalized (convex) infidelity
fidType = 4 

constraintType = 0 # 0: No constraints, 1: unitary constraints on initial conditions, 2: zero norm^2(jump) to make the state continuous across time intervals. Set to 1 for fidType = 2
maxIter= 100 # 100 # 200 #100 # 200
nOuter = 5 # 20 # Only the augmented Lagrangian method uses outer iters
use_multipliers = true # Lagrange multipliers
gammaJump =  0.1 # 5e-3 # 0.1 # initial value
gammaMax = 100.0
gammaFactor = 1.5 # 2.0
derivative_test = true # false # true

nTimeIntervals = 5 # 3 # 6 # 4 # 3 # 3 # 2 # 1

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC, gammaJump=gammaJump, fidType=fidType, constraintType=constraintType)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 0.0 # NOTE: Only measure the infidelity in the last interval
params.objThreshold = -1.0e10 # total objective may go negative with the Augmented-Lagrange method
rollout_infid_threshold = 1e-5
cgtol = 1.0e-5

params.tik0 = 0.0 # 1.0e-2 # 1.0 # Adjust Tikhonov coefficient

params.quiet = false # true # run ipopt in quiet mode

# Optionall use non-zero Lagrange multipliers
if params.nTimeIntervals > 1
    Ntot = params.Ntot
    for q = 1:params.nTimeIntervals-1
        params.Lmult_r[q] = zeros(Ntot, Ntot) # rand(Ntot, Ntot)
        params.Lmult_i[q] = zeros(Ntot, Ntot) # rand(Ntot, Ntot)
    end
end

println("Setup completed\n")

for outerIt in 1:nOuter
    global pcof0, derivative_test, use_multipliers, ipopt_verbose
    println()
    println("Outer iteration # ", outerIt, " gammaJump = ", params.gammaJump, " Calling cgmin optimizer...")
    cg_res = cgmin(lagrange_obj, lagrange_grad, pcof0, params, cgtol=cgtol, maxIter=maxIter)

    global pcof = cg_res[1]
    
    # evaluate fidelity and unitaryhistory
    alpha = pcof[1:params.nAlpha] # extract the B-spline coefficients
    objv, rollout_infid, leakage = Juqbox.traceobjgrad(alpha, params, false, false);

    println()
    println("Rollout infidelity: ", rollout_infid, " max(||Jump||): ", sqrt(maximum(params.nrm2_Cjump)) ) #, " final dual_inf: ", params.dualInfidelityHist[end])

    if rollout_infid < rollout_infid_threshold
        println("Terminating outer iteration with rollout_infid = ", rollout_infid)
        break
    end

    if outerIt < nOuter
        if use_multipliers
            # println("Updating Lagrange multipliers")
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

# pl = plot_results(params,pcof)

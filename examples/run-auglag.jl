### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots
using FileIO
using DelimitedFiles

include("two_sys_noguard.jl")
#include("three_sys_noguard.jl")

# assign the target gate
Vtg = get_swap_1d_gate(length(Ne))
target_gate = Vtg # sqrt(Vtg)

# fidType      Objective
#     1        Frobenius norm^2, 
#     2        Infidelity
#     3        Infidelity-squared
#     4        Generalized (convex) infidelity
fidType = 4 

constraintType = 0 # 0: No constraints, 1: unitary constraints on initial conditions, 2: zero norm^2(jump) to make the state continuous across time intervals. Set to 1 for fidType = 2

initctrl_MHz = 1.0

nTimeIntervals = 4 # 3 # 6 # 4 # 3 # 3 # 2 # 1

maxIter= 80 # 100 # 200 #100 # 200
nOuter = 3 # 20 # Only the augmented Lagrangian method uses outer iters
use_multipliers = true # false # Lagrange multipliers
gammaJump = 0.25 # 1/length(Ne) # 5e-3 # 0.1 # initial value
gammaMax = 100.0
gammaFactor = 1.0 # no change # 1.5 # 2.0
derivative_test = false # true # false # true

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC, gammaJump=gammaJump, fidType=fidType, constraintType=constraintType)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-4 # NOTE: Only measure the infidelity in the last interval
params.objThreshold = -1.0e10 # total objective may go negative with the Augmented-Lagrange method
params.discontThreshold = 1e-4
rollout_infid_threshold = 1e-4

params.tik0 = 1.0e-2 # 1.0 # Adjust Tikhonov coefficient

#params.project_ic = false # Enable projection of intermediate initial conditions?

params.quiet = false # true # run ipopt in quiet mode
if params.quiet
    ipopt_verbose = 0
else
    ipopt_verbose = 5 # default value
end

# Test non-zero Lagrange multipliers, or start from zero multipliers
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
    println("Outer iteration # ", outerIt, " gammaJump = ", params.gammaJump, " Calling run_optimizer...")
    global pcof = run_optimizer(params, pcof0, maxAmp, maxIter=maxIter, derivative_test=derivative_test, print_level = ipopt_verbose)
    
    # evaluate fidelity and unitaryhistory
    alpha = pcof[1:params.nAlpha] # extract the B-spline coefficients
    objv, rollout_infid, leakage = Juqbox.traceobjgrad(alpha, params, false, false);

    println()
    println("Rollout infidelity: ", rollout_infid, " max(||Jump||^2): ", maximum(params.nrm2_Cjump), " final dual_inf: ", params.dualInfidelityHist[end])

    if rollout_infid < rollout_infid_threshold
        println("Terminating outer iteration with rollout_infid = ", rollout_infid)
        break
    end

    #if outerIt <= nOuter # always update lambdas
        if use_multipliers
            println("Updating Lagrange multipliers")
            update_multipliers(pcof, params)

            # save multipliers
            fname = "lag-mult-r-" * string(outerIt) * ".dat";
            writedlm(fname, params.Lmult_r)
            fname = "lag-mult-i-" * string(outerIt) * ".dat";
            writedlm(fname, params.Lmult_i)
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
    #end
    # run ipopt with updated gammaJump coefficient (and Lagrange multiplier)
    pcof0 = pcof
    derivative_test = false
end

println("Outer iteration completed")

# pl = plot_results(params,pcof)

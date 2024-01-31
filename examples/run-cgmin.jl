### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots

#include("two_sys_noguard.jl")
include("three_sys_noguard.jl")

# assign the target gate, sqrt(Swap12)
Vtg = get_swap_1d_gate(length(Ne))
target_gate = Vtg # sqrt(Vtg)

# fidType      Objective
#     1        Frobenius norm^2, 
#     2        Infidelity
#     3        Infidelity-squared
#     4        Generalized (convex) infidelity
fidType = 4 
# 0: No constraints, 1: unitary constraints on initial conditions, 2: zero norm^2(jump) to make the state continuous across time intervals. Set to 1 for fidType = 2
constraintType = 0 

gamma0 = 10.0 # tuneable parameter

nTimeIntervals = 10 # 5 # 3 # 3 # 6 # 4 # 3 # 3 # 2 # 1
Hdim = 2^(length(Ne))

gammaJump = gamma0/Hdim/nTimeIntervals # 0.1 # 5e-3 # 0.1 works well for 3-qubit case # initial value

initctrl_MHz = 1.0

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC, gammaJump=gammaJump, fidType=fidType, constraintType=constraintType)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

# set to false for ipopt
use_cgmin = false # true

nOuter = 1 # 15 # 20 # Only the augmented Lagrangian method uses outer iters
use_multipliers = true # Lagrange multipliers
gammaMax = 100.0

derivative_test = false # true

params.traceInfidelityThreshold = 0.0 # NOTE: Only measures the infidelity in the last interval
params.objThreshold = -1.0e10 # total objective may go negative with the Augmented-Lagrange method

rollout_infid_threshold = 1e-4 # 1e-5 # overall convergence criteria

maxIter= 1000 # Max number of inner iterations
rel_grad_threshold = 2e-4 # inner loop relative convergence criteria on grad (squared for cgmin)
cgtol = 1.0e-8 # 1.0e-5 # inner loop absolute convergence criteria on grad squared (cgmin)
println("Inner loop params: maxIter = ", maxIter, " abs grad conv = ", cgtol, " rel grad conv = ", rel_grad_threshold)

cons_threshold = 1e-4 # threshold for constraint violation
gammaFactor = 2.0 # 1.5 # 1.2 # Increase gammaJump by this factor if the constraints are too violated
println("AL parameters: constraint viloation threshold = ", cons_threshold, " Gamma0 = ", gammaJump, " Gamma inc fact = ", gammaFactor)

params.tik0 = 0.0 # 0.1 works well for 3-qubit case # 1.0e-2 # 1.0 # Adjust Tikhonov coefficient

params.quiet = false # true # run ipopt in quiet mode
if params.quiet
    ipopt_verbose = 0
else
    ipopt_verbose = 4 # default value is 5
end

# Optionally use non-zero Lagrange multipliers
if params.nTimeIntervals > 1
    Ntot = params.Ntot
    for q = 1:params.nTimeIntervals-1
        params.Lmult_r[q] = zeros(Ntot, Ntot) # rand(Ntot, Ntot)
        params.Lmult_i[q] = zeros(Ntot, Ntot) # rand(Ntot, Ntot)
    end
end

println("Setup completed\n")
println()
if use_cgmin
    println("Using CGmin as inner loop optimizer")
else
    println("Using IPOpt as inner loop optimizer")
end


for outerIt in 1:nOuter
    global pcof0, derivative_test, use_multipliers, ipopt_verbose
    println()
    println("Outer iteration # ", outerIt, " gammaJump = ", params.gammaJump, " Calling inner loop optimizer...")
    if use_cgmin
        cg_res = cgmin(lagrange_obj, lagrange_grad, pcof0, params, cgtol=cgtol, maxIter=maxIter, rel_grad_threshold=rel_grad_threshold)
        global pcof = cg_res[1]
    else # use IPOpt
        global pcof = run_optimizer(params, pcof0, maxAmp, maxIter=maxIter, derivative_test=derivative_test, print_level = ipopt_verbose, print_frequency_iter=100, ipTol=rel_grad_threshold)
    end

    # evaluate fidelity and unitaryhistory
    alpha = pcof[1:params.nAlpha] # extract the B-spline coefficients
    objv, rollout_infid, leakage = Juqbox.traceobjgrad(alpha, params, false, false);

    if params.nTimeIntervals > 1
        maxJump = maximum(params.nrm2_Cjump)
    else
        maxJump = 0.0
    end
    println("Rollout infidelity: ", rollout_infid, " max(||Jump||^2): ", maxJump, " final ||grad||: ", params.dualInfidelityHist[end])

    if rollout_infid < rollout_infid_threshold
        println("Terminating outer iteration with rollout_infid = ", rollout_infid)
        break
    end

    if outerIt < nOuter
        if use_multipliers 
            if maximum(params.nrm2_Cjump) < cons_threshold
                println("Updating Lagrange multipliers because constraint violation < threshold = ", cons_threshold," keeping penalty coeff at mu = ", params.gammaJump)
                update_multipliers(pcof, params)
            else
                if maximum(params.nrm2_Cjump)/cons_threshold > 10.0
                    fact = gammaFactor
                else
                    fact = 1.0 # keep the penalty factor unchanged
                end
                params.gammaJump = min(gammaMax, params.gammaJump*fact)
                println("NOT updating Lagrange multipliers because constraint violation: max(norm2(Cjump)) = ", params.constraintViolationHist[end], " >= ", cons_threshold, " new gamma = ", params.gammaJump)
            end

        end
        # for next outer iteration, increase gammaJump
        
    end
    # run ipopt with updated gammaJump coefficient (and Lagrange multiplier)
    pcof0 = pcof
    derivative_test = false
end

println("Outer iteration completed. Total # grad evals: ", length(params.constraintViolationHist))

# pl = plot_results(params,pcof)

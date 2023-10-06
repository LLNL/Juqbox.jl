### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots

include("two_sys_noguard.jl")
maxIter= 500 # 1000 # 230 # 200 #100 # 200
theta = pi/4

#include("three_sys_noguard.jl")
#maxIter= 1000 
# theta = 0.0

# assign the target gate, sqrt(Swap12)
Vtg = get_swap_1d_gate(length(Ne))

# rotate target so that it will agree with the final unitary 
target_gate = exp(-im*theta)*Vtg
#target_gate = Vtg # sqrt(Vtg)

fidType = 1 # fidType=1 for Frobenius norm^2, fidType=2 for Infidelity, or fidType=3 for infidelity-squared

gammaJump = 0.4 # coefficient for the norm^2(Jump) penalty term

constraintType = 2 # 0: No constraints, 1: unitary constraints on initial conditions, 2: zero norm^2(jump) to make the state continuous across time intervals. Set to 1 for fidType = 2

derivative_test = false # true

nTimeIntervals = 6 # 3 # 25 # 3 # 4 # 3 # 3 # 2 # 1

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC, fidType=fidType, gammaJump=gammaJump, constraintType=constraintType)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 0.0 # 1e-3 # better than 99.9% fidelity
params.objThreshold = -1.0
Ntot = params.Ntot
params.tik0 = 1.0e-3 # 1.0 # Adjust Tikhonov coefficient
lbfgsMax = 20

println("Setup completed\n")

pcof = run_optimizer(params, pcof0, maxAmp, maxIter=maxIter, lbfgsMax=lbfgsMax, derivative_test=derivative_test)
pl = plot_results(params, pcof)
println("IPOpt completed")

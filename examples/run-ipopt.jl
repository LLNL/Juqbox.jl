### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots

# 2 qubits
#include("two_sys_noguard.jl")
#theta = pi/4

# 3 qubits
include("three_sys_noguard.jl")
#theta = 0.0

# 4 qubits
#include("four_sys_noguard.jl")

# assign the target gate, sqrt(Swap12)
Vtg = get_swap_1d_gate(length(Ne))

# rotate target so that it will agree with the final unitary 
#target_gate = exp(-im*theta)*Vtg
target_gate = Vtg # sqrt(Vtg)

fidType = 4 # 4 # 1 # fidType=1 for Frobenius norm^2, fidType=2 for Infidelity, or fidType=3 for infidelity-squared, fidType=4 for generalized infidelity

gammaJump = 0.0 # coefficient for the norm^2(Jump) penalty term in the augmented Lagrange method (constraintType=0)

constraintType = 4 # 1 # 2 # 0: Augmented Lagrange method, no explicit constraints, 1: unitary constraints on initial conditions and norm^2(jump), 2: zero norm^2(jump) to make the state continuous across time intervals, 4: U(T_j) - W_j = 0 (pointwise).
# Set to 1 when using fidType = 2

nTimeIntervals = 5 # 100 # 10 # 5 # 1 # 4 #  3 # 25 # 3 # 4 # 3 # 3 # 2 # 1

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC, fidType=fidType, gammaJump=gammaJump, constraintType=constraintType)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-12 # 1e-3 # better than 99.9% fidelity
params.objThreshold = 1e-12
Ntot = params.Ntot
params.tik0 = 1.0e-3 # 1.0 # Adjust Tikhonov coefficient
maxIter= 1 # 400 # 200 # 500 # 1000 # 230 # 200 #100 # 200
lbfgsMax = 200
derivative_test = false # true # false # true

println("Setup completed\n")

pcof = run_optimizer(params, pcof0, maxAmp, maxIter=maxIter, lbfgsMax=lbfgsMax, derivative_test=derivative_test)
pl = plot_results(params, pcof)
println("IPOpt completed")

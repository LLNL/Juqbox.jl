### Set up a test problem using one of the standard Hamiltonian models
using Juqbox
using Printf
using Plots

# Three qubits parameters settings
include("three_sys_noguard.jl") # Dipole-dipole coupling
#include("three_sys_1guard.jl") # 1 guard level per system

# assign the target gate
target_gate = get_swap_1d_gate(3)

maxIter = 250
fidType = 4 # fidType = 1 for Frobenius norm^2, or fidType = 2 for Infidelity, or fidType = 3 for infid^2, 4 for generalized (convex) infidelity

constraintType = 0 # 0: No constraints, 1: unitary constraints on initial conditions, 2: zero norm^2(jump) to make the state continuous across time intervals. Set to 1 for fidType = 2

initctrl_MHz = 1.0

nTimeIntervals = 4 # 3 # 6 # 4 # 3 # 3 # 2 # 1

maxIter= 1 # 100 # 200 #100 # 200
nOuter = 1 # 20 # Only the augmented Lagrangian method uses outer iters
use_multipliers = false # Lagrange multipliers
gammaJump = 1.0 # 1/length(Ne) # 5e-3 # 0.1 # initial value

# multi-windows
retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC, gammaJump=gammaJump, fidType=fidType, constraintType=constraintType)

# single window
#retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, wmatScale=wmatScale, use_carrier_waves=use_carrier_waves, fidType=fidType)


params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 0.0 # 1e-4 # better than 99.99% fidelity

println("Setup complete")

# println("Calling run_optimizer for derivative check")
# pcof = run_optimizer(params, pcof0, maxAmp, maxIter=maxIter)
# pl = plot_results(params, pcof)

# println("IPOpt iteration completed")

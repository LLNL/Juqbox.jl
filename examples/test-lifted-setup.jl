### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots

include("two_sys_noguard.jl")

# assign the target gate
Vtg = get_swap_1d_gate(2)
target_gate = sqrt(Vtg)
fidType = 1 # 1: Frobenius norm^2, 2: infidelity

verbose = true

nTimeIntervals = 3 # 3 # 2 # 1

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, init_amp_frac=init_amp_frac, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, fidType=fidType, zeroCtrlBC=zeroCtrlBC, verbose=verbose)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-3 # better than 99.9% fidelity

Ntot = params.Ntot

# Test non-zero Lagrange multipliers
if params.nTimeIntervals > 1
    for q = 1:params.nTimeIntervals-1
        params.Lmult_r[q] = zeros(Ntot, Ntot) # rand(Ntot, Ntot)
        params.Lmult_i[q] = zeros(Ntot, Ntot) # rand(Ntot, Ntot)
    end
else
    # Only for testing the Lagrange multiplier term
    params.Lmult_r = Vector{Matrix{Float64}}(undef, 1)
    params.Lmult_i = Vector{Matrix{Float64}}(undef, 1)
    params.Lmult_r[1] = rand(Ntot, Ntot)
    params.Lmult_i[1] = rand(Ntot, Ntot)
end

# for 1 interval, try kpar=3, 9, 15, 18
# for 3 intervals with D1=22 try 3, 9, 15, 18
# for 2 intervals with D1=22 try 5, 15
# for 2 intervals and the grad wrt W, try kpar in [177, 208]
# for 3 intervals, Winit^{(1)} has index [177, 208], for Winit^{(2)} add 32
params.kpar = 3 # 178 + 16 + 32 # 177 # 3 # 178 + 32 +16 + 8# 178, 178 + 16, 178 + 32 # test this component of the gradient

println("Setup completed\n")

total_grad = zeros(params.nCoeff)
println("Calling lagrange_objgrad for total objective and gradient")
obj0, _, _ = lagrange_grad(pcof0, params, total_grad, verbose)

println()
println("FD estimate of the gradient based on objectives for perturbed pcof's\n")

pert = 1e-7
pcof_p = copy(pcof0)
pcof_p[params.kpar] += pert
obj_p, _, _ = lagrange_obj(pcof_p, params, false)

pcof_m = copy(pcof0)
pcof_m[params.kpar] -= pert
obj_m, _, _ = lagrange_obj(pcof_m, params, false)

println("kpar = ", params.kpar, " obj_p = ", obj_p, " obj_m = ", obj_m)
obj_grad_fd = 0.5*(obj_p - obj_m)/pert
println("FD testing of the gradient")
println("kpar = ", params.kpar, " obj_grad_fd = ", obj_grad_fd, " obj_grad_adj = ", total_grad[params.kpar], " fd - adj = ", obj_grad_fd - total_grad[params.kpar])
# println("Setup complete")
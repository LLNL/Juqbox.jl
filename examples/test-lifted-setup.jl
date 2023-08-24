### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots

include("two_sys_noguard.jl")

# assign the target gate
target_gate = get_swap_1d_gate(2)

nTimeIntervals = 1

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, init_amp_frac=init_amp_frac, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-3 # better than 99.9% fidelity

Ntot = params.Ntot

# Test non-zero Lagrange multipliers
if params.nTimeIntervals > 1
    for q = 1:params.nTimeIntervals-1
        params.Lmult_r[q] = rand(Ntot, Ntot)
        params.Lmult_i[q] = rand(Ntot, Ntot)
    end
end

# testing the Lagrange multiplier term
params.Lmult_r = Vector{Matrix{Float64}}(undef, 1)
params.Lmult_i = Vector{Matrix{Float64}}(undef, 1)
params.Lmult_r[1] = rand(Ntot, Ntot)
params.Lmult_i[1] = rand(Ntot, Ntot)

params.kpar = 5 # test this component of the gradient

println("Calling lagrange_objgrad")
obj0 = lagrange_objgrad(pcof0, params, true, true)

# FD estimate of the gradient
pert = 1e-7
pcof_p = copy(pcof0)
pcof_p[params.kpar] += pert
obj_p = lagrange_objgrad(pcof_p, params, true, false)

pcof_m = copy(pcof0)
pcof_m[params.kpar] -= pert
obj_m = lagrange_objgrad(pcof_m, params, true, false)

obj_grad_fd = 0.5*(obj_p - obj_m)/pert
println("kpar = ", params.kpar, " obj_grad_fd = ", obj_grad_fd)
# println("Setup complete")
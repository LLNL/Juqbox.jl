### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots

include("two_sys_noguard.jl")

# assign the target gate
Vtg = get_swap_1d_gate(2)
target_gate = sqrt(Vtg)

nTimeIntervals = 2 # 3 # 2 # 1

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, init_amp_frac=init_amp_frac, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC)

p = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

p.traceInfidelityThreshold = 1e-3 # better than 99.9% fidelity

Ntot = p.Ntot

# for 3 intervals with D1=22 try 3, 9, 15, 18
# for 2 intervals with D1=22 try 5, 15
# for 2 intervals and the grad wrt W, try kpar in [177, 208]
# for 3 intervals, Winit^{(1)} has index [177, 208], for Winit^{(2)} add 32
p.kpar = 178 # 178, 178 + 16, 178 + 32 # test this component of the gradient

println("Setup completed\n")

# allocate storage for the constraints
nCons = (p.nTimeIntervals - 1) * p.Ntot^2
unit_cons = zeros(nCons)
println("# constraints: ", nCons)
println("Calling unitary_constraints to evaluate all constraints")
unitary_constraints(pcof0, unit_cons, p, true)
println("unit_cons: ")
println(unit_cons)

# modify Winit to deviate from unitary
for interval = 1:p.nTimeIntervals-1
    println("Interval # ", interval)
    # get initial condition offset in pcof0 array
    offc = p.nAlpha + (interval-1)*p.nWinit # for interval = 1 the offset should be nAlpha
    nMat = p.Ntot^2
    W_r = reshape(pcof0[offc+1:offc+nMat], p.Ntot, p.Ntot)
    W_r += 0.01*rand(p.Ntot, p.Ntot) # Perturb real part
    pcof0[offc+1:offc+nMat] = vec(W_r)
    offc += nMat
    W_i = reshape(pcof0[offc+1:offc+nMat], p.Ntot, p.Ntot)
    pcof0[offc+1:offc+nMat] = vec(W_i) # Leave the imaginary part
end

println("Calling unitary_constraints to evaluate all constraints for the PERTURBED Winit")
unitary_constraints(pcof0, unit_cons, p, true)
println("perturbed unit_cons: ")
println(unit_cons)

println()
println("FD estimate of the constraints for perturbed pcof's\n")

pert = 1e-7
one_cons = 1
pcof_p = copy(pcof0)
pcof_p[p.kpar] += pert
unitary_constraints(pcof_p, unit_cons, p, false)
obj_p = unit_cons[one_cons]

pcof_m = copy(pcof0)
pcof_m[p.kpar] -= pert
unitary_constraints(pcof_m, unit_cons, p, false)
obj_m = unit_cons[one_cons]

println("kpar = ", p.kpar, " one_cons = ", one_cons, " obj_p = ", obj_p, " obj_m = ", obj_m)
obj_grad_fd = 0.5*(obj_p - obj_m)/pert
println("FD testing of the Jacobian")
println("kpar = ", p.kpar, " cons_jac_fd = ", obj_grad_fd)

println("Constraints test complete")
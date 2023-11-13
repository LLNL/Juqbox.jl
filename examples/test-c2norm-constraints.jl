### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots

include("two_sys_noguard.jl")

# assign the target gate
Vtg = get_swap_1d_gate(2)
target_gate = sqrt(Vtg) #sqrt(swap)

nTimeIntervals = 2 # 2 # 1
constraintType = 2

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC, constraintType=constraintType)

p = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

p.traceInfidelityThreshold = 1e-3 # better than 99.9% fidelity

Ntot = p.Ntot

# for 3 intervals with D1=22 try 3, 9, 15, 18
# for 2 intervals with D1=22 try 5, 15
# for 2 intervals and the grad wrt W, try kpar in [177, 208]
# for 3 intervals, Winit^{(1)} has index [177, 208], for Winit^{(2)} add 32
p.kpar = 172 # 280 # 248 # 198 # 248 # 3 # 234 # 217 # 198 # 38 # 3 # 178 # 178, 178 + 16, 178 + 32 # test this component of the gradient

println("Setup completed\n")

function find_elem(row::Int64, col::Int64, jac_rows::Vector{Int32}, jac_cols::Vector{Int32})
    nEleJac = length(jac_rows)
    for j in 1:nEleJac
        if row == jac_rows[j] && col == jac_cols[j]
            return j
        end
    end
    return -1
end

# allocate storage for the constraints
nCons = (p.nTimeIntervals - 1) # one constraint per intermediate initial condition
c2norm_cons = zeros(nCons)
println("# constraints: ", nCons)
println("Calling c2norm_constraints to evaluate all constraints")
c2norm_constraints(pcof0, c2norm_cons, p, true)
println("c2norm_cons: ")
println(c2norm_cons)

println()
testJac = false

if testJac
    nEleJac = (p.nTimeIntervals - 1) * (p.nAlpha + p.nWinit) # begin with the Jacobian wrt B-spline coeffs, and wrt Winit_next)
    if p.nTimeIntervals > 2
        nEleJac += (p.nTimeIntervals - 2) * p.nWinit 
    end
    println("# non-zero elements in Jac: ", nEleJac)
    jac_ele = zeros(nEleJac)
    jac_rows = zeros(Int32,nEleJac)
    jac_cols = zeros(Int32,nEleJac)
    println("Calling c2norm_jacobian to evaluate the Jacobian of all constraints")
    c2norm_jacobian(pcof0, jac_ele, p, true)
    c2norm_jacobian_idx(jac_rows, jac_cols, p, true)

    println("row, col, jac")
    for j in 1:nEleJac
        println(jac_rows[j], ", ", jac_cols[j], ", ", jac_ele[j])
    end

    println()
    pert = 1e-7
    one_cons = 2 # 1 # 2 # constraint number to be tested
    println("FD estimate of the jacobian of constraint = ", one_cons, ", wrt element kpar (col) = ", p.kpar)

    jac_idx = find_elem(one_cons, p.kpar, jac_rows, jac_cols)

    if jac_idx < 0
        println("Warning: no entry in the jacobian matches row, col = ", one_cons, ", ", p.kpar)
        one_jac_ele = 0.0
    else
        println("(row, col) = ", one_cons, ", ", p.kpar, " has index = ", jac_idx) 
        one_jac_ele = jac_ele[jac_idx]
    end

    pcof_p = copy(pcof0)
    pcof_p[p.kpar] += pert
    c2norm_constraints(pcof_p, c2norm_cons, p, false)
    obj_p = c2norm_cons[one_cons]

    pcof_m = copy(pcof0)
    pcof_m[p.kpar] -= pert
    c2norm_constraints(pcof_m, c2norm_cons, p, false)
    obj_m = c2norm_cons[one_cons]

    obj_grad_fd = 0.5*(obj_p - obj_m)/pert
    println("FD testing of the Jacobian")
    println("row # = ", one_cons, " col # = ", p.kpar, " cons_jac_fd = ", obj_grad_fd, " cons_jac_adj = ", one_jac_ele, " diff = ", obj_grad_fd - one_jac_ele)
end

println("Constraints test complete")
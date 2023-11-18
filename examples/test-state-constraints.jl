### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Juqbox
using Printf
using Plots

include("two_sys_noguard.jl")

# assign the target gate
Vtg = get_swap_1d_gate(2)
target_gate = sqrt(Vtg) #sqrt(swap)

nTimeIntervals = 3 # 3 # 2 # 1

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC, constraintType=2)

p = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

# randomize Winit part of pcof
doRandomize = false
if doRandomize
    for q = 1:p.nTimeIntervals-1
        offc = p.nAlpha + (q-1)*p.nWinit # for interval = 1 the offset should be nAlpha
        pcof0[offc+1:offc+p.nWinit] = rand(p.nWinit)
    end
end

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
nCons = (p.nTimeIntervals - 1)*p.nWinit # 2*N^2 constraint per intermediate initial condition

state_cons = zeros(nCons)
println("# constraints: ", nCons)
println("Calling state_constraints to evaluate all constraints")
state_constraints(pcof0, state_cons, p, true)
println("state_cons: ")
println(state_cons)

println()
testJac = true # true # false

if testJac
    nEleJac = 0
    #nEleJac += (p.nTimeIntervals - 1) * 2 * p.nWinit # test
    for q = 1: p.nTimeIntervals - 1
        global nEleJac += 2 * p.NfreqTot * (p.d1_end[q] - p.d1_start[q] + 1) * p.nWinit # Jacobian wrt alpha (Overestimating)
    end

    #nEleJac += (p.nTimeIntervals - 1) * p.nWinit # Jacobian wrt Winit_next
    if p.nTimeIntervals > 2 # Jacobian wrt initial conditions, Winit
        #nEleJac += (p.nTimeIntervals - 2) * p.nWinit * 2 * p.N
    end
    println("# non-zero elements in Jac: ", nEleJac)
    jac_ele = zeros(nEleJac)
    jac_rows = zeros(Int32,nEleJac)
    jac_cols = zeros(Int32,nEleJac)
    println("Calling state_jacobian to evaluate the Jacobian of the state constraints")
    state_jacobian(pcof0, jac_ele, jac_rows, jac_cols, p, true)
    # state_jacobian_idx(jac_rows, jac_cols, p, true)

    printJac = false
    if printJac
        println("row, col, jac")
        for j in 1:nEleJac
            println(jac_rows[j], ", ", jac_cols[j], ", ", jac_ele[j])
        end
    end

    println()
    pert = 1e-7
    #one_cons = 1 # 1 # 1 # 2 # constraint number to be tested
    one_par = 27 # for cons=36: 217-220, 233-236, 252
    c_rge = (33:64) # (1:32) # 
    println("FD estimate of the jacobian of constraints = ", c_rge, ", wrt element kpar (col) = ", one_par)

    max_diff = 0.0
    for one_cons in c_rge

        jac_idx = find_elem(one_cons, one_par, jac_rows, jac_cols)

        if jac_idx < 0
            println("Warning: no entry in the jacobian matches row, col = ", one_cons, ", ", one_par)
            one_jac_ele = 0.0
        else
            println("(row, col) = ", one_cons, ", ", one_par, " has index = ", jac_idx) 
            one_jac_ele = jac_ele[jac_idx]
        end

        pcof_p = copy(pcof0)
        pcof_p[one_par] += pert
        state_constraints(pcof_p, state_cons, p, false)
        obj_p = state_cons[one_cons]

        pcof_m = copy(pcof0)
        pcof_m[one_par] -= pert
        state_constraints(pcof_m, state_cons, p, false)
        obj_m = state_cons[one_cons]

        obj_grad_fd = 0.5*(obj_p - obj_m)/pert

        diff = obj_grad_fd - one_jac_ele
        global max_diff = max(max_diff, abs(diff))
        println("row # = ", one_cons, " col # = ", one_par, " cons_jac_fd = ", obj_grad_fd, " cons_jac_adj = ", one_jac_ele, " diff = ", diff)
    end
    println()
    println("Max difference = ", max_diff)
end

println("Constraints test complete")
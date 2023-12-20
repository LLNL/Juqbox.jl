### Set up a test problem for the lifted approach with intermediate initial conditions
# and continuity constraints
using Printf
using Plots
using DelimitedFiles
using LinearAlgebra
using MPI
using Juqbox

function find_elem(row::Int64, col::Int64, jac_rows::Vector{Int32}, jac_cols::Vector{Int32})
    nEleJac = length(jac_rows)
    for j in 1:nEleJac
        if row == jac_rows[j] && col == jac_cols[j]
            return j
        end
    end
    return -1
end

nTimeIntervals = 12 # 3 # 2 # 1
debug = true # true # false

#MPI.Init_thread(MPI.THREAD_SINGLE) # Seems important to call MPI.Init from the top level (?)
MPI.Init()
mpiObj = Juqbox.setup_mpi(nTimeIntervals, debug) # Initialize MPI and decompose the time intervals among ranks

# include("two_sys_noguard.jl")
#include("three_sys_noguard.jl")
include("four_sys_noguard.jl")

# assign the target gate
Vtg = get_swap_1d_gate(length(Ne))
target_gate = sqrt(Vtg) #sqrt(swap)

rollOutInitialState=false

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, mpiObj,maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, nTimeIntervals=nTimeIntervals, zeroCtrlBC=zeroCtrlBC, constraintType=2, rollOutInitialState=rollOutInitialState)

p = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

doContinue = true

if doContinue

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

    # allocate storage for the constraints
    nStateVecGlobal = p.nTimeIntervals*p.nWinit # 2*N^2 constraint per time interval

    state_vec_global = zeros(nStateVecGlobal)
    
    if mpiObj.myRank == mpiObj.root
        println("# global elements: ", nStateVecGlobal)
        println("Calling rollOutStateMatrices")
    end

    ############################################################
    rollOutStateMatrices(pcof0, state_vec_global, p, mpiObj, true)
    ############################################################

    if mpiObj.myRank == mpiObj.root
        # println("state_vec_global: ")
        # println(state_vec_global)
        println()

        # save/compare constraints on file
        fname = "state-eq-ref-" * string(p.nTimeIntervals) * ".dat"
        if isfile(fname)
            econ_ref = readdlm(fname) # read reference file
        else
            econ_ref = zeros(0)
        end
        if length(state_vec_global) == length(econ_ref)
            println("Comparing to reference solution: norm(e_con - econ_ref) = ", norm(state_vec_global - econ_ref))
        else
            writedlm(fname, state_vec_global)
            println("Saved reference solution on file: ", fname)
        end

        for q = 1: p.nTimeIntervals-1
            println("Interval = ", q, " time span = (", p.T0int[q], ", ", p.T0int[q+1], "]", " B-spline index range = [", p.d1_start[q], ", ", p.d1_end[q], "]")
        end
        q= p.nTimeIntervals
        println("Final interval: time span = (", p.T0int[q], ", ", p.T, "]", " B-spline index range = [", p.d1_start[q], ", ", p.d1_end[q], "]")
        println()
    end

    testJac = false # true # false

    if testJac
        nEleJac = 0
        #nEleJac += (p.nTimeIntervals - 1) * 2 * p.nWinit # test
        for q = 1: p.nTimeIntervals - 1
            global nEleJac += 2 * p.NfreqTot * (p.d1_end[q] - p.d1_start[q] + 1) * p.nWinit # Jacobian wrt alpha (Overestimating)
        end

        nEleJac += (p.nTimeIntervals - 1) * p.nWinit # Jacobian wrt target state (next) Winit
        if p.nTimeIntervals > 2 # Jacobian wrt initial conditions, Winit
            nEleJac += (p.nTimeIntervals - 2) * p.nWinit * 2 * p.N
        end
        println("# allocated elements in Jac: ", nEleJac)
        jac_ele = zeros(nEleJac)
        jac_rows = zeros(Int32,nEleJac)
        jac_cols = zeros(Int32,nEleJac)
        println("Calling state_jacobian to evaluate the Jacobian of the state constraints")
        state_jacobian_idx(jac_rows, jac_cols, p, true)
        state_jacobian(pcof0, jac_ele, p, true)


        printJac = false
        if printJac
            println("row, col, jac")
            for j in 1:nEleJac
                println(jac_rows[j], ", ", jac_cols[j], ", ", jac_ele[j])
            end
        end

        println()
        pert = 1e-7
        
        one_par = 11 # 280 # 249 # 217 # 11 # for cons=36: 217-220, 233-236, 252
        c_rge = (33:64) # (1:32) # range of constraints to test 
        println("FD estimate of the jacobian of constraints = ", c_rge, ", wrt element kpar (col) = ", one_par)

        max_diff = 0.0
        for one_cons in c_rge

            jac_idx = find_elem(one_cons, one_par, jac_rows, jac_cols)

            if jac_idx < 0
                println("Warning: no entry in the jacobian matches row, col = ", one_cons, ", ", one_par)
                one_jac_ele = 0.0
            else
                #println("(row, col) = ", one_cons, ", ", one_par, " has index = ", jac_idx) 
                one_jac_ele = jac_ele[jac_idx]
            end

            pcof_p = copy(pcof0)
            pcof_p[one_par] += pert
            state_constraints(pcof_p, state_vec_global, p, false)
            obj_p = state_vec_global[one_cons]

            pcof_m = copy(pcof0)
            pcof_m[one_par] -= pert
            state_constraints(pcof_m, state_vec_global, p, false)
            obj_m = state_vec_global[one_cons]

            obj_grad_fd = 0.5*(obj_p - obj_m)/pert

            diff = obj_grad_fd - one_jac_ele
            global max_diff = max(max_diff, abs(diff))
            println("row # = ", one_cons, " col # = ", one_par, " cons_jac_fd = ", obj_grad_fd, " cons_jac_analyt = ", one_jac_ele, " diff = ", diff)
        end
        println()
        println("Max difference = ", max_diff)
    end

    if mpiObj.myRank == mpiObj.root
        println("Constraints test complete")
        println()
    end

end
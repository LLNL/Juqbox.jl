### Set up a test problem using one of the standard Hamiltonian models
using Juqbox
using Printf
using Plots
using LinearAlgebra
using Random
using Dates
using HDF5

root_dir = "/home/test/Juqbox.jl"
include(root_dir * "/src/ipopt_interface.jl")

if (length(ARGS) < 1)
    throw(ArgumentError("Input argument: integer (2-5) - number of qubits.\n\t\tfilename - hdf5 file for initial parameter."))
end

nqubit = parse(Int64, ARGS[1])

if (nqubit == 2)
    include(root_dir * "/examples/two_sys_noguard.jl")
    # assign the target gate
    target_gate0 = get_swap_1d_gate(2)
    # rotate target so that it will agree with the final unitary 
    theta = pi/4
    target_gate = exp(-im * theta) * target_gate0
elseif (nqubit == 3)
    include(root_dir * "/examples/three_sys_noguard.jl") # Dipole-dipole coupling
    target_gate = get_swap_1d_gate(3)
elseif (nqubit == 4)
    include(root_dir * "/examples/four_sys_noguard.jl") # Jaynes-Cummings
    # assign the target gate
    target_gate = get_swap_1d_gate(4)
elseif (nqubit == 5)
    include(root_dir * "/examples/five_sys_noguard.jl") # Dispersive nearest neighbor coupling
    # assign the target gate
    target_gate = get_swap_1d_gate(5)
else
    throw(ArgumentError("Number of qubits must be an integer within 2-5."))
end

maxIter = 200
fidType = 1 # fidType = 1 for Frobenius norm^2, or fidType = 2 for Infidelity, or fidType = 3 for infid^2
Ninterval = 3
penalty_strength = 4.0e-4

# TODO(kevin): nqubit=4,5 cases seem to use different setup_std_model function, but cannot find it anywhere.
retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, 
                        maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz,
                        rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres,
                        use_carrier_waves=use_carrier_waves, gammaJump=penalty_strength, nTimeIntervals=Ninterval, fidType=fidType)
#true)

println("time duration: ", T)

params = retval[1];
pcof0 = retval[2];
maxAmp = retval[3];
params.tik0 = 0.0

params.Lmult_r *= 0.0
params.Lmult_i *= 0.0

params.traceInfidelityThreshold = 0.0 # 1e-3 # better than 99.9% fidelity
if (length(ARGS) >= 2)
    pcof0 = h5read(ARGS[2], "/minimizer_parameter")

    # update the Lagrange multipliers
    prev_penalty_str = h5readattr(ARGS[2], "/")["penalty"]
   
    for interval = 1:params.nTimeIntervals-1
        params.Lmult_r[interval] = h5read(ARGS[2], "/Lmult_r/" * string(interval))
        params.Lmult_i[interval] = h5read(ARGS[2], "/Lmult_i/" * string(interval))
    end

    _, _, _, _, _, Cjump_r, Cjump_i = Juqbox.lagrange_obj(pcof0, params, false)
    for interval = 1:params.nTimeIntervals-1
        params.Lmult_r[interval] -= prev_penalty_str * Cjump_r[interval] # Should be a negative sign according to N&W
        params.Lmult_i[interval] -= prev_penalty_str * Cjump_i[interval]
    end
end

println("Setup complete")
println("objFuncType: ", params.objFuncType)
println("constraintType: ", params.constraintType)
println("Tikhonov penalty: ", params.tik0)

# obj_func(pcof_eval::Vector{Float64}) = eval_f_par2(pcof_eval, params)
# for objFuncType == 1, intermediate initial conditions (no leak term and no qudrature)
function obj_func(pcof::Vector{Float64})

    objf, finalDist, nrm2_Cjump, infid, tp, Cjump_r, Cjump_i = Juqbox.lagrange_obj(pcof, params, false)

    # NOTE: when the initial condition isn't unitary, the trace infidelity may be negative
    params.lastTraceInfidelity = max(1e-10, infid) 
    params.lastLeakIntegral = 0.0

    # debugging
    # println(pcof)
    # throw("Intentionally stopping here")
    
    # Add in Tikhonov regularization
    #tikhonovpenalty = Juqbox.tikhonov_pen(pcof, params)

    return objf, finalDist, nrm2_Cjump
end

obj_grad(pcof_eval::Vector{Float64}, grad_f_eval::Vector{Float64}) = eval_grad_f_par2(pcof_eval, grad_f_eval, params)
pcof_min, f_min, J_optim, grad_optim, step_optim, Niter = cgmin(obj_func, obj_grad, pcof0, params, 1.0e-5)
println("Number of iteration: ", Niter)

h5write("./cg_optim.h5", "/J_history", J_optim[1:Niter, :])
h5write("./cg_optim.h5", "/grad_history", grad_optim[1:Niter])
h5write("./cg_optim.h5", "/step_history", step_optim[1:Niter])
h5write("./cg_optim.h5", "/minimizer_parameter", pcof_min)
for interval = 1:params.nTimeIntervals-1
    h5write("./cg_optim.h5", "/Lmult_r/" * string(interval), params.Lmult_r[interval])
    h5write("./cg_optim.h5", "/Lmult_i/" * string(interval), params.Lmult_i[interval])
end
h5writeattr("./cg_optim.h5", "/", Dict("J_minimum" => f_min))
h5writeattr("./cg_optim.h5", "/", Dict("penalty" => penalty_strength))

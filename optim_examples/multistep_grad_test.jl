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
    throw(ArgumentError("Input argument: integer (2-5) - number of qubits."))
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
fidType = 2 # fidType = 1 for Frobenius norm^2, or fidType = 2 for Infidelity, or fidType = 3 for infid^2
Ninterval = 3

# TODO(kevin): nqubit=4,5 cases seem to use different setup_std_model function, but cannot find it anywhere.
retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, 
                        maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz,
                        rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres,
                        use_carrier_waves=use_carrier_waves, gammaJump=1.0, nTimeIntervals=Ninterval, fidType=fidType)
#true)

params = retval[1]
pcof0 = retval[2]
Random.seed!(millisecond(now()))
rnd_factor = 1.0 .+ 0.1 * (2.0 * rand(Float64, length(pcof0)) .- 1.0)
pcof0 = pcof0 .* rnd_factor
maxAmp = retval[3];

params.traceInfidelityThreshold = 0.0 # 1e-3 # better than 99.9% fidelity
params.Lmult_r *= 0.0
params.Lmult_i *= 0.0

if (length(ARGS) >= 2)
    pcof0 = h5read(ARGS[2], "/minimizer_parameter")

    # # update the Lagrange multipliers
    # prev_penalty_str = h5readattr(ARGS[2], "/")["penalty"]
   
    for interval = 1:params.nTimeIntervals-1
        params.Lmult_r[interval] = h5read(ARGS[2], "/Lmult_r/" * string(interval))
        params.Lmult_i[interval] = h5read(ARGS[2], "/Lmult_i/" * string(interval))
    end
end

println("Setup complete")
println("objFuncType: ", params.objFuncType)
println("constraintType: ", params.constraintType)

println(typeof(params.Uinit_r))
println(size(params.Uinit_r))
println(params.Uinit_r)
println(typeof(params.Uinit_i))
println(params.Uinit_i)

f = eval_f_par2(pcof0, params)
nCoeff = length(pcof0)
grad_f = zeros(nCoeff)
eval_grad_f_par2(pcof0, grad_f, params)
println("f0: ", f)
println("|g|: ", norm(grad_f))

gnorm = norm(grad_f)
gvec = grad_f / norm(grad_f);
sc = 1.0 / norm(grad_f);

Nk = 35
@printf("dx\tf1\tdfdx\terror\n");
for k = 1:Nk
    dx = 10^(-0.25 * k);
    pcof = pcof0 + dx * sc * gvec;

    f1 = eval_f_par2(pcof, params)
    dfdx = (f1 - f) / dx / sc;
    error = abs((dfdx - gnorm) / gnorm)
    @printf("%.5E\t%.5E\t%.5E\t%.5E\n", dx, f1, dfdx, error);
end

# println("Calling run_optimizer for derivative check")
# pcof = run_optimizer(params, pcof0, maxAmp, maxIter=maxIter)
# pl = plot_results(params, pcof)

# println("IPOpt iteration completed")

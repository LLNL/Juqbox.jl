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

if (length(ARGS) != 1)
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
Ninterval = 1

# TODO(kevin): nqubit=4,5 cases seem to use different setup_std_model function, but cannot find it anywhere.
retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, 
                        maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz,
                        rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres,
                        use_carrier_waves=use_carrier_waves, gammaJump=1.0, nTimeIntervals=Ninterval, fidType=fidType)
#true)

println("time duration: ", T)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 0.0 # 1e-3 # better than 99.9% fidelity
# zero out the Lagrange multipliers for now
params.Lmult_r *= 0.0
params.Lmult_i *= 0.0

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

Nk = 1000
amps = LinRange(-2e0, 2e0, Nk)
fk = zeros(Nk)
for (k, amp) in enumerate(amps)
    pcof = pcof0 + amp * sc * gvec;

    fk[k] = eval_f_par2(pcof, params)
    if (k % (Nk / 10) == 0)
        println(k)
    end
end

h5write("./param_space.h5", "/func", fk)
h5write("./param_space.h5", "/amp", Array(amps))
h5writeattr("./param_space.h5", "/func", Dict("gnorm" => gnorm))

# println("Calling run_optimizer for derivative check")
# pcof = run_optimizer(params, pcof0, maxAmp, maxIter=maxIter)
# pl = plot_results(params, pcof)

# println("IPOpt iteration completed")

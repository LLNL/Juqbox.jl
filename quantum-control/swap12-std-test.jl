### Set up a test problem using one of the standard Hamiltonian models
using Juqbox
using Printf
using Plots
using LinearAlgebra
using Random
using Dates

include("/home/test/Juqbox.jl/examples/two_sys_noguard.jl")
include("/home/test/Juqbox.jl/src/ipopt_interface.jl")

# assign the target gate
target_gate0 = get_swap_1d_gate(3)

# rotate target so that it will agree with the final unitary 
theta = pi/4
target_gate = exp(-im*theta)*target_gate0

maxIter = 200
fidType = 2 # fidType = 1 for Frobenius norm^2, or fidType = 2 for Infidelity, or fidType = 3 for infid^2
Ninterval = 2

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

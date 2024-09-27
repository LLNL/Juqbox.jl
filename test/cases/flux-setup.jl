#==========================================================
This routine initializes an optimization problem to recover 
a CNOT gate on a single qudit with 4 energy levels (and 2 
guard states) and showcases the use of uncoupled controls. 
The drift Hamiltonian in the rotating frame is
        H_0 = - 0.5*ξ_a(a^†a^†aa),
where a is the annihilation operator for the qudit. Here 
the control Hamiltonian includes the usual symmetric and 
anti-symmetric terms 
        H_s = p(t)(a + a^†),    H_a = q(t)(a - a^†),
which come from the rotating frame approximation and hence 
we refer to these as "coupled" controls. In 
addition, we consider a magnetic flux-tuning control term
                    H_f = f(t) a^†a,
which we refer to as an "uncoupled" control as it is 
invariant to the rotating frame approximation. The problem
parameters for this example are from Jonathan and Pranav at
UChicago: 
                    ω_a/2π = 5.0 GHz,
                    ξ_a/2π = 0.2 GHz.
We use Bsplines with carrier waves with carrier frequencies (rotating frame)
0, -ξ_a rad/ns.
==========================================================# 
using LinearAlgebra
#using Plots
#pyplot()
#using FFTW
using DelimitedFiles
using Printf
using Ipopt
using Random
using SparseArrays
using JLD2

#Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

using Juqbox

verbose = false
N = 4
Nguard = 2
Ntot = N + Nguard
	
samplerate = 64 # default number of time steps per unit time (plotting only)
casename = "flux" # base file name (used in optimize-once.jl)

# Set to false for dense matrix operations
use_sparse = true

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 5.0 # 4.1
xa = 0.2
rot_freq = [fa, fa] # Rotational frequencies for each control Hamiltonian

# duration
T = 11.0 # Tperiod/4

Ident = Matrix{Float64}(I, Ntot, Ntot)   
utarget = Matrix{ComplexF64}(I, Ntot, N)
vtarget = Matrix{ComplexF64}(I, Ntot, N)

# CNOT target
utarget[:,4] = Ident[:,3]
utarget[:,3] = Ident[:,4]

omega1 = Juqbox.setup_rotmatrices([N], [Nguard], [fa])

# Compute Ra*utarget
rot1 = Diagonal(exp.(im*omega1*T))

# target in the rotating frame
vtarget = rot1*utarget

Nosc  = 1 
Nfreq = 2 # number of carrier frequencies

Random.seed!(2456)

# setup drift Hamiltonian
number = Diagonal(collect(0:Ntot-1))

H0 = -0.5*(2*pi)*xa* (number*number - number)

# lowering matrix
amat = Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U) # standard lowering operator matrix
# raising matrix
adag = transpose(amat) # raising operator matrix

Hsym_ops=[ Array(amat+adag), Array(adag*amat) ]
Hanti_ops=[ Array(amat-adag), Array(zeros(Ntot,Ntot)) ]

H0 = Array(H0)

Nctrl = length(Hsym_ops)

# setup carrier frequencies
om = zeros(Nctrl,Nfreq)

@assert(Nfreq==1 || Nfreq==2 || Nfreq==3)
if Nfreq == 2
    om[1:Nctrl,2] .= -2.0*pi*xa # coupling freq for both ctrl funcs (re/im)
elseif Nfreq == 3
    om[:,2] .= -2.0*pi*xa # 1st ctrl, re
    om[:,3] .= -4.0*pi*xa # coupling freq for both ctrl funcs (re/im)
end

# Note: same frequencies for each p(t) (x-drive) and q(t) (y-drive)
if verbose
    println("Carrier frequencies [GHz]: ", om[:,:]./(2*pi))
end

# max parameter amplitude
maxpar = 0.08
max_flux = 2*pi*5.0
# max_flux = maxpar

# Initial conditions
Ident = Matrix{Float64}(I, Ntot, Ntot)   
U0 = Ident[1:Ntot,1:N]

# setup the initial parameter vector, either randomized or from file
startFromScratch = false # true
startFile = "cases/flux.dat"
useBarrier = true

if startFromScratch
    # D1 smaller than 3 does not work
    D1 = 30 # Number of B-spline coefficients per frequency, sin/cos and real/imag
    nCoeff = 2*Nctrl*Nfreq*D1
    pcof0  = zeros(nCoeff)    
    pcof0 = (rand(nCoeff) .- 0.5).*maxpar*0.1
else
    # the data on the startfile must be consistent with the setup!
    # use if you want to have initial coefficients read from file
    pcof0 = vec(readdlm(startFile))
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Nctrl*Nfreq) # factor '2' is for sin/cos

    nCoeff = 2*Nctrl*Nfreq*D1 # just to be safe if the file doesn't contain the right number of elements

    if verbose
        println("*** Starting from B-spline coefficients in file: ", startFile)
    end
end

# Estimate time step for simulation
nsteps = Juqbox.calculate_timestep(T, H0, Hsym_ops, Hanti_ops, [maxpar, max_flux])
if verbose
    println( "# time steps: ", nsteps)
end

Integrator_id = 1
# setup the simulation parameters
params = Juqbox.objparams([N], [Nguard], T, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=rot_freq,
                          Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, use_sparse=use_sparse, Integrator = Integrator_id)
# params = Juqbox.objparams([N], [Nguard], T, nsteps, U0, vtarget, om, H0, Hunc_ops)
params.saveConvHist = true
if Integrator_id == 2
    linear_solver = Juqbox.lsolver_object(solver=Juqbox.JACOBI_SOLVER_M,max_iter=100,tol=1e-12,nrhs=prod(N))
    params.linear_solver = linear_solver
    
end
# Quiet mode for testing
params.quiet = !verbose

#Tikhonov regularization coefficients
params.tik0 = 0.1

params.traceInfidelityThreshold =  1e-5

# Set bounds on coefficients
minCoeff, maxCoeff = Juqbox.assign_thresholds(params,D1,[maxpar, max_flux])


# For ipopt
maxIter = 100 # optional argument
lbfgsMax = 250 # optional argument

if verbose
    println("*** Settings ***")
    println("System Hamiltonian coefficients: (fa, xa) =  ", fa, xa)
    println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
    println("Using B-spline basis functions with carrier wave, # freq = ", Nfreq)
    println("Number of coefficients per spline = ", D1, " Total number of parameters = ", nCoeff)
    println("Max parameter amplitudes: maxpar = ", maxpar)
    println("Tikhonov coefficient: tik0 = ", params.tik0)
end

if params.Integrator_id == 1
    wa = Juqbox.Working_Arrays(params, nCoeff)
elseif params.Integrator_id == 2
    wa = Juqbox.Working_Arrays_M(params, nCoeff)
end
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter=maxIter, lbfgsMax=lbfgsMax)

# uncomment to run the gradient checker for the initial pcof
#addOption( prob, "derivative_test", "first-order"); # for testing the gradient

# experiment with scale factors
if @isdefined addOption
    addOption( prob, "nlp_scaling_method", "user-scaling"); # for testing the gradient
else
    AddIpoptStrOption( prob, "nlp_scaling_method", "user-scaling")
end


if verbose
    println("Initial coefficient vector stored in 'pcof0'")
end

# grad_storage = zeros(size(pcof0))


# for i = 1:length(pcof0)
#     println("Finite difference for parameter: ", i)
#     perturb = zeros(size(pcof0))
#     perturb[i] = 0.0000001
#     objfv, _, _ = traceobjgrad(pcof0, params, wa, false, false)
#     objfv2, _, _ = traceobjgrad(pcof0 + perturb, params, wa, false, false)
#     grad_storage[i] = (objfv2 - objfv)/0.0000001
# end

# save_object("flux-ref-IMR.jld2", grad_storage)
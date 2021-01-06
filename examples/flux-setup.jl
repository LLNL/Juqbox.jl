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
using Plots
pyplot()
using FFTW
using DelimitedFiles
using Printf
using Ipopt
using Random
using SparseArrays

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

using Juqbox

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
rot_freq = [fa] # Rotational frequencies

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


if (use_sparse)
    Hsym_ops=[sparse(amat+adag)]
    Hanti_ops=[sparse(amat-adag)]
    Hunc_ops=[sparse(adag*amat)]
    # Hunc_ops=[sparse(amat + adag)]
    dropzeros!(Hsym_ops[1])
    dropzeros!(Hanti_ops[1])
    dropzeros!(Hunc_ops[1])
    H0 = sparse(H0)
else
    Hsym_ops=[Array(amat+adag)]
    Hanti_ops=[Array(amat-adag)]
    Hunc_ops=[Array(adag*amat)]
    # Hunc_ops=[Array(amat - adag)]
    H0 = Array(H0)
end

Ncoupled = length(Hsym_ops)
Nunc = length(Hunc_ops)

# setup carrier frequencies
om = zeros(Ncoupled,Nfreq)
use_bcarrier = true

if use_bcarrier
    @assert(Nfreq==1 || Nfreq==2 || Nfreq==3)
    if Nfreq == 2
        om[1:Ncoupled,2] .= -2.0*pi*xa # coupling freq for both ctrl funcs (re/im)
    elseif Nfreq == 3
        om[:,2] .= -2.0*pi*xa # 1st ctrl, re
        om[:,3] .= -4.0*pi*xa # coupling freq for both ctrl funcs (re/im)
    end
end

# Note: same frequencies for each p(t) (x-drive) and q(t) (y-drive)
println("Carrier frequencies [GHz]: ", om[:,:]./(2*pi))

# max parameter amplitude
maxpar = 0.08
max_flux = 2*pi*5.0
# max_flux = maxpar

# Initial conditions
Ident = Matrix{Float64}(I, Ntot, Ntot)   
U0 = Ident[1:Ntot,1:N]

# setup the initial parameter vector, either randomized or from file
startFromScratch = true # true
startFile = "flux-pcof-opt-alpha-0.5.dat"
useBarrier = true

if startFromScratch
    # D1 smaller than 3 does not work
    D1 = 30 # Number of B-spline coefficients per frequency, sin/cos and real/imag
    nCoeff = (2*Ncoupled + Nunc)*Nfreq*D1
    pcof0  = zeros(nCoeff)    
    pcof0 = (rand(nCoeff) .- 0.5).*maxpar*0.1
else
    # the data on the startfile must be consistent with the setup!
    # use if you want to have initial coefficients read from file
    pcof0 = vec(readdlm(startFile))
    nCoeff = length(pcof0)
    D1 = div(nCoeff, (2*Ncoupled + Nunc)*Nfreq) # factor '2' is for sin/cos

    nCoeff = (2*Ncoupled + Nunc)*Nfreq*D1 # just to be safe if the file doesn't contain the right number of elements

    println("*** Starting from B-spline coefficients in file: ", startFile)
end

# Estimate time step for simulation
maxeig,nsteps = Juqbox.calculate_timestep(T,D1,H0,Hsym_ops,Hanti_ops,Hunc_ops,[maxpar],[max_flux])
println("Max est. eigenvalue = ", maxeig, " # time steps: ", nsteps)

# setup the simulation parameters
params = Juqbox.objparams([N], [Nguard], T, nsteps, U0, vtarget, om, H0, Hsym_ops, Hanti_ops, Hunc_ops)
# params = Juqbox.objparams([N], [Nguard], T, nsteps, U0, vtarget, om, H0, Hunc_ops)
params.saveConvHist = true

#Tikhonov regularization coefficients
params.tik0 = 0.1

params.traceInfidelityThreshold =  1e-5

# Set bounds on coefficients
minCoeff, maxCoeff = Juqbox.assign_thresholds(params,D1,[maxpar],[max_flux])


# For ipopt
maxIter = 100 # optional argument
lbfgsMax = 250 # optional argument

println("*** Settings ***")
println("System Hamiltonian coefficients: (fa, xa) =  ", fa, xa)
println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
if use_bcarrier
  println("Using B-spline basis functions with carrier wave, # freq = ", Nfreq)
else
  println("Using regular B-spline basis functions")
end
println("Number of coefficients per spline = ", D1, " Total number of parameters = ", nCoeff)
println("Max parameter amplitudes: maxpar = ", maxpar)
println("Tikhonov coefficient: tik0 = ", params.tik0)

# Estimate number of terms in Neumann series for time stepping (Default 3)
tol = eps(1.0); # machine precision
Juqbox.estimate_Neumann!(tol, T, params, [maxpar], [max_flux])

wa = Juqbox.Working_Arrays(params, nCoeff)
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter, lbfgsMax, startFromScratch)

# uncomment to run the gradient checker for the initial pcof
#addOption( prob, "derivative_test", "first-order"); # for testing the gradient

# experiment with scale factors
addOption( prob, "nlp_scaling_method", "user-scaling");

# Scale the variables for flux_charge term
scaling_factor = 1.0

x_scaling = ones(length(pcof0))
g_scaling = ones(length(pcof0))
for s in 2*Ncoupled+Nunc:2*Ncoupled+Nunc # flux charge coefficients
  for f in 1:Nfreq
    offset = (f-1)*D1 + (s-1)*Nfreq*D1
    x_scaling[offset+1:offset+D1] .= scaling_factor
  end
end
obj_scaling = 1.0
setProblemScaling(prob,obj_scaling,x_scaling,g_scaling)

println("Initial coefficient vector stored in 'pcof0'")

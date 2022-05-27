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
addition, we consider a flux-charge term
                    H_f = f(t) a^†a,
which we refer to as an "uncoupled" control as it is 
invariant to the rotating frame approximation. The problem
parameters for this example are from Jonathan and Pranav at
UChicago: 
                    ω_a =  2π × 5.0 Grad/s,
                    ξ_a =  2π × 0.2 Grad/s.
We use Bsplines with carrier waves with frequencies
0, ξ_a Grad/s.
==========================================================# 
using LinearAlgebra
#using Plots
#pyplot()
#using FFTW
#using DelimitedFiles
using Printf
#using Ipopt
using Random
using SparseArrays

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

#import Juqbox

verbose = false
N = 4
Nguard = 2
Ntot = N + Nguard
	
samplerate = 64 # default number of time steps per unit time (plotting only)
casename = "cnot-lab" # base file name (used in optimize-once.jl)

# Set to false for dense matrix operations
use_sparse = true

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 5.0 # 4.1
xa = 0.2

# duration
T = 12.0 # Tperiod/4

Ident = Matrix{Float64}(I, Ntot, Ntot)   
utarget = Matrix{ComplexF64}(I, Ntot, N)

# CNOT target
utarget[:,4] = Ident[:,3]
utarget[:,3] = Ident[:,4]

Nosc  = 1 
Nfreq = 3 # number of carrier frequencies

Random.seed!(2456)

# setup drift Hamiltonian
number = Diagonal(collect(0:Ntot-1))

H0 = 2*pi*fa*number-0.5*(2*pi)*xa* (number*number - number)

# lowering matrix
amat = Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U) # standard lowering operator matrix
# raising matrix
adag = transpose(amat) # raising operator matrix

Hunc_ops=[Array(amat + adag)]
H0 = Array(H0)

Ncoupled = 0
Nunc = length(Hunc_ops)

# setup carrier frequencies
om = zeros(1,Nfreq)
use_bcarrier = true

if use_bcarrier
    @assert(Nfreq==1 || Nfreq==2 || Nfreq==3)
    if Nfreq == 2
        om[1:1,2] .= -2.0*pi*fa
    elseif Nfreq == 3
        om[:,2] .= -2.0*pi*fa
        om[:,3] .= 2.0*pi*fa
    end
end

# Note: same frequencies for each p(t) (x-drive) and q(t) (y-drive)
#println("Carrier frequencies [GHz]: ", om[:,:]./(2*pi))

# max parameter amplitude
max_unc = 2*pi*5.0

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
    pcof0 = (rand(nCoeff) .- 0.5).*max_unc*0.1
else
    # the data on the startfile must be consistent with the setup!
    # use if you want to have initial coefficients read from file
    pcof0 = vec(readdlm(startFile))
    nCoeff = length(pcof0)
    D1 = div(nCoeff, (2*Ncoupled + Nunc)*Nfreq) # factor '2' is for sin/cos

    nCoeff = (2*Ncoupled + Nunc)*Nfreq*D1 # just to be safe if the file doesn't contain the right number of elements

#    println("*** Starting from B-spline coefficients in file: ", startFile)
end

# Estimate time step for simulation
nsteps = Juqbox.calculate_timestep(T, H0, Hunc_ops, [max_unc])

# setup the simulation parameters
params = Juqbox.objparams([N], [Nguard], T, nsteps, Uinit=U0, Utarget=utarget, Cfreq=om, Rfreq=[fa], Hconst=H0, Hunc_ops=Hunc_ops)
params.saveConvHist = true
params.nsteps *= 5

# Quiet mode for testing
params.quiet = true

#Tikhonov regularization coefficients
params.tik0 = 1e-3

# Set bounds on coefficients
minCoeff, maxCoeff = Juqbox.assign_thresholds(params,D1,[0.0],[max_unc])

# For ipopt
maxIter = 50 # optional argument
lbfgsMax = 250 # optional argument

if verbose
    println("*** Settings ***")
    println("System Hamiltonian coefficients: (fa, xa) =  ", fa, xa)
    println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
    if use_bcarrier
        println("Using B-spline basis functions with carrier wave, # freq = ", Nfreq)
    else
        println("Using regular B-spline basis functions")
    end
    println("Number of coefficients per spline = ", D1, " Total number of parameters = ", nCoeff)
    println("Max parameter amplitudes: max_unc = ", max_unc)
    println("Tikhonov coefficient: tik0 = ", params.tik0)
end

# Estimate number of terms in Neumann series for time stepping (Default 3)
tol = eps(1.0); # machine precision
Juqbox.estimate_Neumann!(tol, params, Float64[], [max_unc])

wa = Juqbox.Working_Arrays(params, nCoeff)
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter, lbfgsMax)

if @isdefined addOption
    addOption(prob, "derivative_test", "first-order"); # for testing the gradient
else
    AddIpoptStrOption(prob, "derivative_test", "first-order")
end

#println("Initial coefficient vector stored in 'pcof0'")

#==========================================================
This routine initializes an optimization problem to recover 
a |0⟩ to |2⟩ swap gate on a single qudit with 3 energy 
levels (and 1 guard state). The  drift Hamiltonian in the 
rotating frame is
        H0 = 2π*diagm([0 0 -2.2538e-1 -7.0425e-1]).
Here the control Hamiltonian includes the usual symmetric 
and anti-symmetric terms 
     H_{sym} = p(t)(a + a^†),    H_{asym} = q(t)(a - a^†),
where a is the annihilation operator for the qudit.
The problem parameters for this example are: 
                ω_a =  2π × 4.09947    Grad/s,
                ξ_a =  2π × 2.2538e-01 Grad/s.
We use Bsplines with carrier waves with frequencies
0, ξ_a Grad/s.
==========================================================# 
using LinearAlgebra
using Plots
using FFTW
using DelimitedFiles
using Printf
using Ipopt
using Random
using JLD2
using FastGaussQuadrature

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

using Juqbox # quantum control module

# Default values for uniform distribution on [-ϵ/2,ϵ/2]
if(!@isdefined(ep_max))
    ep_max = 2*pi*3e-2
end

# Default number of quadrature nodes to evaluate the 
# expected value of the objective function via 
# Gaussian quadrature, i.e.
#       E[J] = ∑ w[k] J[x[k]]
# where w,x are the weights and nodes on [-ϵ/2,ϵ/2]
if(!@isdefined(nquad))
    nquad = 20
    nquad = 1
end
nodes, weights = gausslegendre(nquad)

# Map nodes to [-ϵ/2,ϵ/2]
nodes .*= 0.5*ep_max
weights .*= 0.5

N = 3 # Number of essential energy levels
Nguard = 1 # Number of guard/forbidden energy levels
Ntot = N + Nguard # Total number of energy levels

samplerate = 32 # for output files

T = 300.0 # Duration of gate

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.10336
xa = 0.2198
rot_freq = [fa] # Used to calculate the lab frame ctrl function

# setup drift Hamiltonian
number = Diagonal(collect(0:Ntot-1))

H0 = -0.5*(2*pi)*xa* (number*number - number) # xa is in GHz

utarget = zeros(ComplexF64,Ntot,N)

# 0>  to  |2>  swap gate
if N >= 3
    utarget[1,1] = 0 #1/sqrt(2)
    utarget[2,1] = 0
    utarget[3,1] = 1 #1/sqrt(2)
#
    utarget[1,2] = 0
    utarget[2,2] = 1
    utarget[3,2] = 0
    #
    utarget[1,3] = 1 #1/sqrt(2)
    utarget[2,3] = 0
    utarget[3,3] = 0 #-1/sqrt(2)
#
end

if N==4
    utarget[4,4] = 1
end

omega1 = Juqbox.setup_rotmatrices([N], [Nguard], [fa])

# Compute Ra*utarget
rot1 = Diagonal(exp.(im*omega1*T))

# target in the lab frame
vtarget = utarget

startFromScratch = true
# startFile is used when startFromScratch = false
startFile="swap02-baseline-pcof-opt.jld2" # "swap02-pert-4em1-pcof-opt.jld2"

usePrior = false #  true
priorFileName = startFile # Usually makes sense to also use the start file as the prior, but not required

# setup drift Hamiltonian
number = Diagonal(collect(0:Ntot-1))

# lowering matrix 
amat = Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U) # standard lowering matrix
adag = Array(transpose(amat));
Hsym_ops=[Array(amat + adag)]
Hanti_ops=[Array(amat - adag)]
H0 = Array(H0)

Ncoupled = length(Hsym_ops) # Number of paired Hamiltonians
Nfreq= 2 # number of carrier frequencies 3 gives a cleaner sol than 2

# setup carrier frequencies
use_bcarrier = true
om = zeros(Ncoupled,Nfreq)
if use_bcarrier
    om[1:Ncoupled,2] .= -2.0*pi *xa       # Note negative sign
end
println("Carrier frequencies [GHz]: ", om[1,:]./(2*pi))
println("H0: ", H0)

maxctrl = 2*pi*1.2e-2

#max amplitude (in angular frequency) 2*pi*GHz
maxamp = zeros(Nfreq)
if Nfreq >= 5
    const_fact = 1.0/(Nfreq-2)
    maxamp[1] = maxctrl*const_fact
    maxamp[2:Nfreq] .= maxctrl*(1.0-const_fact)/(Nfreq-1) # max B-spline coefficient amplitude, factor 3.0 is ad hoc
elseif Nfreq >= 4
    const_fact = 0.4
    maxamp[1] = maxctrl*const_fact
    maxamp[2:Nfreq] .= maxctrl*(1.0-const_fact)/(Nfreq-1) # max B-spline coefficient amplitude, factor 3.0 is ad hoc
elseif Nfreq >= 3
    const_fact = 0.45
    maxamp[1] = maxctrl*const_fact
    maxamp[2:Nfreq] .= maxctrl*(1.0-const_fact)/(Nfreq-1) # max B-spline coefficient amplitude, factor 3.0 is ad hoc
else
    maxamp .= maxctrl/Nfreq
end

maxpar = maximum(maxamp)
nsteps = calculate_timestep(T, H0, Hsym_ops, Hanti_ops, [maxctrl])
println("# time steps: ", nsteps)

# Initial conditions for basis
U0 = initial_cond([N], [Nguard])

# params = Juqbox.parameters([N], [Nguard], T, nsteps, U0, vtarget, om, H0, Hsym_ops, Hanti_ops)
params = Juqbox.objparams([N], [Nguard], T, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=rot_freq,
                          Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, wmatScale=1.0)

if usePrior
    Juqbox.setup_prior!(params, priorFileName)
    params.tik0 = 100.0 # increase Tikhonov regularization coefficient
end

Random.seed!(2456)

# initial parameter guess
if startFromScratch
  D1 = 12 # Number of B-spline coefficients per frequency, sin/cos and real/imag
  nCoeff = 2*Ncoupled*Nfreq*D1 # factor '2' is for sin/cos
  pcof0 = (rand(nCoeff) .- 0.5).*maxpar*0.1

  if(nquad == 1)
    D1 = 12 # Number of B-spline coefficients per frequency, sin/cos and real/imag
    nCoeff = 2*Ncoupled*Nfreq*D1 # factor '2' is for sin/cos
    pcof0 = (rand(nCoeff) .- 0.5).*maxpar*0.1
  end
  println("*** Starting from random pcof with amplitude ", maxpar*0.1)
else
    # use if you want to have initial coefficients read from file
    @load startFile pcof
    pcof0 = pcof
    println("*** Starting from B-spline coefficients in file: ", startFile)
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Ncoupled*Nfreq)  # number of B-spline coeff per control function
end


# min and max coefficient values (set first and last two to zero)
useBarrier = true
minCoeff, maxCoeff = Juqbox.assign_thresholds_freq(maxamp, Ncoupled, Nfreq, D1)
zero_start_end!(params, D1, minCoeff, maxCoeff)

println("*** Settings ***")
println("System Hamiltonian coefficients [GHz]: (fa, xa) =  ", fa, xa)
println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
println("Using B-spline basis functions with carrier wave, # freq = ", Nfreq)
println("Number of coefficients per spline = ", D1, " Total number of parameters = ", nCoeff)
println("Max parameter amplitudes: maxpar = ", maxpar)
println("Target rotating frame control amplitude, maxctrl = ", maxctrl, " [rad/ns], ", maxctrl*0.5/pi, " [GHz]")
# params.tik0 = 0
println("Tikhonov coefficients: tik0 = ", params.tik0)

# optional arguments to setup_ipopt_problem()
maxIter = 150
lbfgsMax = 5
ipTol = 1e-5 
acceptTol = 1e-5
acceptIter = 15

# Estimate number of terms in Neumann series for time stepping (Default 3)
tol = eps(1.0); # machine precision
Juqbox.estimate_Neumann!(tol, params, [maxpar])

# Allocate all working arrays
wa = Juqbox.Working_Arrays(params,nCoeff)
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter=maxIter, lbfgsMax=lbfgsMax, startFromScratch=startFromScratch, ipTol=ipTol,acceptTol=acceptTol, acceptIter=acceptIter, nodes=nodes, weights=weights)

println("Initial coefficient vector stored in 'pcof0'")
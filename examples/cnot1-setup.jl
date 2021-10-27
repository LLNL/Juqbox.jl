#==========================================================
This routine initializes an optimization problem to recover 
a CNOT gate on a single qudit with 4 energy levels (and 2 
guard states). The drift Hamiltonian in the rotating frame 
is
              H_0 = - 0.5*ξ_a(a^†a^†aa),
where a is the annihilation operator for the qudit. 
Here the control Hamiltonian includes the usual symmetric 
and anti-symmetric terms 
     H_{sym} = p(t)(a + a^†),    H_{asym} = q(t)(a - a^†)
which come from the rotating frame approximation and hence 
we refer to these as "coupled" controls.
The problem parameters are:
                ω_a =  2π × 4.10595   Grad/s,
                ξ_a =  2π × 2(0.1099) Grad/s.
We use Bsplines with carrier waves with frequencies
0, ξ_a, 2ξ_a Grad/s.
==========================================================# 
using LinearAlgebra
using Plots
gr()
#pyplot()
using FFTW
using DelimitedFiles
using Printf
using Ipopt
using Random

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

using Juqbox # quantum control module

Nosc = 1 # Number of coupled sub-systems = oscillators
N = 4 # Number of essential energy levels

Nguard = 2 # Number of extra guard levels
Ntot = N + Nguard # Total number of energy levels
	
T = 100.0 # Duration of gate. 

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.10336
xa = 0.2198
rot_freq = [fa] # Used to calculate the lab frame ctrl function

# setup drift Hamiltonian
number = Diagonal(collect(0:Ntot-1))

H0 = -0.5*(2*pi)*xa* (number*number - number) # xa is in GHz

# lowering matrix 
amat = Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U) # standard lowering matrix
adag = Array(transpose(amat));
Hsym_ops=[Array(amat + adag)]
Hanti_ops=[Array(amat - adag)]
H0 = Array(H0)

# Estimate time step
maxctrl = 0.001*2*pi * 8.5 #  9, 10.5, 12, 15 MHz

# K1 =  H0 + maxctrl.*( amat +  amat') + 1im* maxctrl.*(amat - amat')
# lamb = eigvals(Array(K1))
# maxeig = maximum(abs.(lamb)) 

# Pmin = 40
# samplerate1 = maxeig*Pmin/(2*pi)
# nsteps = ceil(Int64, T*samplerate1)

nsteps = calculate_timestep(T, H0, Hsym_ops, Hanti_ops, [maxctrl])
println("# time steps: ", nsteps)

use_bcarrier = true # Use carrier waves in the control pulses?

if use_bcarrier
    Nfreq = 3 # number of carrier frequencies
else
    Nfreq = 1
end

Ncoupled = length(Hsym_ops) # Here, Ncoupled = 1
om = zeros(Ncoupled,Nfreq)

if use_bcarrier
    om[1:Ncoupled,2] .= -2.0*pi *xa       # Note negative sign
    om[1:Ncoupled,3] .= -2.0*pi* 2.0*xa
end
println("Carrier frequencies [GHz]: ", om[1,:]./(2*pi))

maxamp = zeros(Nfreq)

if Nfreq >= 3
    const_fact = 0.45
    maxamp[1] = maxctrl*const_fact
    maxamp[2:Nfreq] .= maxctrl*(1.0-const_fact)/(Nfreq-1) # split the remainder equally
else
    # same threshold for all frequencies
    maxamp .= maxctrl/Nfreq
end

maxpar = maximum(maxamp)

Ident = Matrix{Float64}(I, Ntot, Ntot)   

# CNOT trarget
utarget = Ident[1:Ntot,1:N]
utarget[:,3] = Ident[:,4]
utarget[:,4] = Ident[:,3]

omega1 = Juqbox.setup_rotmatrices([N], [Nguard], [fa])

# Compute Ra*utarget
rot1 = Diagonal(exp.(im*omega1*T))

# target in the rotating frame
vtarget = rot1*utarget

# Initial conditions
U0 = Ident[1:Ntot,1:N]

params = Juqbox.objparams([N], [Nguard], T, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=rot_freq,
                          Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops)

# initial parameter guess

# D1 smaller than 5 does not work
D1 = 10 # Number of B-spline coefficients per segment
nCoeff = 2*Ncoupled*Nfreq*D1 # factor '2' is for sin/cos

# Random.seed!(12456)

startFromScratch = true # false
startFile="cnot-pcof-opt.dat"

# initial parameter guess
if startFromScratch
    pcof0 = maxpar*0.01 * rand(nCoeff)
    println("*** Starting from random pcof with amplitude ", maxpar*0.01)
else
    # use if you want to read the initial coefficients from file
    pcof0 = vec(readdlm(startFile))
    println("*** Starting from B-spline coefficients in file: ", startFile)
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Ncoupled*Nfreq)  # number of B-spline coeff per control function
end

# min and max coefficient values
useBarrier = true
minCoeff, maxCoeff = Juqbox.assign_thresholds_freq(maxamp, Ncoupled, Nfreq, D1)

samplerate = 32 # only used for plotting
casename = "cnot1" # base file name (used in optimize-once.jl)

maxIter = 200 # 0  # optional argument
lbfgsMax = 250 # optional argument
ipTol = 1e-5   # optional argument
acceptTol = ipTol # 1e-4 # acceptable tolerance 
acceptIter = 15

println("*** Settings ***")
println("System Hamiltonian coefficients [GHz]: (fa, xa) =  ", fa, xa)
println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
if use_bcarrier
  println("Using B-spline basis functions with carrier wave, # freq = ", Nfreq)
else
  println("Using regular B-spline basis functions")
end
println("Number of coefficients per spline = ", D1, " Total number of parameters = ", nCoeff)
for q=1:Nfreq
    println("Carrier frequency: ", om[q]/(2*pi), " GHz, max parameter amplitude = ", 1000*maxamp[q]/(2*pi), " MHz")
end
println("Tikhonov coefficients: tik0 = ", params.tik0)

# Estimate number of terms in Neumann series for time stepping (Default 3)
tol = eps(1.0); # machine precision
Juqbox.estimate_Neumann!(tol, params, [maxpar])

wa = Juqbox.Working_Arrays(params,nCoeff)
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter, lbfgsMax, startFromScratch)

println("Initial coefficient vector stored in 'pcof0'")

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
#using Plots
#pyplot()
#using FFTW
using DelimitedFiles
using Printf
#using Ipopt
using Random

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

#import Juqbox

verbose = false
Nosc=1 # Number of oscillators

N = 3 # 4 # Number of essential energy levels
Nguard = 1 # 0 # Number of guard/forbidden energy levels
Ntot = N + Nguard # Total number of energy levels

samplerate = 32 # for output files
casename = "swap02" # naming output files"

T = 150.0 # 150.0 # 200.0 # 100.0 # Duration of gate

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
freq_alice=[0, 4.09947, 3.87409, 3.6206] # GHz
fa = freq_alice[2]
@assert(Ntot <= 4) # we don't know any higher frequencies

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

omega1 = Juqbox.setup_rotmatrices([N], [Nguard], [freq_alice[2]])

# Compute Ra*utarget
rot1 = Diagonal(exp.(im*omega1*T))

# target in the rotating frame
vtarget = rot1*utarget

startFromScratch = false # false
startFile = "cases/swap02.dat"
use_bcarrier = true

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.10595
xa = 2* 0.1099

number = Diagonal(collect(0:Ntot-1))

H0 = -0.5*(2*pi)*xa* (number*number - number) # xa is in GHz

# lowering matrix 
amat = Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U) # standard loweing matrix
adag = Array(transpose(amat));
Hsym_ops=[Array(amat + adag)]
Hanti_ops=[Array(amat - adag)]
H0 = Array(H0)

# END NEW SETUP

Ncoupled = length(Hsym_ops) # Number of paired Hamiltonians
Nfreq= 2 # number of carrier frequencies 3 gives a cleaner sol than 2

# setup carrier frequencies
om = zeros(Ncoupled,Nfreq)
# seg=1 is the real, seg=2 is the imaginary. Use the same frequencies for both
if Nfreq == 3
    om[1:Ncoupled,2] .= H0[3,3]
    om[1:Ncoupled,3] .= H0[4,4] - H0[3,3]
elseif Nfreq == 2
    om[1:Ncoupled,2] .= H0[3,3]
end
#println("Carrier frequencies [GHz]: ", om[1,:]./(2*pi))

#max amplitude (in angular frequency) 2*pi*GHz
# NOTE: optimize-once.jl uses maxpar to generate filenames
maxpar =2*pi*0.0132/Nfreq/2 # attempt to limit lab frame amplitude to 0.0132 = 0.5/6/2/pi

# Estimate time step
K1 =  H0 + maxpar.*( amat +  amat') + 1im* maxpar.*(amat - amat')
lamb = eigvals(Array(K1))
maxeig = maximum(abs.(lamb)) 

Pmin = 80
samplerate1 = maxeig*Pmin/(2*pi)
nsteps = ceil(Int64, T*samplerate1)
#println("Duration T = ", T, " number of time-steps = ", nsteps)

# Initial conditions
Ident = Matrix{Float64}(I, Ntot, Ntot)   
U0 = Ident[1:Ntot,1:N]

# try the new Juqbox2 module
params = Juqbox.objparams([N], [Nguard], T, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=[freq_alice[2]],
                          Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops)

# Quiet mode for testing
params.quiet = true

# set random number seed to make algorithm deterministic
Random.seed!(2456)

# initial parameter guess
if startFromScratch
  D1 = 10 # Number of B-spline coefficients per frequency, sin/cos and real/imag
  nCoeff = 2*Ncoupled*Nfreq*D1 # factor '2' is for sin/cos
  #pcof0  = zeros(nCoeff)
  pcof0 = (rand(nCoeff) .- 0.5).*maxpar*0.1
#  println("*** Starting from random pcof with amplitude ", maxpar*0.1)
else
  # use if you want to have initial coefficients read from file
  pcof0 = vec(readdlm(startFile))
    #  println("*** Starting from B-spline coefficients in file: ", startFile)
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Ncoupled*Nfreq)  # number of B-spline coeff per control function
end

# min and max coefficient values
useBarrier = true
minCoeff = -maxpar*ones(nCoeff);
maxCoeff = maxpar*ones(nCoeff);

if verbose
    println("*** Settings ***")
    println("Resonant frequencies [GHz] = ", freq_alice[1:Ntot])
    println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
    if use_bcarrier
        println("Using B-spline basis functions with carrier wave, # freq = ", Nfreq)
    else
        println("Using regular B-spline basis functions")
    end
    println("Number of coefficients per spline = ", D1, " Total number of parameters = ", nCoeff)
    println("Max parameter amplitudes: maxpar = ", maxpar)
    println("Tikhonov coefficients: tik0 = ", params.tik0)
end

# optional arguments to setup_ipopt_problem()
maxIter = 50 
lbfgsMax = 250 


# Estimate number of terms in Neumann series for time stepping (Default 3)
tol = eps(1.0); # machine precision
Juqbox.estimate_Neumann!(tol, params, [maxpar])

# Allocate all working arrays
wa = Juqbox.Working_Arrays(params, nCoeff)
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter=maxIter, lbfgsMax=lbfgsMax)

# uncomment to run the gradient checker for the initial pcof
# if @isdefined addOption
#     addOption( prob, "derivative_test", "first-order"); # for testing the gradient
# else
#     AddIpoptStrOption( prob, "derivative_test", "first-order")
# end


#println("Initial coefficient vector stored in 'pcof0'")

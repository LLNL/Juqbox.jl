#==========================================================
This routine initializes an optimization problem on a 
single qubit with 2 energy levels (and no guard states) 
where the analytical solution is a constant control 
function, i.e. a Rabi oscillator. The drift Hamiltonian in 
the rotating frame is
              H_0 = - 0.5*ξ_a(a^†a^†aa),
where a is the annihilation operator for the qubit. Here 
the control Hamiltonian includes the usual symmetric and 
anti-symmetric terms 
    H_{sym} = p(t)(a + a^†),    H_{asym} = q(t)(a - a^†),
which come from the rotating frame approximation and hence 
we refer to these as "coupled" controls. For this
example we evolve the state forward a full period T=2π. The
parameters for this example are: 
                ω_a =  2π × 0.0       Grad/s,
                ξ_a =  2π × 2(0.1099) Grad/s.
We use the usual Bsplines (no carrier waves) in this
example.
==========================================================# 
using LinearAlgebra
using Plots
using FFTW
using DelimitedFiles
using Printf
using Ipopt
using Random

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

using Juqbox

N = 2

Nguard = 0
Ntot = N + Nguard
	
samplerate = 32 # default number of time steps per unit time
casename = "rabi" # base file name (used in optimize-once.jl)

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 5.0
xa = 2* 0.1099
rot_freq = [fa] # Rotational frequencies

# period of oscillation
Tperiod = 100.0 # ns # 2*pi

# duration
T = Tperiod # Tperiod/4

utarget = Matrix{ComplexF64}(I, Ntot, N)
vtarget = Matrix{ComplexF64}(I, Ntot, N)

# Rabi target (one period)

theta = pi/4 # phase angle
aOmega = pi/Tperiod
Omega = (cos(theta) + 1im*sin(theta)) * aOmega
println("Amplitude |Omega| = ", aOmega, " phase angle theta = ", theta)

# unitary target matrix
utarget[1,1] = cos(aOmega*T)
utarget[2,1] = -(sin(theta) + 1im*cos(theta))*sin(aOmega*T)
utarget[1,2] = (sin(theta) - 1im*cos(theta))*sin(aOmega*T)
utarget[2,2] = cos(aOmega*T)

omega1 = Juqbox.setup_rotmatrices([N], [Nguard], rot_freq)

# Compute Ra*utarget
rot1 = Diagonal(exp.(im*omega1*T))

# target in the rotating frame
vtarget = rot1*utarget

# target in lab frame
#vtarget = utarget

Nctrl = 1
Nfreq = 1 # number of carrier frequencies

Random.seed!(2456)
# initial parameter guess

# setup carrier frequencies
om = zeros(Nctrl,Nfreq)
# Note: same frequencies for each p(t) (x-drive) and q(t) (y-drive)
println("Carrier frequencies [GHz]: ", om[:,:]./(2*pi))

# setup drift Hamiltonian
number = Diagonal(collect(0:Ntot-1))

H0 = -0.5*(2*pi)*xa* (number*number - number)
println("Drift Hamiltonian/2*pi: ", H0)

# lowering matrix
amat = Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U) # standard lowering operator matrix
# raising matrix
adag = transpose(amat) # raising operator matrix

# max parameter amplitude
maxpar = 1.0*aOmega/Nfreq

# package the lowering and raising matrices together into an one-dimensional array of two-dimensional arrays
# Here we choose dense or sparse representation
# NOTE: the above eigenvalue calculation does not work with sparse arrays!

# sparse matrices
# Hsym_ops=[sparse(amat+adag)]
# Hanti_ops=[sparse(amat-adag)]
# H0 = sparse(H0)

# dense matrices
Hsym_ops=[Array(amat + adag)]
Hanti_ops=[Array(amat - adag)]
H0 = Array(H0)

# Estimate time step
Pmin = 80
nsteps = calculate_timestep(T, H0, Hsym_ops, Hanti_ops, [maxpar], Pmin)
println("Duration = ", T, " # time steps per min-period, P = ", Pmin, " # time steps: ", nsteps)

# Initial conditions
Ident = Matrix{Float64}(I, Ntot, Ntot)   
U0 = Ident[1:Ntot,1:N]

# setup the simulation parameters
params = Juqbox.objparams([N], [Nguard], T, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=rot_freq,
                          Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops)

# setup the initial parameter vector, either randomized or from file
startFromScratch = true # true
startFile = "rabi-pcof-opt-alpha-0.5.dat"

if startFromScratch
    # D1 smaller than 3 does not work
    D1 = 3 # Number of B-spline coefficients per frequency, sin/cos and real/imag
    nCoeff = 2*Nctrl*Nfreq*D1 # factor '2' is for sin/cos
    pcof0  = zeros(nCoeff)

    if Nfreq == 1 # analytical solution
        pcof0[1:D1]  .= aOmega*cos(theta) # real coefficients
        pcof0[D1+1:2*D1]  .= aOmega*sin(theta) # imag coefficients
        println("*** Starting from constant pcof with (Re/Im) amplitudes: ", aOmega*cos(theta), " ", aOmega*sin(theta) )
    else
        pcof0 = (rand(nCoeff) .- 0.5).*maxpar*0.1
        println("*** Starting from random pcof with amplitude ", maxpar*0.1)
    end
else
    # the data on the startfile must be consistent with the setup!
    # use if you want to have initial coefficients read from file
    pcof0 = vec(readdlm(startFile))
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Nctrl*Nfreq) # factor '2' is for sin/cos
    nCoeff = 2*Nctrl*Nfreq*D1 # just to be safe if the file doesn't contain the right number of elements
    println("*** Starting from B-spline coefficients in file: ", startFile)
end

# min and max coefficient values
minCoeff, maxCoeff = Juqbox.assign_thresholds(params, D1, [maxpar])

# For ipopt
maxIter = 150 # optional argument
lbfgsMax = 250 # optional argument

println("*** Settings ***")
println("System Hamiltonian coefficients: (fa, xa) =  ", fa, xa)
println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
println("Using B-spline basis functions with carrier wave, # freq = ", Nfreq)
println("Number of coefficients per spline = ", D1, " Total number of parameters = ", nCoeff)
println("Max parameter amplitudes: maxpar = ", maxpar)
println("Tikhonov coefficients: tik0 = ", params.tik0)

# Allocate all working arrays
wa = Juqbox.Working_Arrays(params, nCoeff)
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter=maxIter, lbfgsMax=lbfgsMax, startFromScratch=startFromScratch)

#uncomment to run the gradient checker for the initial pcof
#=
if @isdefined addOption
    addOption( prob, "derivative_test", "first-order"); # for testing the gradient
else
    AddIpoptStrOption( prob, "derivative_test", "first-order")
end
=#

println("Initial coefficient vector stored in 'pcof0'")

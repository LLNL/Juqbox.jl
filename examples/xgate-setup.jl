#==========================================================
X-gate for qubit #5 on IBM Casablanca
==========================================================# 
using LinearAlgebra
using Plots
using FFTW
using DelimitedFiles
using Printf
using Ipopt
using Random

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)
gr()
#pyplot()

using Juqbox

N = 2 # 3
Nguard = 1 # 0

Ntot = N + Nguard

casename = "x-gate" # base file name (used in optimize-once.jl)

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.9639697
xa = 0.3215826
rot_freq = [fa] # Rotational frequencies

# Duration tailored for the IBM IQ mixer
dt_IQ = 2.0/9
Npulses = 160 # Must be evenly divisable by 16
T = dt_IQ * Npulses # Nano sec

utarget = Matrix{ComplexF64}(I, Ntot, N)
vtarget = Matrix{ComplexF64}(I, Ntot, N)

# unitary target matrix
if Nguard == 0
    utarget[1,1] = 0.0
    utarget[2,1] = 1.0
    utarget[1,2] = 1.0
    utarget[2,2] = 0.0
elseif Nguard == 1
    utarget[1,1] = 0.0
    utarget[2,1] = 1.0
    utarget[3,1] = 0.0
    utarget[1,2] = 1.0
    utarget[2,2] = 0.0
    utarget[3,2] = 0.0
end

omega1 = Juqbox.setup_rotmatrices([N], [Nguard], rot_freq)

# Compute Ra*utarget
rot1 = Diagonal(exp.(im*omega1*T))

# target in the rotating frame
vtarget = rot1*utarget

# target in lab frame
#vtarget = utarget

# setup ansatz for control functions
Nctrl = 1
Nosc = Nctrl 
Nfreq = 1 # number of carrier frequencies

Random.seed!(2456)
# initial parameter guess

# setup carrier frequencies
om = zeros(Nosc,Nfreq)
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

# package the lowering and raising matrices together into an one-dimensional array of two-dimensional arrays
Hsym_ops=[Array(amat + adag)]
Hanti_ops=[Array(amat - adag)]
H0 = Array(H0)

# max parameter amplitude
maxpar = 4.0*(2*pi/T)/Nfreq # Bigger amplitude than for a constant pulse

maxamp = zeros(Nctrl, Nfreq)
maxamp .= maxpar # Same parameter bounds for all ctrl Hamiltonians and frequencies

# Estimate time step
Pmin = 80
nsteps = calculate_timestep(T, H0, Hsym_ops, Hanti_ops, [maxpar], Pmin)

println("Final time T = ", T, " # time steps per min-period, P = ", Pmin, " # time steps: ", nsteps)

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
    D1 = 5 # Number of B-spline coefficients per frequency, sin/cos
    nCoeff = 2*Nosc*Nfreq*D1 # factor '2' is for sin/cos
    pcof0  = maxpar*0.05.*ones(nCoeff)

    println("*** Starting from constant pcof with amplitude ", maxpar*0.05)
else
    # the data on the startfile must be consistent with the setup!
    # use if you want to have initial coefficients read from file
    pcof0 = vec(readdlm(startFile))
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Nosc*Nfreq) # factor '2' is for sin/cos
    nCoeff = 2*Nosc*Nfreq*D1 # just to be safe if the file doesn't contain the right number of elements
    println("*** Starting from B-spline coefficients in file: ", startFile)
end

# min and max coefficient values
useBarrier = true
minCoeff, maxCoeff = assign_thresholds_ctrl_freq(params, D1, maxamp)

zero_start_end!(params, D1, minCoeff, maxCoeff)

#-maxpar*ones(nCoeff);
#maxCoeff = maxpar*ones(nCoeff);

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
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter, lbfgsMax, startFromScratch)

# uncomment to run the gradient checker for the initial pcof
# addOption( prob, "derivative_test", "first-order"); # for testing the gradient

println("Initial coefficient vector stored in 'pcof0'")

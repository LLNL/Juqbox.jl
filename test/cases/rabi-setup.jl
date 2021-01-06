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
import Juqbox

verbose = false
N = 2

Nguard = 0
Ntot = N + Nguard
	
samplerate = 32 # default number of time steps per unit time
casename = "rabi" # base file name (used in optimize-once.jl)

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 0.0
xa = 2* 0.1099

# period of oscillation
Tperiod = 2*pi

# duration
T = Tperiod # Tperiod/4

utarget = Matrix{ComplexF64}(I, Ntot, N)
vtarget = Matrix{ComplexF64}(I, Ntot, N)

# Rabi target (one period)

theta = pi/2 # phase angle
aOmega = pi/Tperiod
Omega = (cos(theta) + 1im*sin(theta)) * aOmega
#println("Amplitude |Omega| = ", aOmega, " phase angle theta = ", theta)

# unitary target matrix
utarget[1,1] = cos(aOmega*T)
utarget[2,1] = -(sin(theta) + 1im*cos(theta))*sin(aOmega*T)
utarget[1,2] = (sin(theta) - 1im*cos(theta))*sin(aOmega*T)
utarget[2,2] = cos(aOmega*T)

omega1 = Juqbox.setup_rotmatrices([N], [Nguard], [fa])

# Compute Ra*utarget
rot1 = Diagonal(exp.(im*omega1*T))

# target in the rotating frame
vtarget = rot1*utarget

# setup ansatz for control functions
use_bcarrier = true # new Bcarrier allows a constant control function

Nctrl = 1
Nosc = Nctrl 
Nfreq = 1 # number of carrier frequencies

Random.seed!(2456)
# initial parameter guess

# setup carrier frequencies
om = zeros(Nosc,Nfreq)
# Note: same frequencies for each p(t) (x-drive) and q(t) (y-drive)
#println("Carrier frequencies [GHz]: ", om[:,:]./(2*pi))

# setup drift Hamiltonian
number = Diagonal(collect(0:Ntot-1))

H0 = -0.5*(2*pi)*xa* (number*number - number)
#println("Drift Hamiltonian/2*pi: ", H0)

# lowering matrix
amat = Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U) # standard lowering operator matrix
# raising matrix
adag = transpose(amat) # raising operator matrix

# max parameter amplitude
maxpar = 1.0*aOmega/Nfreq

K1 =  H0 + maxpar.*( amat +  amat') + 1im* maxpar.*(amat - amat')
lamb = eigvals(Array(K1))
maxeig = maximum(abs.(lamb)) 

# Estimate time step
Pmin = 80
samplerate1 = maxeig*Pmin/(2*pi)
nsteps = ceil(Int64, T*samplerate1)

#println("Max est. eigenvalue = ", maxeig, " Min period = ", 2*pi/maxeig, " # time steps per min-period, P = ", Pmin, " # time steps: ", nsteps)

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

# Weights in the W matrix for discouraging population of guarded states
wmatScale = 1.0
wmat = wmatScale .* Juqbox.wmatsetup([N], [Nguard])

# Initial conditions
Ident = Matrix{Float64}(I, Ntot, Ntot)   
U0 = Ident[1:Ntot,1:N]

# setup the simulation parameters
params = Juqbox.parameters([N], [Nguard], T, nsteps, U0, vtarget, om, H0, Hsym_ops, Hanti_ops)
params.saveConvHist = true
params.use_bcarrier = true 

# Quiet mode for testing
params.quiet = true

# setup the initial parameter vector, either randomized or from file
startFromScratch = true # true
startFile = "rabi-pcof-opt-alpha-0.5.dat"

if startFromScratch
    # D1 smaller than 3 does not work
    D1 = 3 # Number of B-spline coefficients per frequency, sin/cos and real/imag
    nCoeff = 2*Nosc*Nfreq*D1 # factor '2' is for sin/cos
    pcof0  = zeros(nCoeff)

    if Nfreq == 1 # analytical solution
        pcof0[1:D1]  .= aOmega*cos(theta) # real coefficients
        pcof0[D1+1:2*D1]  .= aOmega*sin(theta) # imag coefficients
#        println("*** Starting from constant pcof with (Re/Im) amplitudes: ", aOmega*cos(theta), " ", aOmega*sin(theta) )
    else
        pcof0 = (rand(nCoeff) .- 0.5).*maxpar*0.1
#        println("*** Starting from random pcof with amplitude ", maxpar*0.1)
    end
else
    # the data on the startfile must be consistent with the setup!
    # use if you want to have initial coefficients read from file
    pcof0 = vec(readdlm(startFile))
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Nosc*Nfreq) # factor '2' is for sin/cos
    nCoeff = 2*Nosc*Nfreq*D1 # just to be safe if the file doesn't contain the right number of elements
    #    println("*** Starting from B-spline coefficients in file: ", startFile)
end

# min and max coefficient values
useBarrier = true
minCoeff = -maxpar*ones(nCoeff);
maxCoeff = maxpar*ones(nCoeff);

# For ipopt
maxIter = 150 # optional argument
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
    println("Max parameter amplitudes: maxpar = ", maxpar)
    println("Tikhonov coefficients: tik0 = ", params.tik0)
end

# Estimate number of terms in Neumann series for time stepping (Default 3)
tol = eps(1.0); # machine precision
Juqbox.estimate_Neumann!(tol, T, params, [maxpar])

# Allocate all working arrays
wa = Juqbox.Working_Arrays(params, nCoeff)
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter, lbfgsMax)

# uncomment to run the gradient checker for the initial pcof
# addOption( prob, "derivative_test", "first-order"); # for testing the gradient

#println("Initial coefficient vector stored in 'pcof0'")

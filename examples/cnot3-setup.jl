#==========================================================
This routine initializes an optimization problem to recover 
a CNOT gate on a coupled 3-qubit system. In particular,
    Oscillator A: 2 energy levels, 2 guard states,
    Oscillator B: 2 energy levels, 2 guard states,
    Oscillator S: 1 energy level,  5 guard states,
The drift Hamiltonian in the rotating frame is
    H_0 = - 0.5*ξ_a(a^†a^†aa)
          - 0.5*ξ_b(b^†b^†bb)
          - 0.5*ξ_s(s^†s^†s)
          - ξ_{ab}(a^†ab^†b)
          - ξ_{as}(a^†as^†s)
          - ξ_{bs}(b^†bs^†s).
Here the control Hamiltonian in the rotating frame
includes the usual symmetric and anti-symmetric terms 
 H_{sym,1} = p_1(t)(a + a^†), H_{asym,1} = q_1(t)(a - a^†),
 H_{sym,2} = p_2(t)(b + b^†), H_{asym,2} = q_2(t)(b - b^†),
 H_{sym,3} = p_3(t)(s + s^†), H_{asym,3} = q_3(t)(s - s^†),
where a,b,s are the annihilation operators for each qubit.
The problem parameters for this example are,
            ω_a / 2π    = 4.10595      GHz,
            ξ_a / 2π     =  2.198e-02  GHz,
            ω_b / 2π    =  4.81526     GHz,
            ξ_b / 2π     =  2.252e-01  GHz,
            ω_s / 2π    =  7.8447       GHz,
            ξ_s / 2π     =  2.8299e-05 GHz,
            ξ_{ab} / 2π  =  1.0e-06     GHz,
            ξ_{as} / 2π  =  2.494e-03  GHz,
            ξ_{bs} / 2π  =  2.52445e-03 GHz.
We use Bsplines with carrier waves and 3 frequencies per 
oscillator:
    Oscillator A: 0, ξ_a, ξ_b
    Oscillator B: 0, ξ_a, ξ_b
    Oscillator S: 0, ξ_{as}, ξ_{bs}.
==========================================================# 
using LinearAlgebra
using Ipopt
using Base.Threads
using Random
using DelimitedFiles
using Printf
using FFTW
using Plots
gr()
#pyplot()
using SparseArrays
using FileIO

#include("Juqbox.jl") # using the consolidated Juqbox module
using Juqbox

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

function initial_cond(Ntot, N, Ne, Ng)
    Ident = Matrix{Float64}(I, Ntot, Ntot)
    U0 = Ident[1:Ntot,1:N] # initial guess
    #adjust initial guess
    if Ng[1]+Ng[2]+Ng[3] > 0
        Nt = Ne + Ng

        col = 0
        m = 0
        for k3 in 1:Nt[3]
            for k2 in 1:Nt[2]
                for k1 in 1:Nt[1]
                    m += 1
                    # is this a guard level?
                    guard = (k1 > Ne[1]) || (k2 > Ne[2]) || (k3 > Ne[3])
                    if !guard
                        col = col+1
                        U0[:,col] = Ident[:,m]
                    end # if ! guard
                end #for
            end # for
        end # for
        
    end # if
    return U0
end

fnt = Plots.font("Helvetica", 16)
lfnt = Plots.font("Helvetica", 10)
Plots.default(titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=lfnt)

Nosc = 3

Ne1 = 2 # essential energy levels per oscillator # AP: want Ne1=Ne2=2, but Ne3 = 1
Ne2 = 2
Ne3 = 1
Ng1 = 2 # Osc-1, number of guard states
Ng2 = 2 # Osc-2, number of guard states
Ng3 = 5 # Osc-3, number of guard states

Ne = [Ne1, Ne2, Ne3]
Ng = [Ng1, Ng2, Ng3]
Nt = Ne + Ng

N = Ne1*Ne2*Ne3; # Total number of nonpenalized energy levels
Ntot = Nt[1]*Nt[2]*Nt[3]
Nguard = Ntot - N # Total number of guard states

Tmax = 550.0 # 700.0

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.10595
fb = 4.81526  # official
fs = 7.8447 # storage   # official
rot_freq = [fa, fb, fs] # rotational frequencies
xa = 2 * 0.1099
xb = 2 * 0.1126 # official
xs = 0.002494^2/xa # 2.8298e-5 # official
xab = 1.0e-6 # 1e-6 official
xas = sqrt(xa*xs) # 2.494e-3 # official
xbs = sqrt(xb*xs) # 2.524e-3 # official

# Note: The ket psi = kji> = e_k kron e_j kron e_i.
# We order the elements in the vector psi such that i varies the fastest with i in [1,Nt1], j in [1,Nt2], , k in [1,Nt3]
# The matrix amat = I kron I kron a1 acts on alpha in psi = gamma kron beta kron alpha
# The matrix bmat = I kron a2 kron I acts on beta in psi = gamma kron beta kron alpha
# The matrix cmat = a3 kron I2 kron I1 acts on gamma in psi = gamma kron beta kron alpha

# construct the lowering and raising matricies: amat, bmat, cmat
# and the system Hamiltonian: H0

a1 = Array(Bidiagonal(zeros(Nt[1]),sqrt.(collect(1:Nt[1]-1)),:U))
a2 = Array(Bidiagonal(zeros(Nt[2]),sqrt.(collect(1:Nt[2]-1)),:U))
a3 = Array(Bidiagonal(zeros(Nt[3]),sqrt.(collect(1:Nt[3]-1)),:U))

I1 = Array{Float64, 2}(I, Nt[1], Nt[1])
I2 = Array{Float64, 2}(I, Nt[2], Nt[2])
I3 = Array{Float64, 2}(I, Nt[3], Nt[3])

# create the a, a^\dag, b and b^\dag vectors
amat = kron(I3, kron(I2, a1))
bmat = kron(I3, kron(a2, I1))
cmat = kron(a3, kron(I2, I1))

adag = Array(transpose(amat))
bdag = Array(transpose(bmat))
cdag = Array(transpose(cmat))

# number ops
num1 = Diagonal(collect(0:Nt[1]-1))
num2 = Diagonal(collect(0:Nt[2]-1))
num3 = Diagonal(collect(0:Nt[3]-1))

# number operators
Na = Diagonal(kron(I3, kron(I2, num1)) )
Nb = Diagonal(kron(I3, kron(num2, I1)) )
Nc = Diagonal(kron(num3, kron(I2, I1)) )

H0 = -2*pi*(xa/2*(Na*Na-Na) + xb/2*(Nb*Nb-Nb) + xs/2*(Nc*Nc-Nc) + xab*(Na*Nb) + xas*(Na*Nc) + xbs*(Nb*Nc))

# max coefficient amplitudes, rotating frame
amax = 0.05
bmax = 0.1
cmax = 0.1
maxpar = [amax, bmax, cmax] 

# package the lowering and raising matrices together into an one-dimensional array of two-dimensional arrays
# Here we choose dense or sparse representation
use_sparse = true

# dense matrices run faster, but take more memory
Hsym_ops=[Array(amat+adag), Array(bmat+bdag), Array(cmat+cdag)]
Hanti_ops=[Array(amat-adag), Array(bmat-bdag), Array(cmat - cdag)]
H0 = Array(H0)

# Estimate time step
Pmin = 40 # should be 20 or higher
nsteps = calculate_timestep(Tmax, H0, Hsym_ops, Hanti_ops, maxpar, Pmin)

println("Number of time steps = ", nsteps)

samplerate = 32 # for plotting (?)
kpar = 5 # test this component of the gradient

# type of basis function
use_bcarrier = true # false

Ncoupled = length(Hsym_ops)
if use_bcarrier
  Nfreq = 3 # 3 # number of carrier frequencies
else # regular B-splines
  Nfreq=1
  D1 = 400
  nCoeff = 2*Ncoupled*D1 # Total number of B-spline coeffs'. Must be divisible by 6
end

om = zeros(Ncoupled,Nfreq) # In the rotating frame all ctrl Hamiltonians have a zero resonace frequency

# initialize the carrier frequencies
if use_bcarrier 
  @assert(Nfreq == 1 || Nfreq == 2 || Nfreq == 3)
  if Nfreq==2
    # freq 2 and 3 for seg 3 and 6 (coupler)
    om[1:Ncoupled,2] .= -1.0*pi*xas # coupling freq for all (re/im)
  elseif Nfreq==3
    # fundamental resonance frequencies for the transmons 
      om[1:2,2] .= -2.0*pi*xa # carrier freq's for ctrl Hamiltonian 1 & 2
      om[1:2,3] .= -2.0*pi*xb # carrier freq's for ctrl Hamiltonian 1 & 2
      om[3,2] = -2.0*pi*xas # carrier freq 2 for ctrl Hamiltonian #3
      om[3,3] = -2.0*pi*xbs # carrier freq 2 for ctrl Hamiltonian #3
  end
end
println("Carrier frequencies 1st ctrl Hamiltonian [GHz]: ", om[1,:]./(2*pi))
println("Carrier frequencies 2nd ctrl Hamiltonian [GHz]: ", om[2,:]./(2*pi))
println("Carrier frequencies 3rd ctrl Hamiltonian [GHz]: ", om[3,:]./(2*pi))

casename = "cnot-storage"

# specify target gate
#target for CNOT (2 essential states, 1 g, 1 g, 7 guard states)

# The 2-osc CNOT gate is a 9x4 matrix and the identity for the 3rd oscillator with
# 7 guards is an 9x2 identity matrix

N2tot = Nt[1] * Nt[2]
N2 = Ne[1]*Ne[2]

utarget = zeros(ComplexF64, N2tot*Nt[3], N2*Ne[3])
# target for CNOT gate between oscillators 1 and 2
G2 = zeros(ComplexF64, N2tot, N2)
@assert(Ng[1] == 2 || Ng[1] == 1 || Ng[1] == 0)
@assert(Ne[1] == 2 && Ne[2] == 2)
if Ne[1]==2 && Ne[2] == 2
  if Ng[1] == 0
    G2[1,1] = 1.0
    G2[2,2] = 1.0
    G2[3,4] = 1.0
    G2[4,3] = 1.0
  elseif Ng[1] == 1
    G2[1,1] = 1.0
    G2[2,2] = 1.0
    G2[4,4] = 1.0
    G2[5,3] = 1.0
  elseif Ng[1] == 2
    G2[1,1] = 1.0
    G2[2,2] = 1.0
    G2[5,4] = 1.0
    G2[6,3] = 1.0
  end
end

Ident = Matrix{Float64}(I, Ntot, Ntot)
I3 = Matrix{Float64}(I, Nt[3], Ne[3]);
utarget = kron(I3, G2) # The CNOT is between oscillator 1 and 2. Identity for the 3rd oscillator

# rotation matrices
omega1, omega2, omega3 = Juqbox.setup_rotmatrices(Ne, Ng, rot_freq)

# Compute Ra*Rb*utarget
rot1 = Diagonal(exp.(im*omega1*Tmax))
rot2 = Diagonal(exp.(im*omega2*Tmax))
rot3 = Diagonal(exp.(im*omega3*Tmax))

# target in the rotating frame
vtarget = rot1*rot2*rot3*utarget

U0 = initial_cond(Ntot, N, Ne, Ng)

# NOTE: maxpar is now a vector with 3 elements: amax, bmax, cmax
params = Juqbox.objparams(Ne, Ng, Tmax, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=rot_freq,
                          Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, use_sparse=use_sparse)

Random.seed!(2456)

# setup the initial parameter vector, either randomized or from file
startFromScratch = false # true
startFile="drives/cnot3-pcof-opt.jld2"

if startFromScratch
    D1 = 15 # 20 # number of B-spline coeff per oscillator, freq, p/q
    nCoeff = 2*Ncoupled*Nfreq*D1 # Total number of parameters.
    pcof0 = amax*0.01 * rand(nCoeff)
    println("*** Starting from random pcof with amplitude ", amax*0.01)
else
    #  read initial coefficients from file
    pcof0 = read_pcof(startFile);
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Ncoupled*Nfreq)  # number of B-spline coeff per control function
    nCoeff = 2*Nosc*Nfreq*D1 # just to be safe if the file doesn't contain the right number of elements
    println("*** Starting from B-spline coefficients in file: ", startFile)
end

# min and max B-spline coefficient values
useBarrier = true
# minCoeff, maxCoeff = Juqbox.assign_thresholds(maxpar, Ncoupled, Nfreq, D1)
minCoeff, maxCoeff = Juqbox.assign_thresholds(params,D1,maxpar)

# for ipopt
maxIter = 0 #100 # 0 # 250 #50 # optional argument
lbfgsMax = 250 # optional argument

# output run information
println("*** Settings ***")
println("Frequencies: Alice = ", fa, " Bob = ", fb, " Storage = ", fs)
println("Anharmonic coefficients in the Hamiltonian: xa = ", xa, " xb = ", xb, " xs = ", xs)
println("Coupling coefficients in the Hamiltonian: xab = ", xab, " xas = ", xas, " xbs = ", xbs)
println("Essential states in osc = ", Ne, " Guard states in osc = ", Ng)
println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
println("Number of B-spline parameters per spline = ", D1, " Total number of parameters = ", nCoeff)
println("Max parameter amplitudes: maxpar = ", maxpar)
println("Tikhonov coefficients: tik0 (L2) = ", params.tik0)
if use_sparse
    println("Using a sparse representation of the Hamiltonian matrices")
else
    println("Using a dense representation of the Hamiltonian matrices")
end

wa = Juqbox.Working_Arrays(params,nCoeff)
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter, lbfgsMax, startFromScratch)

# uncomment to run the gradient checker for the initial pcof
#addOption( prob, "derivative_test", "first-order"); # for testing the gradient

# tmp: test call traceJuqbox()
#objv, objgrad, u_hist, infidelity = Juqbox.traceobjgrad(pcof0, params, true, true);

println("Initial coefficient vector stored in 'pcof0'")

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
using SparseArrays
using FileIO

#include("Juqbox.jl") # using the consolidated Juqbox module
using Juqbox

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

Ne1 = 2 # essential energy levels per oscillator # AP: want Ne1=Ne2=2, but Ne3 = 1
Ne2 = 2
Ne3 = 1

Ng1 = 2 # Osc-1, number of guard states
Ng2 = 2 # Osc-2, number of guard states
Ng3 = 3 # 5 # Osc-3, number of guard states

Ne = [Ne1, Ne2, Ne3]
Ng = [Ng1, Ng2, Ng3]
Nt = Ne + Ng

N = prod(Ne) # Total number of nonpenalized energy levels
Ntot = prod(Nt)
Nguard = Ntot - N # Total number of guard states

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.10595
fb = 4.81526  # official
fs = 7.8447 # storage   # official

rot_freq = [fa, fb, fs] # rotational frequencies
xa = -2 * 0.1099
xb = -2 * 0.1126 # official
xs = -0.002494^2/xa # 2.8298e-5 # official

couple_type = 1 # 1=cross-Kerr, 2=Jaynes-Cummings
xab = -1.0e-6 # 1e-6 official
xas = -sqrt(abs(xa*xs)) # 2.494e-3 # official
xbs = -sqrt(abs(xb*xs)) # 2.524e-3 # official

msb_order = true # false # true: original Juqbox, false: Quandary
println("Hamiltonian is setup for ", (msb_order ? "MSB" : "LSB"), " ordering")

# setup the Hamiltonian matrices
H0, Hsym_ops, Hanti_ops = hamiltonians_three_sys(Ness=Ne, Nguard=Ng, freq01=[fa, fb, fs], anharm=[xa, xb, xs], rot_freq=rot_freq, couple_coeff=[xab, xas, xbs], couple_type=couple_type, msb_order = msb_order)

amax = 0.1 # Approx max amplitude for each (p & q) ctrl function [rad/ns]

# calculate resonance frequencies & diagonalizing transformation
om, maxAmp, Utrans = get_resonances(Ness=Ne, Nguard=Ng, Hsys=H0, Hsym_ops=Hsym_ops, maxCtrl_pq=amax, msb_order=msb_order)

Nctrl = size(om, 1)
Nfreq = size(om, 2)
println("Nctrl = ", Nctrl, " Nfreq = ", Nfreq)

for q = 1:Nctrl
    println("Carrier frequencies in ctrl Hamiltonian # ", q, " [GHz]: ", om[q,:]./(2*pi))
    println("Amplitude bounds for p & q-functions in system # ", q, " [GHz]: ", maxAmp[q,:]./(2*pi))
end

use_diagonal_H0 = true # false
if use_diagonal_H0 # transformation to diagonalize the system Hamiltonian
    transformHamiltonians!(H0, Hsym_ops, Hanti_ops, Utrans)
end

Tmax = 550.0 # 700.0
# Estimate time step
Pmin = 40 # should be 20 or higher
nsteps = calculate_timestep(Tmax, H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, maxCoupled=maxAmp, Pmin=Pmin)
println("Duration T = ", T, " Number of time steps = ", nsteps)

# initialize the carrier frequencies
# @assert(Nfreq == 1 || Nfreq == 2 || Nfreq == 3)
# if Nfreq==2
#     om[1,2] = -2.0*pi*xa # carrier freq for ctrl Hamiltonian 1
#     om[2,2] = -2.0*pi*xb # carrier freq for ctrl Hamiltonian 2
#     om[3,2] = -2.0*pi*sqrt(xas*xbs) # carrier freq for ctrl Hamiltonian #3
# elseif Nfreq==3
#     # fundamental resonance frequencies for the transmons 
#     om[1:2,2] .= -2.0*pi*xa # carrier freq's for ctrl Hamiltonian 1 & 2
#     om[1:2,3] .= -2.0*pi*xb # carrier freq's for ctrl Hamiltonian 1 & 2
#     om[3,2] = -2.0*pi*xas # carrier freq 2 for ctrl Hamiltonian #3
#     om[3,3] = -2.0*pi*xbs # carrier freq 2 for ctrl Hamiltonian #3
# end

# println("Carrier frequencies 1st ctrl Hamiltonian [GHz]: ", om[1,:]./(2*pi))
# println("Carrier frequencies 2nd ctrl Hamiltonian [GHz]: ", om[2,:]./(2*pi))
# println("Carrier frequencies 3rd ctrl Hamiltonian [GHz]: ", om[3,:]./(2*pi))

# casename = "cnot-storage"

# target for CNOT gate between oscillators 1 and 2
gate_cnot = zeros(ComplexF64, 4, 4)
gate_cnot[1,1] = 1.0
gate_cnot[2,2] = 1.0
gate_cnot[3,4] = 1.0
gate_cnot[4,3] = 1.0

if Ne[3] == 1
    Utarg = gate_cnot
else
    Ident3 = Array{Float64, 2}(I, Ne[3], Ne[3])
    Utarg = kron(Ident3, gate_cnot) # assumes msb_order=true
end

# U0 has size Ntot x Ness. Each of the Ness columns has one non-zero element, which is 1.
U0 = initial_cond(Ne, Ng, msb_order)
utarget = U0 * Utarg # Initial basis with guard levels

# create a linear solver object
linear_solver = Juqbox.lsolver_object(solver=Juqbox.JACOBI_SOLVER,max_iter=100,tol=1e-12,nrhs=N)

# Here we choose dense or sparse representation
use_sparse = true
# use_sparse = false

# NOTE: maxpar is now a vector with 3 elements: amax, bmax, cmax
params = Juqbox.objparams(Ne, Ng, Tmax, nsteps, Uinit=U0, Utarget=utarget, Cfreq=om, Rfreq=rot_freq, Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, use_sparse=use_sparse, msb_order = msb_order)

Random.seed!(2456)

# setup the initial parameter vector, either randomized or from file
startFromScratch = true # false
startFile="drives/cnot3-pcof-opt.jld2"

if startFromScratch
    D1 = 50 # 15 # 20 # number of B-spline coeff per oscillator, freq, p/q
    nCoeff = 2*Nctrl*Nfreq*D1 # Total number of parameters.
    maxrand = amax*0.01
    pcof0 = init_control(params, maxrand=maxrand, nCoeff=nCoeff, seed=2456)
else
    #  read initial coefficients from file
    pcof0 = read_pcof(startFile);
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Nctrl*Nfreq)  # number of B-spline coeff per control function
    @assert nCoeff == 2*Nctrl*Nfreq*D1 "Inconsistent lengths of pcof from file and Nctrl, Nfreq, D1"
    nCoeff = 2*Nctrl*Nfreq*D1 
    println("*** Starting from B-spline coefficients in file: ", startFile)
end

println("*** Settings ***")
println("Number of coefficients per spline = ", D1, " Total number of control parameters = ", length(pcof0))
println("Tikhonov coefficients: tik0 = ", params.tik0)
println()
println("Problem setup (Hamiltonian, carrier freq's, time-stepper, etc) is stored in 'params' object")
println("Initial coefficient vector is stored in 'pcof0' vector")
println("Max control amplitudes is stored in 'maxAmp' vector")

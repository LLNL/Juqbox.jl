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
using FFTW
using DelimitedFiles
using Printf
using Ipopt
using Random

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

using Juqbox # quantum control module

Ne = [3] # Number of essential energy levels
Ng = [2] # Number of extra guard levels
Ntot = prod(Ne + Ng) # Total number of energy levels

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = [3.445779438]
xa = [-0.208343564]
rot_freq = copy(fa)

# form the Hamiltonian matrices
H0, Hsym_ops, Hanti_ops = hamiltonians_one_sys(Ness=Ne, Nguard=Ng, freq01=fa, anharm=xa, rot_freq=rot_freq)

maxctrl = 0.001*2*pi * 3.0 #  15.0 MHz (approx) max amplitude for each (p & q) control function

# calculate resonance frequencies
om, Nfreq, Utrans = get_resonances(Ness=Ne, Nguard=Ng, Hsys=H0, Hsym_ops=Hsym_ops)

Nctrl = length(om)

println("Nctrl = ", Nctrl, " Nfreq = ", Nfreq)

maxAmp = maxctrl*ones(Nctrl) # Note: Here we only have one control Hamiltonian

for q = 1:Nctrl
     println("Carrier frequencies in ctrl Hamiltonian # ", q, " [GHz]: ", om[q]./(2*pi))
     # println("Amplitude bounds for p & q-functions in system # ", q, " [GHz]: ", maxAmp[q,:]./(2*pi))
 end

T = 250.0 # Duration of gate
# Estimate time step
nsteps = calculate_timestep(T, H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, maxCoupled=maxAmp, Pmin=40)
println("Duration T = ", T, " # time steps: ", nsteps)

# CNOT target
gate_swap02 =  zeros(ComplexF64, Ne[1], Ne[1])
gate_swap02[1,3] = 1.0
gate_swap02[2,2] = 1.0
gate_swap02[3,1] = 1.0

# Initial basis with guard levels
U0 = initial_cond(Ne, Ng)
utarget = U0 * gate_swap02 # Add zero rows for the guard levels

# create a linear solver object
linear_solver = Juqbox.lsolver_object(solver=Juqbox.JACOBI_SOLVER, max_iter=100, tol=1e-12, nrhs=prod(Ne))

# number of B-splines per ctrl/freq/real-imag
D1 = 78
NfreqTot = sum(Nfreq)
nCoeff = 2*D1*NfreqTot

maxrand = 0.01*maxctrl/Nfreq[1]  # amplitude of the random control vector
pcof0 = init_control(Nctrl=Nctrl, Nfreq=Nfreq, maxrand=maxrand, nCoeff=nCoeff, seed=12456)

params = Juqbox.objparams(Ne, Ng, T, nsteps, Uinit=U0, Utarget=utarget, Cfreq=om, Rfreq=rot_freq, Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, linear_solver=linear_solver, nCoeff=length(pcof0))

params.tik0 = 1e-2

println("*** Settings ***")
println("Number of coefficients per spline = ", D1, " Total number of control parameters = ", length(pcof0))
println("Tikhonov coefficients: tik0 = ", params.tik0)
println()
println("Problem setup (Hamiltonian, carrier freq's, time-stepper, etc) is stored in 'params' object")
println("Initial coefficient vector is stored in 'pcof0' vector")
println("Max control amplitudes is stored in 'maxAmp' vector")
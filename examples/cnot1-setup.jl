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

N = 4 # Number of essential energy levels
Nguard = 2 # Number of extra guard levels
Ntot = N + Nguard # Total number of energy levels

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.10336
xa = -0.2198

# form the Hamiltonian matrices
H0, Hsym_ops, Hanti_ops, rot_freq = hamiltonians_one_sys(Ness=[N], Nguard=[Nguard], freq01=fa, anharm=xa, rotfreq=fa)

# calculate resonance frequencies
om = get_resonances(Ness=[N], Nguard=[Nguard], Hsys=H0, Hsym_ops=Hsym_ops)
Nctrl = size(om, 1)
Nfreq = size(om, 2)

println("Nctrl = ", Nctrl, " Nfreq = ", Nfreq)

maxctrl = 0.001*2*pi * 8.5 #  8.5 MHz

T = 100.0 # Duration of gate
# Estimate time step
nsteps = calculate_timestep(T, H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, maxCop=[maxctrl], Pmin=40)
println("Duration T = ", T, " # time steps: ", nsteps)

# Initial basis with guard levels
U0 = initial_cond([N], [Nguard])

# CNOT target
gate_cnot =  zeros(ComplexF64, N, N)
gate_cnot[1,1] = 1.0
gate_cnot[2,2] = 1.0
gate_cnot[3,4] = 1.0
gate_cnot[4,3] = 1.0

# Initial basis with guard levels
U0 = initial_cond([N], [Nguard])
utarget = U0 * gate_cnot # Add zero rows for the guard levels

# create a linear solver object
linear_solver = Juqbox.lsolver_object(solver=Juqbox.JACOBI_SOLVER,max_iter=100,tol=1e-12,nrhs=N)

params = Juqbox.objparams([N], [Nguard], T, nsteps, Uinit=U0, Utarget=utarget, Cfreq=om, Rfreq=rot_freq, Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, linear_solver=linear_solver)

# number of B-splines per ctrl/freq/real-imag
D1 = 10
nCoeff = 2*D1*Nctrl*Nfreq

maxrand = 0.05*maxctrl/Nfreq  # amplitude of the random control vector
pcof0 = init_control(params, maxrand=maxrand, nCoeff=nCoeff, seed=2345)

# same ctrl threshold for all frequencies
maxAmp = maxctrl/Nfreq .* ones(Nfreq)

println("*** Settings ***")
println("Number of coefficients per spline = ", D1, " Total number of control parameters = ", length(pcof0))
println("Tikhonov coefficients: tik0 = ", params.tik0)
println()
println("Problem setup (Hamiltonian, carrier freq's, time-stepper, etc) is stored in 'params' object")
println("Initial coefficient vector is stored in 'pcof0' vector")
println("Max control amplitudes is stored in 'maxAmp' vector")

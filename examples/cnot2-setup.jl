#==========================================================
This routine initializes an optimization problem to recover 
a CNOT gate on a coupled 2-qubit system with 2 energy 
levels on each oscillator (with 1 guard state on one and 
2 guard states on the other). The drift Hamiltonian in 
the rotating frame is
    H_0 = - 0.5*ξ_a(a^†a^†aa) 
          - 0.5*ξ_b(b^†b^†bb) 
          - ξ_{ab}(a^†ab^†b),
where a,b are the annihilation operators for each qubit.
Here the control Hamiltonian in the rotating frame
includes the usual symmetric and anti-symmetric terms 
H_{sym,1} = p_1(t)(a + a^†),  H_{asym,1} = q_1(t)(a - a^†),
H_{sym,2} = p_2(t)(b + b^†),  H_{asym,2} = q_2(t)(b - b^†).
The problem parameters for this example are,
            ω_a    =  2π × 4.10595   Grad/s,
            ξ_a    =  2π × 2(0.1099) Grad/s,
            ω_b    =  2π × 4.81526   Grad/s,
            ξ_b    =  2π × 2(0.1126) Grad/s,
            ξ_{ab} =  2π × 0.1       Grad/s,
We use Bsplines with carrier waves with frequencies
0, ξ_a, 2ξ_a Grad/s for each oscillator.
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

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

using Juqbox # quantum control module

Ne1 = 2 # essential energy levels per oscillator 
Ne2 = 2
Ng1 = 2 # 0 # Osc-1, number of guard states
Ng2 = 2 # 0 # Osc-2, number of guard states

Ne = [Ne1, Ne2]
Ng = [Ng1, Ng2]
Nt = Ne + Ng

N = prod(Ne) # Total number of nonpenalized energy levels
Ntot = prod(Nt) # Total number of all energy levels
Nguard = Ntot - N # total number of guard levels

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.10595    # official
fb = 4.81526   # official
favg = 0.5*(fa+fb)
rot_freq = [favg, favg]

#rot_freq = [favg, favg] # rotational frequencies
x1 = -2* 0.1099  # official
x2 = -2* 0.1126   # official
x12 = [-0.1] # Artificially large to allow fast coupling. Actual value: 1e-6 
couple_type = 1 # cross-Kerr
msb_order = false # true: original Juqbox, false: Quandary
println("Hamiltonian is setup for ", (msb_order ? "MSB" : "LSB"), " ordering")

# setup the Hamiltonian matrices
H0, Hsym_ops, Hanti_ops = hamiltonians_two_sys(Ness=Ne, Nguard=Ng, freq01=[fa, fb], anharm=[x1, x2], rot_freq=rot_freq, couple_coeff=x12, couple_type=couple_type, msb_order=msb_order)

# max coefficients, rotating frame
maxctrl = 2*pi * 0.015 # 15 MHz max amplitude for each (p & q) ctrl function

# calculate resonance frequencies & diagonalizing transformation
om, Nfreq, Utrans = get_resonances(Ness=Ne, Nguard=Ng, Hsys=H0, Hsym_ops=Hsym_ops, msb_order=msb_order)

Nctrl = length(om)

println("Nctrl = ", Nctrl, " Nfreq = ", Nfreq)

maxAmp = maxctrl*ones(Nctrl)

for q = 1:Nctrl
    println("Carrier frequencies in ctrl Hamiltonian # ", q, " [GHz]: ", om[q,:]./(2*pi))
    #println("Amplitude bounds for p & q-functions in system # ", q, " [GHz]: ", maxAmp[q,:]./(2*pi))
end

use_diagonal_H0 = true # false
if use_diagonal_H0 # transformation to diagonalize the system Hamiltonian
    transformHamiltonians!(H0, Hsym_ops, Hanti_ops, Utrans)
end

Tmax = 50.0 # Duration of gate
# estimate time step
Pmin = 40 # should be 20 or higher
nsteps = calculate_timestep(Tmax, H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, maxCoupled=maxAmp, Pmin=Pmin)
println("Duration T = ", Tmax, " Number of time steps = ", nsteps)

# CNOT target for the essential levels
gate_cnot =  zeros(ComplexF64, N, N)
ctrlQubit = 2 # or 2
@assert(ctrlQubit == 1 || ctrlQubit == 2)
if ctrlQubit == 1
    gate_cnot[1,1] = 1.0
    gate_cnot[2,2] = 1.0
    gate_cnot[3,4] = 1.0
    gate_cnot[4,3] = 1.0
elseif ctrlQubit == 2
    gate_cnot[1,1] = 1.0
    gate_cnot[2,4] = 1.0
    gate_cnot[3,3] = 1.0
    gate_cnot[4,2] = 1.0
end

# Initial basis with guard levels
U0 = initial_cond(Ne, Ng, msb_order)
utarget = U0 * gate_cnot

# create a linear solver object
linear_solver = Juqbox.lsolver_object(solver=Juqbox.JACOBI_SOLVER,max_iter=100,tol=1e-12,nrhs=N)

# Here we choose dense or sparse representation
# use_sparse = true
use_sparse = false

# dimensions for the parameter vector
D1 = 10 # number of B-spline coeff per oscillator, freq and sin/cos
nCoeff = 2*D1*sum(Nfreq) # Total number of parameters.

maxrand = 0.01*maxctrl/Nfreq[1]  # amplitude of the random control vector. Here Nfreq[1]=Nfreq[2]
pcof0 = init_control(Nctrl=Nctrl, Nfreq=Nfreq, maxrand=maxrand, nCoeff=nCoeff, seed=2456)

params = Juqbox.objparams(Ne, Ng, Tmax, nsteps, Uinit=U0, Utarget=utarget, Cfreq=om, Rfreq=rot_freq, Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, nCoeff=length(pcof0), linear_solver=linear_solver, use_sparse=use_sparse, msb_order=msb_order)

println("*** Settings ***")
println("Number of coefficients per spline = ", D1, " Total number of control parameters = ", length(pcof0))
println("Tikhonov coefficients: tik0 = ", params.tik0)
println()
println("Problem setup (Hamiltonian, carrier freq's, time-stepper, etc) is stored in 'params' object")
println("Initial coefficient vector is stored in 'pcof0' vector")
println("Max control amplitudes is stored in 'maxAmp' vector")
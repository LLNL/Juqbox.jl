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
#using Plots
#pyplot()
#using Base.Threads
#using FFTW
using DelimitedFiles
using Printf
#using Ipopt
using Random
using SparseArrays

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

#import Juqbox

verbose = false


function orig_wmatsetup(Ne::Array{Int64,1}, Ng::Array{Int64,1})
    Nt = Ne + Ng
    Ndim = length(Ne)
    @assert(Ndim == 1 || Ndim == 2 || Ndim ==3)
    
    Ntot = prod(Nt)
    w = zeros(Ntot)
    coeff = 1.0

    # reset temp variables
    temp = zeros(length(Ne))

    if sum(Ng) > 0
        nForb = 0 # number of states with the highest index in at least one dimension

        if Ndim == 1
            fact = 0.1
            for q in 0:Ng[1]-1
                w[Ntot-q] = fact^q
            end
            nForb = 1
            coeff = 1.0
        elseif Ndim == 2
            fact = 1e-3 # for more emphasis on the "forbidden" states. Old value: 0.1
            q = 0 # element in the array 'w'

            for i2 = 1:Nt[2]
                for i1 = 1:Nt[1]
                    q += 1
                    # initialize temp variables
                    temp[1] = 0.0
                    temp[2] = 0.0
                    if i1 <= Ne[1] && i2 <= Ne[2]
                        w[q] = 0.0
                    else
                        # determine and assign the largest penalty
                        if i1 > Ne[1]   #only included if at a guard level
                            temp[1] = fact^(Nt[1]-i1)
                        end
                        if i2 > Ne[2]   #only included if at a guard level
                            temp[2] = fact^(Nt[2]-i2)
                        end

                        if i1 == Nt[1] || i2 == Nt[2]
                            nForb += 1 
                        end

                        forbFact=1.0

                        # additional weighting (ad hoc)
                        # if i1 == Nt1 && i2<=Ne2 
                        #   forbFact=100
                        # end
                        # if i2 == Nt2 && i1<=Ne1 
                        #   forbFact=100
                        # end

                        w[q] = forbFact*maximum(temp)
          
                    end # if guard level
                end # for i1
            end # for i2

            # normalize by the number of entries with w=1
            coeff = 10.0/nForb # was 1/nForb
        elseif Ndim == 3
            fact = 1e-3 #  0.1 # for more emphasis on the "forbidden" states. Old value: 0.1
            nForb = 0 # number of states with the highest index in at least one dimension
            q = 0
            for i3 = 1:Nt[3]
                for i2 = 1:Nt[2]
                    for i1 = 1:Nt[1]
                        q += 1
                        # initialize temp variables
                        temp1 = 0.0
                        temp2 = 0.0
                        temp3 = 0.0
                        if i1 <= Ne[1] && i2 <= Ne[2] && i3 <= Ne[3]
                            w[q] = 0.0
                        else
                            # determine and assign the largest penalty
                            if i1 > Ne[1]   #only included if at a guard level
#                                temp1 = (Nt[1] - Ne[1]) * fact^(Nt[1]-i1)
                                temp1 = fact^(Nt[1]-i1)
                            end
                            if i2 > Ne[2]   #only included if at a guard level
#                                temp2 = (Nt[2] - Ne[2]) *fact^(Nt[2]-i2)
                                temp2 = fact^(Nt[2]-i2)
                            end
                            if i3 > Ne[3]   #only included if at a guard level
#                                temp3 = (Nt[3] - Ne[3]) *fact^(Nt[3]-i3)
                                temp3 = fact^(Nt[3]-i3)
                            end

                            forbFact=1.0
                            # additional weighting (ad hoc)
                            # if i1 == Nt[1] && i2<=Ne[2] && i3<=Ne3
                            #   forbFact=100
                            # end
                            # if i2 == Nt[2] && i1<=Ne[1] && i3<=Ne3
                            #   forbFact=100
                            # end
                            if i3 == Nt[3] && i1<=Ne[1] && i2<=Ne[2]
                               forbFact=100
                            end

                            w[q] = forbFact*max(temp1, temp2, temp3)

                            if i1 == Nt[1] || i2 == Nt[2] || i3 == Nt[3]
                                nForb += 1
                            end

                        end # if
                    end # for
                end # for
            end # for

            # normalize by the number of entries with w=1
            coeff = 10.0/nForb # was 1/nForb
        end # if ndim == 3

        # println("wmatsetup: Number of forbidden states = ", nForb, " scaling coeff = ", coeff)
    end # if sum(Ng) > 0
    wmat = coeff .* Diagonal(w) # turn vector into diagonal matrix
    return wmat
end

Nosc = 2 # number of coupled oscillators

Ne1 = 2 # essential energy levels per oscillator 
Ne2 = 2
Ng1 = 1 # 0 # Osc-1, number of guard states
Ng2 = 2 # 0 # Osc-2, number of guard states

Ne = [Ne1, Ne2]
Ng = [Ng1, Ng2]

N = Ne1*Ne2; # Total number of nonpenalized energy levels
Ntot = (Ne1+Ng1)*(Ne2+Ng2)
Nguard = Ntot - N # total number of guard states

Nt1 = Ne1 + Ng1
Nt2 = Ne2 + Ng2

Tmax = 100.0 # Duration of gate

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.10595    # official
fb = 4.81526   # official
rot_freq = [fa, fb]
x1 = 2* 0.1099  # official
x2 = 2* 0.1126   # official
x12 = 0.1 # Artificially large to allow coupling. Actual value: 1e-6 
  
# construct the lowering and raising matricies: amat, bmat, cmat
# and the system Hamiltonian: H0
#

# Note: The ket psi = ji> = e_j kron e_i.
# We order the elements in the vector psi such that i varies the fastest with i in [1,Nt1] and j in [1,Nt2]
# The matrix amat = I kron a1 acts on alpha in psi = beta kron alpha
# The matrix bmat = a2 kron I acts on beta in psi = beta kron alpha
a1 = Array(Bidiagonal(zeros(Nt1),sqrt.(collect(1:Nt1-1)),:U))
a2 = Array(Bidiagonal(zeros(Nt2),sqrt.(collect(1:Nt2-1)),:U))

I1 = Array{Float64, 2}(I, Nt1, Nt1)
I2 = Array{Float64, 2}(I, Nt2, Nt2)

# create the a, a^\dag, b and b^\dag vectors
amat = kron(I2, a1)
bmat = kron(a2, I1)

adag = Array(transpose(amat))
bdag = Array(transpose(bmat))

# number ops
num1 = Diagonal(collect(0:Nt1-1))
num2 = Diagonal(collect(0:Nt2-1))

# number operators
N1 = Diagonal(kron(I2, num1) )
N2 = Diagonal(kron(num2, I1) )

# System Hamiltonian
H0 = -2*pi*( x1/2*(N1*N1-N1) + x2/2*(N2*N2-N2) + x12*(N1*N2) )

# max coefficients, rotating frame
amax = 0.02 # max amplitude ctrl func for Hamiltonian #1
bmax = 0.05 # max amplitude ctrl func for Hamiltonian #2
maxpar = [amax, bmax]

# estimate max magnitude of eigenvalue
K1 =  H0 +
    (amax.*(amat +  amat') + 1im*amax.*(amat -  amat') +
     bmax.*(bmat + bmat') + 1im*bmax.*(bmat - bmat'))
lamb = eigvals(K1)
maxeig = maximum(abs.(lamb))

# Estimate time step
Pmin = 40 # should be 20 or higher
samplerate1 = maxeig*Pmin/(2*pi)
nsteps = ceil(Int64,Tmax*samplerate1)

#println("Number of time steps = ", nsteps)

# package the lowering and raising matrices together into an one-dimensional array of two-dimensional arrays
# Here we choose dense or sparse representation
use_sparse = false

# dense matrices run faster, but take more memory
Hsym_ops=[Array(amat+adag), Array(bmat+bdag)]
Hanti_ops=[Array(amat-adag), Array(bmat-bdag)]
H0 = Array(H0)

use_bcarrier = true # Use carrier waves in the control pulses?

if use_bcarrier
    Nfreq = 2 # number of carrier frequencies
else
    Nfreq = 1
end

Ncoupled = length(Hsym_ops)
om = zeros(Ncoupled, Nfreq) # Allocate space for the carrier wave frequencies

if use_bcarrier
    @assert(Nfreq==1 || Nfreq==2 || Nfreq==3)
    if Nfreq == 2
        om[1:Ncoupled,2] .= -2.0*pi*x12 # coupling freq for both ctrl funcs (re/im)
    elseif Nfreq == 3
        om[1,2] = -2.0*pi*x1 # 1st ctrl, re
        om[2,2] = -2.0*pi*x2 # 2nd ctrl, re
        om[1:Ncoupled,3] .= -2.0*pi*x12 # coupling freq for both ctrl funcs (re/im)
    end
end
#println("Carrier frequencies 1st ctrl Hamiltonian [GHz]: ", om[1,:]./(2*pi))
#println("Carrier frequencies 2nd ctrl Hamiltonian [GHz]: ", om[2,:]./(2*pi))

# specify target gate
# target for CNOT gate N=2, Ng = 1 coupled
utarget = zeros(ComplexF64, Ntot, N)
@assert(Ng1 == 0 || Ng1 == 1 || Ng1 == 2)
if Ng1 == 0
    utarget[1,1] = 1.0
    utarget[2,2] = 1.0
    utarget[3,4] = 1.0
    utarget[4,3] = 1.0
elseif Ng1 == 1
    utarget[1,1] = 1.0
    utarget[2,2] = 1.0
    utarget[4,4] = 1.0
    utarget[5,3] = 1.0
elseif Ng1 == 2
    utarget[1,1] = 1.0
    utarget[2,2] = 1.0
    utarget[5,4] = 1.0
    utarget[6,3] = 1.0
end

# rotation matrices
omega1, omega2 = Juqbox.setup_rotmatrices(Ne, Ng, rot_freq)

# Compute Ra*Rb*utarget
rot1 = Diagonal(exp.(im*omega1*Tmax))
rot2 = Diagonal(exp.(im*omega2*Tmax))

# target in the rotating frame
vtarget = rot1*rot2*utarget

U0 = Juqbox.initial_cond(Ne, Ng)

#Build jacobi solver
linear_solver = Juqbox.lsolver_object(solver=Juqbox.JACOBI_SOLVER,iter=100,tol=1e-15)

# assemble problem description for the optimization
params = Juqbox.objparams(Ne, Ng, Tmax, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=rot_freq,
                          Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, use_sparse=use_sparse,linear_solver=linear_solver)

# overwrite default wmat with the old style
params.wmat_real =  orig_wmatsetup(Ne, Ng)

# Quiet mode for testing
params.quiet = true

# test
# custom = 0
# guardlev = Juqbox.identify_guard_levels(params, custom)
# forbiddenlev = Juqbox.identify_forbidden_levels(params, custom)
# end

# initial parameter guess
startFromScratch = false # false
startFile = "cases/cnot2-jacobi.dat"

# dimensions for the parameter vector
D1 = 10 # number of B-spline coeff per oscillator, freq and sin/cos

nCoeff = 2*Ncoupled*Nfreq*D1 # Total number of parameters.

Random.seed!(2456)
if startFromScratch
  pcof0 = amax*0.01 * rand(nCoeff)
#  println("*** Starting from pcof with random amplitude ", amax*0.01)
else
  pcof0 = vec(readdlm(startFile))
#  println("*** Starting from B-spline coefficients in file: ", startFile)

  nCoeff = length(pcof0)
  D1 = div(nCoeff, 2*Ncoupled*Nfreq)  # number of B-spline coeff per control function
end

samplerate = 32 # for plotting
casename = "cnot2" # for constructing file names

# min and max B-spline coefficient values
useBarrier = true
minCoeff, maxCoeff = Juqbox.assign_thresholds(params,D1,maxpar)
#println("Number of min coeff: ", length(minCoeff), "Max Coeff: ", length(maxCoeff))

maxIter = 150 # 0 #250 #50 # optional argument
lbfgsMax = 250 # optional argument

if verbose
    println("*** Settings ***")
    # output run information
    println("Frequencies: fa = ", fa, " fb = ", fb)
    println("Coefficients in the Hamiltonian: x1 = ", x1, " x2 = ", x2, " x12 = ", x12)
    println("Essential states in osc = ", [Ne1, Ne2], " Guard states in osc = ", [Ng1, Ng2])
    println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
    println("Number of B-spline parameters per spline = ", D1, " Total number of parameters = ", nCoeff)
    println("Max parameter amplitudes: maxpar = ", maxpar)
    println("Tikhonov regularization tik0 (L2) = ", params.tik0)
    if use_sparse
        println("Using a sparse representation of the Hamiltonian matrices")
    else
        println("Using a dense representation of the Hamiltonian matrices")
    end
end

# Estimate number of terms in Neumann series for time stepping (Default 3)
tol = eps(1.0); # machine precision
Juqbox.estimate_Neumann!(tol, params, maxpar)

# Allocate all working arrays
wa = Juqbox.Working_Arrays(params, nCoeff)
prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter=maxIter, lbfgsMax=lbfgsMax)

# uncomment to run the gradient checker for the initial pcof
# if @isdefined addOption
#     addOption( prob, "derivative_test", "first-order"); # for testing the gradient
# else
#     AddIpoptStrOption( prob, "derivative_test", "first-order")
# end
# uncomment to run the gradient checker for the initial pcof
# if @isdefined addOption
#     addOption( prob, "print_level", 0); # for testing the gradient
# else
#     AddIpoptIntOption( prob, "print_level", 0)
# end

#println("Initial coefficient vector stored in 'pcof0'")


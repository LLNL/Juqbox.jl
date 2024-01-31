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
f01 = [3.448646] # [3.445779438]
xi = [-0.208396]; # [-0.208343564]
xi12 = zeros(0)
couple_type = 1 # only to define all args
rot_freq = copy(f01)

T = 300.0 # 250.0 # Duration of gate

dtau = 10.0 # 3.33
D1 = ceil(Int64,T/dtau) + 2
D1 = max(D1,5)

# Points per shortest period
Pmin = 40 # 60 # 40 # 80

# bounds on the ctrl vector elements (rot frame)
# This number is divided by Nfreq
maxctrl_MHz = 20.0 # 30.0 # ?

# Internal ordering of the basis for the state vector
# msb_order = true # | i3, i2, i1> = |i3> \kron |i2> \kron |i1> 
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)

initctrl_MHz = 1.0 # amplitude for initial random guess of B-spline coeff's
rand_seed = 5432 # 2345

# Tikhonov coeff
tikCoeff = 1e-2 # 1.0 # 0.1

cw_amp_thres = 1e-7 # Include cross-resonance
cw_prox_thres = 1e-2 # 1e-2 # 1e-3

use_carrier_waves = true # false

zeroCtrlBC = true # Impose zero boundary conditions for each B-spline segemnt

# SWAP02 target
gate_swap02 =  zeros(ComplexF64, Ne[1], Ne[1])
gate_swap02[1,3] = 1.0
gate_swap02[2,2] = 1.0
gate_swap02[3,1] = 1.0

fidType = 2 # fidType = 1 for Frobenius norm^2, or fidType = 2 for Infidelity, or fidType = 3 for infid^2

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, gate_swap02, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, initctrl_MHz=initctrl_MHz, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, use_carrier_waves=use_carrier_waves, fidType=fidType)
#true)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-5 # better than 0.99999 fidelity
maxIter = 200 # only 12 are needed

println("Setup complete")

println("Calling run_optimizer")
pcof = run_optimizer(params, pcof0, maxAmp, maxIter=maxIter)
pl = plot_results(params, pcof)

println("IPOpt iteration completed")

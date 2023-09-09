### Set up a test problem using one of the standard Hamiltonian models
using Juqbox
using Printf
using Plots

# Four qubit parameter setup

Ne = [2, 2, 2, 2] # Number of essential energy levels
Ng = [1, 1, 1, 1] # Number of extra guard levels

# Qubits 0,1,4,7 from IBM Guadelope
f01 = [5.113535725239690, 5.160748676896810, 5.3534263008668200, 5.202793466283090] # 0-1 transition freq's

nSys = length(Ne)
#xi = -0.34*ones(Int64, nSys) # same anharmonicity for all oscillators = f12 - f01
xi = [-0.33523244411102300, -0.31826942892534200, -0.33130748866056200, -0.31732888617110500] # anharmonicity = f12 - f01

couple_type = 2 # Jaynes-Cummings coupling coefficients
# T-intersection coupling
xi12 = 5e-3 * [1.0, 0.0, 0.0, 1.0, 0.0, 1.0] # order: x12, x13, x14, x23, x24, x34

# Setup frequency of rotations in computational frame
favg = sum(f01)/nSys
rot_freq = favg * ones(nSys)

# Set the initial duration
T = 1500.0 # 1000.0
# Number of coefficients per spline
D1 = 75 # 50 # 50

# Points per shortest period
Pmin = 40 # 60 # 40 # 80

# Tikhonov coeff
tikCoeff = 1e-2 # 1.0 # 0.1

# bounds on the ctrl vector elements (rot frame)
# This number is divided by Nfreq
maxctrl_MHz = 100.0 # 30.0 # ?

# Internal ordering of the basis for the state vector
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)

# Set Quandary's executable if needed (develop branch)
quandary_exec= "./main"   # set to "" for Juqbox, or "./main" for Quandary
ncores = 16 # prod(Ne)

init_amp_frac = 0.5
rand_seed = 5432

cw_amp_thres = 4e-2 # For an identity gate use 0.5
cw_prox_thres = 1e-3 # To only get a single frequency in each group

wmatScale = 1.0

# assign the target gate
target_gate = get_swap_1d_gate(4)

verbose = verbose

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, init_amp_frac=init_amp_frac, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, verbose=verbose)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-3 # better than 99.9% fidelity

println("Setup complete")

using Juqbox

## Four qubits, each with 2 essential + 2 guard levels

Ne = [2, 2, 2, 2] # Number of essential energy levels
Ng = [0, 0, 0, 0] # Number of extra guard levels

f01 = [5.18, 5.12, 5.06, 5.00] # 0-1 transition freq's

nSys = length(Ne)
xi = -0.34*ones(Int64, nSys) # same anharmonicity for all oscillators = f12 - f01

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
maxctrl_MHz = 5*45.0 # 30.0 # ?

# Internal ordering of the basis for the state vector
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)

# Set Quandary's executable if needed (develop branch)
quandary_exec= "./main"   # set to "" for Juqbox, or "./main" for Quandary
ncores = 16 # prod(Ne)

init_amp_frac = 0.9/5 # 2e-1 # 8e-3
rand_seed = 5432

cw_amp_thres = 6e-2 # For an identity gate use 0.5
cw_prox_thres = 1e-3 # To only get a single frequency in each group
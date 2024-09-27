# Four qubits, each with 2 essential + 0 guard levels

Ne = [2, 2, 2, 2] # Number of essential energy levels
Ng = [0, 0, 0, 0] # Number of extra guard levels

# 0-1 transition freq's
#
f01 = [4.93, 4.98, 5.04, 5.11] # Unequal spacing in frequencies

nSys = length(Ne)
xi = -0.34*ones(Int64, nSys) # same anharmonicity for all oscillators = f12 - f01

couple_type = 2 # Dipole-dipole coupling coefficients
# T-intersection coupling
xi12 = 5e-3 * [1.0, 0.0, 0.0, 1.0, 0.0, 1.0] # order: x12, x13, x14, x23, x24, x34
#xi12 = 1e-6 * [1.0, 0.0, 0.0, 1.0, 0.0, 1.0] # order: x12, x13, x14, x23, x24, x34

# Setup frequency of rotations in computational frame
favg = sum(f01)/nSys
rot_freq = favg * ones(nSys)
#rot_freq = zeros(nSys) # Lab frame

# Set the initial duration
T = 600.0 # 1500.0 # 1000.0

# Number of coefficients per spline
# D1 = 75 # 50 # 50
dtau = 3.33
D1 = ceil(Int64,T/dtau) + 2
D1 = max(D1,5)

# Points per shortest period
Pmin = 40 # 60 # 40 # 80

# Tikhonov coeff
tikCoeff = 1e-2 # 1.0 # 0.1

# bounds on the ctrl vector elements (rot frame)
# This number is divided by Nfreq
maxctrl_MHz = 30.0 # 100.0 # 30.0 # ?

# Internal ordering of the basis for the state vector
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)

# Set Quandary's executable if needed (develop branch)
quandary_exec= "./main"   # set to "" for Juqbox, or "./main" for Quandary
ncores = 16 # prod(Ne)

# Note: to get Hessian at ctrl = 0, set rand_amp_frac = 0.0
initctrl_MHz = 10.0 # 0.1 # 0.5 # 0.3
rand_seed = 5432

cw_amp_thres = 5e-2
cw_prox_thres = 1e-3

wmatScale = 1.0

use_carrier_waves = true
zeroCtrlBC = true


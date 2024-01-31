# Three qubit test case

Ne = [2,2,2] # Number of essential energy levels
Ng = [0,0,0] # Number of extra guard levels

f01 = [4.94, 5.0, 5.06] # 0-1 transition freq's

xi = [-0.34, -0.34, -0.34] # anharmonicity = f12 - f01

couple_type = 2 # Dipole-dipole coupling coefficients
xi12 = 5e-3 * [1.0, 0.0, 1.0] # order: x12, x13, x23

# Setup frequency of rotations in computational frame
nSys = length(Ne)
favg = sum(f01)/nSys
rot_freq = favg * ones(nSys) # Rotating frame
#rot_freq = zeros(nSys) # Lab frame

# Set the initial duration
T = 300.0 # 600.0 200.0 # was 500.0. 
# Number of coefficients per spline
# D1 = 26
dtau = 3.33
D1 = ceil(Int64,T/dtau) + 2
D1 = max(D1,5)

# Points per shortest period
Pmin = 40 # 60 # 40 # 80

# Tikhonov coeff
tikCoeff = 1e-2 # 1.0 # 0.1

# bounds on the ctrl vector elements (rot frame)
# This number is divided by Nfreq
maxctrl_MHz = 30.0 #100.0 # 30.0 # 100.0 # ?

# Internal ordering of the basis for the state vector
# msb_order = true # | i3, i2, i1> = |i3> \kron |i2> \kron |i1> 
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)

# Set Quandary's executable if needed (develop branch)
quandary_exec= "./main"   # set to "" for Juqbox, or "./main" for Quandary
ncores = 4 # prod(Ne)

# Note: to get Hessian at ctrl = 0, set rand_amp_frac = 0.0
initctrl_MHz = 10.0 # 0.1 # 0.5 # 0.3
rand_seed = 5432

cw_amp_thres = 6e-2
cw_prox_thres = 1e-3

wmatScale = 1.0

use_carrier_waves = true
zeroCtrlBC = true

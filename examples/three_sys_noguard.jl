using Juqbox

# Three qubit test case

Ne = [2,2,2] # Number of essential energy levels
Ng = [0,0,0] # Number of extra guard levels

f01 = [5.18, 5.12, 5.06] # 0-1 transition freq's

xi = [-0.34, -0.34, -0.34] # anharmonicity = f12 - f01

couple_type = 2 # Jaynes-Cummings coupling coefficients
xi12 = 5e-3 * [1.0, 0.0, 1.0] # order: x12, x13, x23

# Setup frequency of rotations in computational frame
nSys = length(Ne)
favg = sum(f01)/nSys
rot_freq = favg * ones(nSys)

# Set the initial duration
T = 500.0
# Number of coefficients per spline
D1 = 26

# Points per shortest period
Pmin = 40 # 60 # 40 # 80

# Tikhonov coeff
tikCoeff = 1e-2 # 1.0 # 0.1

# bounds on the ctrl vector elements (rot frame)
# This number is divided by Nfreq
maxctrl_MHz = 40.0 # 30.0 # 100.0 # ?

# Internal ordering of the basis for the state vector
# msb_order = true # | i3, i2, i1> = |i3> \kron |i2> \kron |i1> 
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)

# Set Quandary's executable if needed (develop branch)
quandary_exec= "./main"   # set to "" for Juqbox, or "./main" for Quandary
ncores = 4 # prod(Ne)

#Initialize first ctrl vector with random numbers, with amplitude rand_amp
# Note: to get Hessian at ctrl = 0, set rand_amp = 0.0
init_amp_frac = 0.1 # 0.9/5 # 0.9 
rand_seed = 5432

cw_amp_thres = 6e-2
cw_prox_thres = 1e-3

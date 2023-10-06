using Juqbox

# Two qubit test case, modified to use different anharmonicities 

Ne = [2,2] # Number of essential energy levels
Ng = [0,0] # Number of extra guard levels

# IBM Jakarta (simplified)
f01 = [5.12, 5.06] 

nSys = length(Ne)
xi = -0.34 * ones(Int64, nSys)

couple_type = 2 # Jaynes-Cummings coupling coefficients
xi12 = [5.0e-3]

# Setup frequency of rotations in computational frame
favg = sum(f01)/nSys
rot_freq = favg * ones(nSys)

# Set the initial duration
T = 250.0
# Number of coefficients per spline
# D1 = 26
dtau = 10.0 # 3.33
D1 = ceil(Int64,T/dtau) + 2
D1 = max(D1,5)

# Points per shortest period
Pmin = 40 # 60 # 40 # 80

# bounds on the ctrl vector elements (rot frame)
# This number is divided by Nfreq
maxctrl_MHz = 100.0 # 30.0 # ?

# Internal ordering of the basis for the state vector
# msb_order = true # | i3, i2, i1> = |i3> \kron |i2> \kron |i1> 
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)

# Set Quandary's executable if needed (develop branch)
quandary_exec= "./main"   # set to "" for Juqbox, or "./main" for Quandary
ncores = 4 # prod(Ne)

#Initialize first ctrl vector with random numbers, with amplitude rand_amp
# Note: neeed by setup_std_model()
initctrl_MHz = 10.0 # amplitude for initial random guess of B-spline coeff's
rand_seed = 5432 # 2345

# Tikhonov coeff
tikCoeff = 1e-2 # 1.0 # 0.1

cw_amp_thres = 1e-7 # Include cross-resonance
cw_prox_thres = 1e-2 # 1e-2 # 1e-3

use_carrier_waves = true # false

zeroCtrlBC = true # Impose zero boundary conditions for each B-spline segemnt
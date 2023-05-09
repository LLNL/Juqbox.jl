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
T = 200.0
# Number of coefficients per spline
D1 = 26

# Points per shortest period
Pmin = 40 # 60 # 40 # 80

# Tikhonov coeff
tikCoeff = 1e-2 # 1.0 # 0.1

# bounds on the ctrl vector elements (rot frame)
# This number is divided by Nfreq
maxctrl_MHz = 30.0 # ?

# Internal ordering of the basis for the state vector
# msb_order = true # | i3, i2, i1> = |i3> \kron |i2> \kron |i1> 
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)

# Set Quandary's executable if needed (develop branch)
quandary_exec= "./main"   # set to "" for Juqbox, or "./main" for Quandary
ncores = 4 # prod(Ne)

#Initialize first ctrl vector with random numbers, with amplitude rand_amp
# Note: Not used by do_continuation_target(), but neeed by setup_std_model()
init_amp_frac = 0.9/5 # Fraction of max ctrl amplitude for initial random guess
rand_seed = 2345

cw_amp_thres = 1e-7 # Include cross-resonance
cw_prox_thres = 1e-2 # 1e-3
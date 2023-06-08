# Three qubit test case

Ne = [2,2,2] # Number of essential energy levels
Ng = [0,0,0] # Number of extra guard levels

#f01 = [5.18, 5.12, 5.06] # 0-1 transition freq's
f01 = [5.113535725239690, 5.160748676896810, 5.3534263008668200] # 0-1 transition freq's

# xi = [-0.34, -0.34, -0.34] # anharmonicity = f12 - f01
xi = [-0.33523244411102300, -0.31826942892534200, -0.33130748866056200] # anharmonicity = f12 - f01

couple_type = 2 # Jaynes-Cummings coupling coefficients
xi12 = 5e-3 * [1.0, 0.0, 1.0] # order: x12, x13, x23

# Setup frequency of rotations in computational frame
nSys = length(Ne)
favg = sum(f01)/nSys
rot_freq = favg * ones(nSys)

# Set the initial duration
T = 500.0
# Number of coefficients per spline
#D1 = 26
dtau = 3.33
D1 = ceil(Int64,T/dtau) + 2
D1 = max(D1,5)

# Points per shortest period
Pmin = 40 # 60 # 40 # 80

# Tikhonov coeff
tikCoeff = 1e-2 # 1.0 # 0.1

# Box constraints on the ctrl vector elements (rot frame)
# This number is divided by Nfreq
maxctrl_MHz = 100.0 # 30.0 # 100.0 # ?


# Internal ordering of the basis for the state vector
# msb_order = true # | i3, i2, i1> = |i3> \kron |i2> \kron |i1> 
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)

# Set Quandary's executable if needed (develop branch)
quandary_exec= "./main"   # set to "" for Juqbox, or "./main" for Quandary
ncores = 4 # prod(Ne)

# Note: to get Hessian at ctrl = 0, set rand_amp_frac = 0.0
#Initialize ctrl vector with random numbers, with amplitude rand_amp_frac*maxctrl_MHz
init_amp_frac = 0.5
rand_seed = 5432

cw_amp_thres = 5e-2
cw_prox_thres = 1e-3

wmatScale = 1.0

use_carrier_waves = true

# assign the target gate
target_gate = get_swap_1d_gate(3)

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, init_amp_frac=init_amp_frac, rand_seed=rand_seed, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, wmatScale=wmatScale, use_carrier_waves=use_carrier_waves)
# cw_prox_thres=5e-3, , cw_amp_thres=6e-2
params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-4 # better than 99.99% fidelity

println("Setup complete")

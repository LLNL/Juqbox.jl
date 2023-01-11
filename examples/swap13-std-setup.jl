### Set up a test problem using one of the standard Hamiltonian models
using Juqbox

## Three qubits, each with 2 essential + 2 guard levels

Ne = [2,2,2] # Number of essential energy levels
Ng = [2,2,2] # Number of extra guard levels
f01 = [4.10595, 4.81526, 5.0]
favg = sum(f01)/2
rot_freq = copy(f01)
xi = [-0.22, -0.225, -0.23]
xi12 = [-0.1, 0.0, -0.12] # order: x12, x13, x23
couple_type = 1 # 1: cross-kerr (negative coeffs), 2: Jaynes-Cummings (positive coeffs)
# assign the target gate
N = prod(Ne)
target_gate = get_swap_1d_gate(3) # get_swap_13_1()
# Set the initial duration#
T = 200.0
# Number of coefficients per spline
D1 = 20
# bounds on the ctrl vector elements (rot frame)
# This number is divided by Nfreq
maxctrl_MHz = 50.0 

# Amplitude of the random initial ctrl vector
rand_amp = 8e-3

# Internal ordering of the basis for the state vector
# msb_order = true # | i3, i2, i1> = |i3> \kron |i2> \kron |i1> 
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)

# Points per shortest period
Pmin = 40 # 60 # 40 # 80

# Set Quandary's executable if needed (develop branch)
quandary_exec= "./main"   # set to "" for Juqbox, or "./main" for Quandary
ncores = 4 # prod(Ne)

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, rand_amp=rand_amp, Pmin=Pmin)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];


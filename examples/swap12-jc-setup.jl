### Set up a test problem using one of the standard Hamiltonian models
using Juqbox

## Two qubits

Ne = [2,2] # Number of essential energy levels
Ng = [2,2] # Number of extra guard levels
f01 = [4.914, 5.114]
favg = sum(f01)/2
rot_freq = [favg, favg]
xi = [-0.33, -0.23]
xi12 = [5.0e-3]
couple_type = 2 # 1: cross-kerr, 2: Jaynes-Cummings

# assign the target gate
N = prod(Ne)
target_gate = get_swap_1d_gate(2) # get_H4_gate() #get_swap_1d_gate(2) # 

# Set the initial duration
T = 150.0 # 150.0 # 130.0 #121.2 # 250.0 # 200.0 # 300.0 # 100.0 # 10.0 #70.0 
dtau = 10.0 # 10.0/3
D1 = ceil(Int64, T/dtau) # 20 # 

# bounds on the ctrl vector elements (rot frame)
# This number is divided by Nfreq
maxctrl_MHz = 30.0 # 40.0 # 100.0 # 30.0 # 80.0

# Amplitude of the random initial ctrl vector
init_amp_frac = 0.5

# Internal ordering of the basis for the state vector
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)
# true: | i3, i2, i1> = |i3> \kron |i2> \kron |i1>

# Points per shortest period
Pmin = 40 # 60 # 40 # 80

cw_amp_thres = 1e-7 # Include cross-resonance
cw_prox_thres = 1e-2 # 1e-2 # 1e-3

# Set Quandary's executable if needed (develop branch)
quandary_exec= "./main"   # set to "" for Juqbox, or "./main" for Quandary
ncores = prod(Ne)

verbose = true

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, init_amp_frac=init_amp_frac, Pmin=Pmin, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, verbose=verbose) # use_eigenbasis=true is experimental

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];

params.traceInfidelityThreshold = 1e-3 # better than 99.9% fidelity

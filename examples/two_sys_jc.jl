Ne = [2,2] # Number of essential energy levels
Ng = [2,2] # Number of extra guard levels
f01 = [4.0, 4.5]
favg = sum(f01)/2
#rot_freq = [favg, favg]
rot_freq = 4.5 * ones(2)

xi = [-0.22, -0.225]

#couple_type = 1 # 1: cross-kerr, 2: Jaynes-Cummings
#xi12 = [-0.1] # [3.8e-3]

couple_type = 2 # 1: cross-kerr, 2: Jaynes-Cummings
xi12 = [5.0e-3]
# xi12 = [0.0]

# assign the target gate
N = prod(Ne)
target_gate = get_swap_1d_gate(2) # get_H4_gate() #get_swap_1d_gate(2) # 
# Set the initial duration
T = 150.0 # 150.0 # 130.0 #121.2 # 250.0 # 200.0 # 300.0 # 100.0 # 10.0 #70.0 
# dtau = 10.0/3
D1 = 20 # ceil(Int64, T/dtau) # 16 # 25 # 10
# bounds on the ctrl vector elements (rot frame)
# This number is divided by Nfreq
maxctrl_MHz = 50.0 # 40.0 # 100.0 # 30.0 # 80.0

# Amplitude of the random initial ctrl vector
rand_amp = 8e-3

# Internal ordering of the basis for the state vector
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)
# true: | i3, i2, i1> = |i3> \kron |i2> \kron |i1>

# Points per shortest period
Pmin = 40 # 60 # 40 # 80

# Set Quandary's executable if needed (develop branch)
quandary_exec= "./main"   # set to "" for Juqbox, or "./main" for Quandary
ncores = prod(Ne)

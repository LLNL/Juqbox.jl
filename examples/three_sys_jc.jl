using Juqbox

# Three qubit test case, modified to use different anharmonicities 

Ne = [2,2,2] # Number of essential energy levels
Ng = [2,2,2] # Number of extra guard levels
#Ng = [1,1,1] # Number of extra guard levels

f01 = [4.0, 4.5, 5.0] # 0-1 transition freq's

xi = [-0.22, -0.225, -0.23] # anharmonicity = f12 - f01

couple_type = 2 # Jaynes-Cummings coupling coefficients
xi12 = [2.5e-2, 0.0, 3.0e-2] # order: x12, x13, x23
#xi12 = [0.0e-3, 0.0, 5.0e-3] # order: x12, x13, x23
#xi12 = [0.0, 0.0, 0.0] # order: x12, x13, x23


# Setup frequency of rotations in computational frame
nSys = length(Ne)
favg = sum(f01)/nSys
rot_freq = favg * ones(nSys)

# Set the initial duration
T = 200.0
# Number of coefficients per spline
D1 = 20

# Points per shortest period
Pmin = 40 # 60 # 40 # 80

# Tikhonov coeff
tikCoeff = 1e-2 # 0.1 # 1.0 # 0.1

# bounds on the ctrl vector elements (rot frame)
# This number is divided by Nfreq
maxctrl_MHz = 50.0 # ?

# Internal ordering of the basis for the state vector
# msb_order = true # | i3, i2, i1> = |i3> \kron |i2> \kron |i1> 
msb_order = false # | i1, i2, i3> = |i1> \kron |i2> \kron |i3> (compatible with quandary)

# Set Quandary's executable if needed (develop branch)
quandary_exec= "./main"   # set to "" for Juqbox, or "./main" for Quandary
ncores = 4 # prod(Ne)

# assign the target gate
target_gate = get_swap_1d_gate(nSys) # get_Hd_gate(8) # get_CpNOT(2) #  get_swap_13_1()

#Initialize first ctrl vector with random numbers, with amplitude rand_amp
# Note: to get Hessian at ctrl = 0, set rand_amp = 0.0
rand_amp = 1e-2 # 5e-3 # 1e-4 # 2e-3 # 1e-2 # 1e-2




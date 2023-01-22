### Set up a test problem using one of the standard Hamiltonian models
using Juqbox
using Printf

Base.show(io::IO, f::Float64) = @printf(io, "%10.3e", f)

## Three qubits, each with 2 essential + 2 guard levels
#include("three_sys_xkerr.jl") # cross-Kerr
include("three_sys_jc.jl") # Jaynes-Cummings

retval = setup_std_model(Ne, Ng, f01, xi, xi12, couple_type, rot_freq, T, D1, target_gate, maxctrl_MHz=maxctrl_MHz, msb_order=msb_order, rand_amp=rand_amp, Pmin=Pmin, cw_prox_thres=5e-3, cw_amp_thres=6e-2)

params = retval[1]
pcof0 = retval[2]
maxAmp = retval[3];


#==========================================================
This routine initializes an optimization problem to recover 
a CNOT gate on a single qudit with 4 energy levels (and 2 
guard states). The drift Hamiltonian in the rotating frame 
is
              H_0 = - 0.5*ξ_a(a^†a^†aa),
where a is the annihilation operator for the qudit. 
Here the control Hamiltonian includes the usual symmetric 
and anti-symmetric terms 
     H_{sym} = p(t)(a + a^†),    H_{asym} = q(t)(a - a^†)
which come from the rotating frame approximation and hence 
we refer to these as "coupled" controls.
The problem parameters are:
                ω_a =  2π × 4.10595   Grad/s,
                ξ_a =  2π × 2(0.1099) Grad/s.
We use Bsplines with carrier waves with frequencies
0, ξ_a, 2ξ_a Grad/s.
==========================================================# 
using LinearAlgebra
using Plots
pyplot()
using FFTW
using DelimitedFiles
using Printf
using Ipopt
using Random

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

using Juqbox # quantum control module

Nosc = 1 # Number of coupled sub-systems = oscillators
N = 4 # Number of essential energy levels

Nguard = 2 # Number of extra guard levels
Ntot = N + Nguard # Total number of energy levels
	
T = 100.0 # Duration of gate. 

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.10336
xa = 0.2198
rot_freq = [fa] # Used to calculate the lab frame ctrl function

# setup drift Hamiltonian
number = Diagonal(collect(0:Ntot-1))

H0 = -0.5*(2*pi)*xa* (number*number - number) # xa is in GHz

# lowering matrix 
amat = Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U) # standard lowering matrix
adag = Array(transpose(amat));
Hsym_ops=[Array(amat + adag)]
Hanti_ops=[Array(amat - adag)]
H0 = Array(H0)

# Estimate time step
maxctrl = 0.001*2*pi * 8.5 #  9, 10.5, 12, 15 MHz

nsteps = calculate_timestep(T, H0, Hsym_ops, Hanti_ops, [maxctrl])
println("# time steps: ", nsteps)

use_bcarrier = true # Use carrier waves in the control pulses?

if use_bcarrier
    Nfreq = 3 # number of carrier frequencies
else
    Nfreq = 1
end

Ncoupled = length(Hsym_ops) # Here, Ncoupled = 1
om = zeros(Ncoupled,Nfreq)

if use_bcarrier
    om[1:Ncoupled,2] .= -2.0*pi *xa       # Note negative sign
    om[1:Ncoupled,3] .= -2.0*pi* 2.0*xa
end
println("Carrier frequencies [GHz]: ", om[1,:]./(2*pi))

maxamp = zeros(Nfreq)

if Nfreq >= 3
    const_fact = 0.45
    maxamp[1] = maxctrl*const_fact
    maxamp[2:Nfreq] .= maxctrl*(1.0-const_fact)/(Nfreq-1) # split the remainder equally
else
    # same threshold for all frequencies
    maxamp .= maxctrl/Nfreq
end

maxpar = maximum(maxamp)

Ident = Matrix{Float64}(I, Ntot, Ntot)   

# CNOT trarget
utarget = Ident[1:Ntot,1:N]
utarget[:,3] = Ident[:,4]
utarget[:,4] = Ident[:,3]

omega1 = Juqbox.setup_rotmatrices([N], [Nguard], [fa])

# Compute Ra*utarget
rot1 = Diagonal(exp.(im*omega1*T))

# target in the rotating frame
vtarget = rot1*utarget

# Initial conditions
U0 = Ident[1:Ntot,1:N]

params = Juqbox.objparams([N], [Nguard],
                          T, nsteps, Uinit=U0, Utarget=vtarget,
                          Cfreq=om, Rfreq=rot_freq,
                          Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops)

# initial parameter guess

# D1 smaller than 5 does not work
D1 = 10 # Number of B-spline coefficients per segment
nCoeff = 2*Ncoupled*Nfreq*D1 # factor '2' is for sin/cos

# Random.seed!(12456)

startFromScratch = true 
startFile="cnot-pcof-opt.dat"

# initial parameter guess
if startFromScratch
    pcof0 = maxpar*0.01 * rand(nCoeff)
    println("*** Starting from random pcof with amplitude ", maxpar*0.01)
else
    # use if you want to read the initial coefficients from file
    pcof0 = vec(readdlm(startFile))
    println("*** Starting from B-spline coefficients in file: ", startFile)
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Ncoupled*Nfreq)  # number of B-spline coeff per control function
end

# min and max coefficient values
useBarrier = true
minCoeff, maxCoeff = Juqbox.assign_thresholds_freq(maxamp, Ncoupled, Nfreq, D1)

samplerate = 32 # only used for plotting
casename = "cnot1" # base file name (used in optimize-once.jl)

maxIter = 200 # 0  # optional argument
lbfgsMax = 250 # optional argument
ipTol = 1e-5   # optional argument
acceptTol = ipTol # 1e-4 # acceptable tolerance 
acceptIter = 15

println("*** Settings ***")
println("System Hamiltonian coefficients [GHz]: (fa, xa) =  ", fa, xa)
println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
if use_bcarrier
  println("Using B-spline basis functions with carrier wave, # freq = ", Nfreq)
else
  println("Using regular B-spline basis functions")
end
println("Number of coefficients per spline = ", D1, " Total number of parameters = ", nCoeff)
for q=1:Nfreq
    println("Carrier frequency: ", om[q]/(2*pi), " GHz, max parameter amplitude = ", 1000*maxamp[q]/(2*pi), " MHz")
end
println("Tikhonov coefficients: tik0 = ", params.tik0)

# Estimate number of terms in Neumann series for time stepping (Default 3)
tol = eps(1.0); # machine precision
Juqbox.estimate_Neumann!(tol, params, [maxpar])

wa = Juqbox.Working_Arrays(params,nCoeff)

# prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter, lbfgsMax, startFromScratch)
# println("Initial coefficient vector stored in 'pcof0'")


pcof0 = maxpar*0.5*(rand(nCoeff).-0.5)
pbjfv, U, mfidelityrot = Juqbox.traceobjgrad(pcof0,params,wa,true,false)

W = U[:,:,end]
W = W[:]
ndata = 10 # 1e6
nin = length(pcof0)
nout = 2*length(W)

function generate_data(nin,nout,ndata,params,wa)

    input_data = zeros(nin,ndata)
    output_data = zeros(nout,ndata)
    nouthalf = Int64(nout/2)
    
    for i = 1:ndata
        pcof0 = maxpar*0.5*(rand(nin).-0.5)*10 #10 is magic number to increase dynamic range
        objfv, U, mfidelityrot = Juqbox.traceobjgrad(pcof0,params,wa,true,false) #changed verbose to false
        # U is the result of solving an ODE with initial data (identity matrix)
        # evolved up to some time T (defined above)
        # U has 3 dims: (i x j) x time (2D stacked over time)
        W = U[:,:,end]
        W = W[:] # Stacks everything (flatten)
        input_data[:,i] = pcof0
        output_data[1:nouthalf,i] = real.(W) # split real part of W
        output_data[nouthalf+1:nout,i] = imag.(W) # split imaginary part of W
    end
    return input_data, output_data
end

#in_data, out_data = generate_data(nin,nout,50000,params,wa)

#writedlm( "in_data_60_10.csv",  in_data, ',')
#writedlm( "out_data_48_10.csv",  out_data, ',')

using Flux


#model = Chain(Dense(nin, 10, relu),
#              Dense(10,10),Dense(10,10))
#model = Chain(model, Dense(10, nout))



function fit_model!(model)
    loss(x, y) = Flux.Losses.mse(model(x), y)

    opt = Descent()
    parameters = Flux.params(model)
    data = [(in_data, out_data)]
    
    loss(in_data,out_data)
    for epoch in 1:200
        Flux.train!(loss, parameters, data, opt)
    end

    return loss(in_data,out_data)
end

function fit_model_data!(model, in, out)
    # Fits model with given in and out data.
    loss(x, y) = Flux.Losses.mse(model(x), y)

    out_mod = copy(out)
    opt = Descent()
    parameters = Flux.params(model)
    data = [(in, out_mod)]
    println("START:")
    println(loss(in,out))
    for epoch in 1:200
        Flux.train!(loss, parameters, data, opt)
    end

    return loss(in,out)
end


#fit_model!(model)



#=
### ARCHITECTURE EXPERIMENT
# Tries to find the best network architecture by testing over many numbers of layers
# and neurons per layer.
max_layers = 20
max_neurons = 50

outs = zeros(max_layers,max_neurons)
for n_layers = 1:max_layers
    for n_neurons = 1:max_neurons
        model = Chain(Dense(nin, n_neurons, relu)) # Initialize input layer with correct dims
        for i = 1:n_layers # Initialize model with n_layers hidden layers and n_neurons neurons per layer
            model = Chain(model, Dense(n_neurons, n_neurons))
        end
        model = Chain(model, Dense(n_neurons, nout)) # Add output layer with correct dims

        outs[n_layers, n_neurons] = fit_model!(model)
    end
end
writedlm( "grid_search_output.csv",  outs, ',')
=#

n_in = 60
n_out = 48

in_data = readdlm("in_data_60_10.csv", ',', Float64, '\n')
out_data = readdlm("out_data_48_10.csv", ',', Float64, '\n')

outs = zeros(100)

println("***********")
println("STARTING SAMPLING EXPERIMENT")
println("***********")

for i = 1:100
    data_size = i*500
    model = Chain(Dense(n_in, 100, relu)) 
    # Initialize input layer with correct dims
    for j = 1:5 # 40 neurons, 5 layers
        model = Chain(model, Dense(100, 100))
    end
    model = Chain(model, Dense(100, n_out))

    in_new = in_data[:,1:data_size]
    out_new = out_data[:,1:data_size]

    outs[i] = fit_model_data!(model, in_new, out_new)
    println(outs[i])
 #outs = zeros(max_layers,max_neurons)
 # Add output layer with correct dims
end
writedlm( "data_size_search_large_scaling.csv",  outs, ',')
plot(1:100, outs, xlabel="Dataset size", xticks=0:1000:50000)



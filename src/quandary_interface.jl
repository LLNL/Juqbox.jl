using DelimitedFiles
using Printf

function write_Quandary_config_file(configfilename::String, Nt::Vector{Int64}, Ne::Vector{Int64}, T::Float64, nsteps::Int64, freq01::Vector{Float64}, rotfreq::Vector{Float64}, selfkerr::Vector{Float64}, couple_coeff::Vector{Float64}, couple_type::Int64, D1::Int64, carrierfreq::Vector{Vector{Float64}}, gatefilename::String, initialpcof_filename::String, optim_bounds::Vector{Float64}, inftol::Float64, maxiter::Int64, tik0::Float64, leakage_weights::Vector{Float64}, print_frequency_iter::Int64; runtype::String="optimization", gamma_dpdm::Float64=0.0, final_objective::Int64=1, gamma_energy::Float64=0.0, splines_real_imag::Bool = true, phase_scaling_factor::Float64=1.0)

    # final_objective = 1 uses the trace infidelity; 
    # final_objective = 2 uses the Frobenius norm squared
    # gamma_dpdm > 0 to suppress 2nd time-derivative of the population
    # gamma_dpdm = 0.1
    @assert(final_objective == 1 || final_objective == 2)
    
    mystring = "nlevels = " * (string(Nt)[2:end-1]) * "\n"
    mystring *= "nessential= " * (string(Ne)[2:end-1]) * "\n"
    mystring *= "ntime = " * string(nsteps) * "\n"
    mystring *= "dt = " * string(T/nsteps) * "\n"
    mystring *= "transfreq = " * (string(freq01)[2:end-1]) * "\n"
    mystring *= "rotfreq= " * (string(rotfreq)[2:end-1]) * "\n"
    mystring *= "selfkerr = " * (string(selfkerr)[2:end-1]) * "\n"
    if couple_type == 1
      mystring *= "crosskerr= " * (string(couple_coeff)[2:end-1]) * "\n"
      mystring *= "Jkl= 0.0\n"
    else
      mystring *= "crosskerr= 0.0\n" 
      mystring *= "Jkl= " * (string(couple_coeff)[2:end-1]) * "\n"
    end
    mystring *= "collapse_type=none\n"
    mystring *= "initialcondition=basis\n"
    
    # choose between having splines for both the real & imaginary parts, or only for the amplitude with a fixed phase
    if splines_real_imag
      for iosc in 1:length(Ne)
        mystring*= "control_segments" * string(iosc-1) * " = spline, "*string(D1) * "\n"
        mystring*= "control_initialization" * string(iosc-1) * " = file, ./" * string(initialpcof_filename) * "\n"
        mystring*= "control_bounds" * string(iosc-1) * " = " * string(optim_bounds[iosc]) * "\n"
        mystring *= "carrier_frequency" * string(iosc-1) * " = "
        omi = carrierfreq[iosc]
        for j in 1:length(omi)
            mystring *= string(omi[j]/(2*pi)) * ", "
        end
        mystring *= "\n"
      end
    else
      for iosc in 1:length(Ne)
        mystring*= "control_segments" * string(iosc-1) * " = spline_amplitude, " * string(D1) * ", " * string(phase_scaling_factor) * "\n"
        mystring*= "control_initialization" * string(iosc-1) * " = file, ./" * string(initialpcof_filename) * "\n"
        mystring*= "control_bounds" * string(iosc-1) * " = " * string(optim_bounds[iosc]) * "\n"
        mystring *= "carrier_frequency" * string(iosc-1) * " = "
        omi = carrierfreq[iosc]
        for j in 1:length(omi)
            mystring *= string(omi[j]/(2*pi)) * ", "
        end
        mystring *= "\n"
      end
    end

    mystring *= "optim_target = gate, fromfile, " * string(gatefilename) * "\n"
    if final_objective == 1
      mystring *= "optim_objective = Jtrace\n" 
    elseif final_objective == 2
      mystring *= "optim_objective = Jfrobenius\n"
    end
    mystring *= "gate_rot_freq = 0.0\n"
    mystring *= "optim_weights= 1.0\n"
    mystring *= "optim_atol= 1e-7\n"
    mystring *= "optim_rtol= 1e-8\n"
    #mystring *= "optim_ftol= 1e-4\n"
    mystring *= "optim_ftol= " * string(inftol) * "\n"
    mystring *= "optim_inftol= " * string(inftol)*"\n"
    mystring *= "optim_maxiter= "*string(maxiter)*"\n"
    mystring *= "optim_regul= "*string(tik0)*"\n"
    mystring *= "optim_penalty= 1.0\n" #* string(gamma1) *"\n" # gamma1
    mystring *= "optim_penalty_param= 0.0\n" # a (used with infidelity)
    ninitscale = prod(Ne)
    mystring *= "optim_leakage_weights= "*(string(leakage_weights.*ninitscale)[2:end-1])*"\n"
    mystring *= "optim_regul_dpdm= " * string(gamma_dpdm) * "\n" # gamma_dpdm
    mystring *= "optim_penalty_energy= " * string(gamma_energy) * "\n" # gamma_energy
    mystring *= "datadir= ./data_out\n"
    for iosc in 1:length(Ne)
      mystring *= "output"*string(iosc-1)*"=expectedEnergy, population, fullstate\n"
    end
    mystring *= "output_frequency = "*string(nsteps)*"\n"
    mystring *= "optim_monitor_frequency = "*string(print_frequency_iter)*"\n"
    mystring *= "runtype = "*runtype*"\n"
    mystring *= "usematfree = true\n"
    mystring *= "linearsolver_type = gmres\n"
    mystring *= "linearsolver_maxiter = 20\n"
    mystring *= "np_init = " *string(prod(Ne))*"\n"
   
    open(configfilename, "w") do io
      write(io, mystring)
    end;
  
    #println("Quandary config file: ", configfilename)
  end
  
  function get_Quandary_results(params::objparams, datadir::String, Nt::Vector{Int64}, Ne::Vector{Int64}; runtype::String="simulation")
    # Get pcof
    pcof = convert(Vector{Float64},readdlm(datadir*"/params.dat")[:,1])
  
    # Get Infidelity, norm of gradient and iterations taken
    optim_hist = readdlm(datadir*"/optim_history.dat")
    nOptimIter = size(optim_hist)[1] - 1
    
    objective_last = optim_hist[end,2]
    infid_last = optim_hist[end,6]
    tikhonov_last = optim_hist[end,7]
    penalty_last = optim_hist[end,9] # dpdm penalty
  
    # copy history arrays to the params structure for post processing
    params.saveConvHist = true
    params.objHist = copy(optim_hist[2:end,2])
    params.dualInfidelityHist = copy(optim_hist[2:end,3])
    params.primaryHist = copy(optim_hist[2:end,6])
    params.secondaryHist = copy(optim_hist[2:end,8]) # penalty = leakage
  
    #println("Quandary result: Infidelity=", infid_last)
  
    # Get last time-step unitary
    uT = zeros(ComplexF64, prod(Nt), prod(Ne))
  
    for i in 1:prod(Ne)
      # Read from file
      xre = readdlm(datadir*"/rho_Re.iinit"*lpad(string(i-1),4,"0")*".dat")[end,2:end]
      xim = readdlm(datadir*"/rho_Im.iinit"*lpad(string(i-1),4,"0")*".dat")[end,2:end]
      uT[:,i] = xre + im*xim 
    end
  
  
    grad = zeros(Float64, length(pcof))
    if (runtype=="gradient") # the grad.dat file is not created by the optimization mode
        # chop up the long vector into individual column vectors for the result
        grad = convert(Vector{Float64}, readdlm(datadir*"/grad.dat")[:,1])
        gradnorm = norm(grad)/sqrt(length(grad))
    end
  
    # make the return args similar to traceobjgrad()
    if runtype=="simulation"
      return infid_last+tikhonov_last+penalty_last, infid_last, penalty_last, uT
    elseif runtype == "gradient"
      return infid_last+tikhonov_last+penalty_last, grad, infid_last, penalty_last, 1.0 - infid_last
    else # "optimization"
      # similar to run_optimizer()
      return pcof, infid_last, penalty_last, tikhonov_last, params.objHist, nOptimIter
    end
  end

# Run Quandary
# runIdx can be 1=simulation, 2=gradient, or 3=optimization
"""
    qres = run_Quandary(params, pcof0, maxAmp; 
                        runIdx = 3, maxIter = 100, ncores = 1, quandary_exec = "./main", 
                        print_frequency_iter = 1, gamma_dpdm = 0.0, gamma_energy = 0.0, 
                        final_objective = 1, splines_real_imag::Bool = true, phase_scaling_factor::Float64=1.0)

Execute the Quandary solver in a sub-process to perform either a forward simulation, evaluate the gradient of the objective function, or optimize the control vector.
 
# Required arguments
- `params::objparams`: Object holding the optimization problem description
- `pcof0::Vector{Float64}:` Vector of length 2*D1*sum(Nfreq) holding initial control vector
- `maxAmp::Vector{Float64}`: Approximate bounds on the control amplitude [MHz] for the p(t) and q(t) control function, for each control Hamiltonian

# Optional key-word arguments
- `runIdx::Int64 = 3`: Use `1` for simulation, `2` for the gradient, and `3` for optimization (default)
- `maxIter::Int64 = 100`: Maximum number of optimization iterations
- `ncores::Int64 = 1`:  Number of MPI-tasks to use. Must be evenly divisible by the total number of essential states
- `quandary_exec::String="./main"`: Quandary executable
- `print_frequency_iter::Int64 = 1`: Output frequency for the optimizer
- `gamma_dpdm::Float64 = 0.0`: Coefficient for the penalty term that penalizes oscillations of the population
- `gamma_energy::Float64 = 0.0`: Coefficient for the penalty term that penalizes the energy in the control functions
- `final_objective::Int64 = 1`: Type of final objective function: 1=trace fidelity (default), 2=Frobenius norm squared
- `splines_real_imag::Bool = true`: B-spline parameterization: `true` (default) use both real and imaginary parts; `false` only use the amplitude and a fixed phase
- `phase_scaling_factor::Float64 = 1.0`: Scaling of the phase during the optimization

# Return argument
The return argument `qres` is a tuple with a content that depends on the input argument `runIdx`. 
- `runIdx=1`: qres[1] = objective, qres[2] = infidelity, qres[3] = penalty, qres[4] = unitary transformation at final time. 
- `runIdx=2`: qres[1] = objective, qres[2] = gradient, qres[3] = infidelity, qres[4] = penalty, qres[5] = fidelity. 
- `runIdx=3`: qres[1] = optimized control vector, qres[2] = infidelity, qres[3] = penalty, qres[4] = Tikhonov, qres[5] = optimization history, qres[6] = number of optimization iterations.
"""
function run_Quandary(params::objparams, pcof0::Vector{Float64}, optim_bounds::Vector{Float64}; runIdx::Int64 = 3, maxIter::Int64 = 100, ncores::Int64 = 1, quandary_exec::String="./main", print_frequency_iter::Int64 = 1, gamma_dpdm::Float64 = 0.0, gamma_energy::Float64 = 0.0, final_objective::Int64 = 1, splines_real_imag::Bool = true, phase_scaling_factor::Float64=1.0)
    # gamma_dpdm > 0.0 to penalize the 2nd time derivative of the population
    # final_objective = 1 corresponds to the trace infidelity
    
    # Future development: pcof0 could be a matrix(nCoeff,nRhs)
    if runIdx == 1
      runtype="simulation"
    elseif runIdx == 2
      runtype = "gradient"
    else
      runtype = "optimization"
    end
  
    Nt = params.Nt
    Ne = params.Ne
    T = params.T
    nsteps = params.nsteps
    couple_type = params.couple_type
    #println("\n run_quandary: couple_type = ", couple_type)
    
    rotfreq = params.Rfreq
    carrierfreq = params.Cfreq # Vector{Vector{Float64}}
    
    tikQ = 2*params.tik0/params.nCoeff # Scale the tikhonov coefficient
  
    leakage_weights = diag(params.wmat) # Scaling of leakage contribution
  
    if splines_real_imag # specify both the real and imaginary coefficients in the ctrl vector
      D1 = div(params.nCoeff, 2*params.NfreqTot)
      @assert(2*D1*params.NfreqTot == params.nCoeff) # no remainder is allowed
    else # only specify the amplitude vector and a constant phase
      D1p1 = div(params.nCoeff, params.NfreqTot)
      @assert(D1p1*params.NfreqTot == params.nCoeff) # no remainder is allowed
      D1 = D1p1 - 1
    end
  
    # transition frequencies
    freq01 = params.freq01 
    selfkerr = -params.self_kerr # self-Kerr coefficient (Quandary reverses the sign)
    if couple_type==1
      couple_coeff = -params.couple_coeff# cross-Kerr coefficient (Quandary reverses the sign)
    else
      couple_coeff = params.couple_coeff# cross-Kerr coefficient (Quandary reverses the sign)
    end
  
    inftol = params.traceInfidelityThreshold
  
    # Extract the essential levels from the target gate
    isEss, it2in = Juqbox.identify_essential_levels(Ne, Nt, false) # Quandary uses LSB ordering
    Ness = params.N # total number of essential levels
    Ntot = prod(params.Nt)
    targetgate = zeros(ComplexF64, Ness, Ness)
    i0 = 0
    for i = 1:Ntot
      if isEss[i]
        i0 += 1
        targetgate[i0,:] = params.Utarget_r[i,:] + im*params.Utarget_i[i,:]
      end
    end
  
  
    # Write gate to file"
    gatefilename = "targetgate.dat"
    @assert(size(targetgate)[1] == prod(Ne))
    @assert(size(targetgate)[2] == prod(Ne))
    # vectorize the matrices and append the vectors
    gate_1d = vcat(vec(real.(targetgate)), vec(imag.(targetgate)))
    f = open(gatefilename, "w")
    for k in 1:length(gate_1d)
      @printf(f, "%20.13e\n", gate_1d[k])
    end
    close(f)
    #writedlm(gatefilename, gate_1d)
  
    # Write initial pcof to file"
    initialpcof_filename = "pcof_init.dat"
    f = open(initialpcof_filename, "w")
    for k in 1:length(pcof0)
      # Write all columns in one long vector 
      @printf(f, "%20.13e\n", pcof0[k])
    end
    close(f)
    #writedlm(initialpcof_filename, pcof0)
  
    # Write Quandaries configuration file
    config_filename = "config.cfg"
    write_Quandary_config_file(config_filename, Nt, Ne, T, nsteps, freq01, rotfreq, selfkerr, couple_coeff, couple_type, D1, carrierfreq, gatefilename, initialpcof_filename, optim_bounds, inftol, maxIter, tikQ, leakage_weights, print_frequency_iter, runtype = runtype, gamma_dpdm = gamma_dpdm, final_objective = final_objective, gamma_energy = gamma_energy, splines_real_imag = splines_real_imag, phase_scaling_factor = phase_scaling_factor)
  
    # Set up the run command
    if ncores > 1
      runcommand = `mpirun -np $ncores ./main $config_filename --quiet`
    else 
      runcommand = `./main $config_filename --quiet`
    end
    # If not optimizing: Pipe std output to file rather than screen
    if (runtype == "simulation" || runtype == "gradient")
      exec = pipeline(runcommand, stdout="out.log", stderr="err.log")  
    else 
      exec = pipeline(runcommand, stderr="err.log")
    end
  
    # Run quandary
    #println("  -> Running Quandary (", exec, "), ...")
    @time "Quandary run time: " run(exec)
    #println("  -> Quandary done.")
  
    return get_Quandary_results(params, "./data_out", params.Nt, params.Ne, runtype=runtype)
  end
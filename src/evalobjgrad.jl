# The working_arrays struct holds all of the working arrays needed to call traceobjgrad. Preallocated for efficiency
"""
    wa = working_arrays(N:: Int64, Ntot:: Int64, Hconst::MyRealMatrix, Hsym_ops::Vector{MyRealMatrix}, Hanti_ops::Vector{MyRealMatrix}, Hunc_ops::Vector{MyRealMatrix}, isSymm::BitArray{1}, nCoeff::Int64)

Constructor for the mutable struct working_arrays containing preallocated temporary storage for time stepping.

 
# Arguments
- `param:: objparams`: Struct with problem definition
- `nCoeff:: Int64`: Number of parameters in optimization
"""
mutable struct working_arrays
    # Hamiltonian matrices
    K0  ::MyRealMatrix
    K05 ::MyRealMatrix
    K1  ::MyRealMatrix
    S0  ::MyRealMatrix
    S05 ::MyRealMatrix
    S1  ::MyRealMatrix

    # Forward/Adjoint variables+stages
    #vtargetr    ::Array{Float64,2} # moved to params
    #vtargeti    ::Array{Float64,2}
    lambdar     ::Array{Float64,2}
    lambdar0    ::Array{Float64,2}
    lambdai     ::Array{Float64,2}
    lambdai0    ::Array{Float64,2}
    lambdar05   ::Array{Float64,2}
    lambdar_nfrc  ::Array{Float64,2}
    lambdar0_nfrc ::Array{Float64,2}
    lambdai_nfrc  ::Array{Float64,2}
    lambdai0_nfrc ::Array{Float64,2}
    lambdar05_nfrc::Array{Float64,2}
    κ₁          ::Array{Float64,2}
    κ₂          ::Array{Float64,2}
    ℓ₁          ::Array{Float64,2}
    ℓ₂          ::Array{Float64,2}
    rhs         ::Array{Float64,2}
    gr0         ::Array{Float64,2}
    gi0         ::Array{Float64,2}
    gr1         ::Array{Float64,2}
    gi1         ::Array{Float64,2}
    hr0         ::Array{Float64,2}
    hi0         ::Array{Float64,2}
    hi1         ::Array{Float64,2}
    hr1         ::Array{Float64,2}
    vr          ::Array{Float64,2}
    vi          ::Array{Float64,2}
    vi05        ::Array{Float64,2}
    vr0         ::Array{Float64,2}
    vfinalr     ::Array{Float64,2}
    vfinali     ::Array{Float64,2}
    gr          ::Array{Float64,1}
    gi          ::Array{Float64,1}
    gradobjfadj ::Array{Float64,1}
    tr_adj      ::Array{Float64,1}

    function working_arrays(N:: Int64, Ntot:: Int64, Hconst::MyRealMatrix, Hsym_ops::Vector{MyRealMatrix}, Hanti_ops::Vector{MyRealMatrix}, Hunc_ops::Vector{MyRealMatrix}, isSymm::BitArray{1}, pFidType::Int64, objFuncType::Int64, nCoeff::Int64)
        #N = params.N
        #Ntot = N + params.Nguard

        # K0,S0,K05,S05,K1,S1,vtargetr,vtargeti = KS_alloc(params)
        # ks_alloc(Ntot:: Int64, Hconst::MyRealMatrix, Hsym_ops::Vector{MyRealMatrix}, Hanti_ops::Vector{MyRealMatrix}, Hunc_ops::Vector{MyRealMatrix}, isSymm::BitArray{1})
        K0, S0, K05, S05, K1, S1 = ks_alloc(Ntot, Hconst, Hsym_ops, Hanti_ops, Hunc_ops, isSymm)

        lambdar,lambdar0,lambdai,lambdai0,lambdar05,κ₁,κ₂,ℓ₁,ℓ₂,rhs,gr0,gi0,gr1,gi1,hr0,hi0,hi1,hr1,vr,vi,vi05,vr0,vfinalr,vfinali = time_step_alloc(Ntot,N)
        if pFidType == 3
            gr, gi, gradobjfadj, tr_adj = grad_alloc(nCoeff-1)
        else
            gr, gi, gradobjfadj, tr_adj = grad_alloc(nCoeff)
        end
        if objFuncType != 1
            lambdar_nfrc  = zeros(Float64,size(lambdar))
            lambdar0_nfrc = zeros(Float64,size(lambdar0))
            lambdai_nfrc  = zeros(Float64,size(lambdai))
            lambdai0_nfrc = zeros(Float64,size(lambdai0))
            lambdar05_nfrc= zeros(Float64,size(lambdar05))
        else
            lambdar_nfrc  = zeros(0,0)
            lambdar0_nfrc = zeros(0,0)
            lambdai_nfrc  = zeros(0,0)
            lambdai0_nfrc = zeros(0,0)
            lambdar05_nfrc= zeros(0,0)
        end
        # new(K0,S0,K05,S05,K1,S1,vtargetr,vtargeti,
        #     lambdar,lambdar0,lambdai,lambdai0,lambdar05,
        #     lambdar_nfrc,lambdar0_nfrc,lambdai_nfrc,lambdai0_nfrc,lambdar05_nfrc,
        #     κ₁,κ₂,ℓ₁,ℓ₂,rhs,gr0,gi0,gr1,gi1,hr0,hi0,hi1,hr1,
        #     vr,vi,vi05,vr0,vfinalr,vfinali,gr, gi, gradobjfadj, tr_adj)
        new(K0, S0, K05, S05, K1, S1,
            lambdar,lambdar0,lambdai,lambdai0,lambdar05,
            lambdar_nfrc,lambdar0_nfrc,lambdai_nfrc,lambdai0_nfrc,lambdar05_nfrc,
            κ₁,κ₂,ℓ₁,ℓ₂,rhs,gr0,gi0,gr1,gi1,hr0,hi0,hi1,hr1,
            vr, vi, vi05, vr0, vfinalr, vfinali, gr, gi, gradobjfadj, tr_adj)
    end
    
end

"""
    params = objparams(Ne, Ng, T, Nsteps;
                        Uinit=Uinit, 
                        Utarget=Utarget,
                        Cfreq=Cfreq, 
                        Rfreq=Rfreq, 
                        Hconst=Hconst [, 
                        Hsym_ops=Hsym_ops,
                        Hanti_ops=Hanti_ops, 
                        Hunc_ops=Hunc_ops,
                        wmatScale=wmatScale,
                        objFuncType=objFuncType,
                        leak_ubound=leak_ubound,
                        linear_solver = lsolver_object(),
                        use_sparse = use_sparse],
                        dVds = dVds)

Constructor for the mutable struct objparams. The sizes of the arrays in the argument list are based on
`Ntot = prod(Ne + Ng)`, `Ness = prod(Ne)`, `Nosc = length(Ne) = length(Ng)`.

Notes: It is assumed that `length(Hsym_ops) = length(Hanti_ops) =: Ncoupled`. The matrices `Hconst`,
`Hsym_ops[j]`and `Hanti_ops[j]`, for j∈[1,Ncoupled], must all be of size `Ntot × Ntot`. The matrices
`Hsym_ops[j]` must be symmetric and `Hanti_ops[j]` must be skew-symmetric. The matrices
`Hunc_ops[j]`, for j∈[1,Nunc], where `Nunc = length(Hunc_ops)`, must also be of size `Ntot × Ntot`
and either be symmetric or skew-symmetric.
 
# Arguments
- `Ne::Array{Int64,1}`: Number of essential energy levels for each subsystem
- `Ng::Array{Int64,1}`: Number of guard energy levels for each subsystem
- `T::Float64`: Duration of gate
- `Nsteps::Int64`: Number of timesteps for integrating Schroedinger's equation
- `Uinit::Array{Float64,2}`: (keyword) Matrix holding the initial conditions for the solution matrix of size Uinit[Ntot, Ness]
- `Utarget::Array{Complex{Float64},2}`: (keyword) Matrix holding the target gate matrix of size Uinit[Ntot, Ness]
- `Cfreq::Vector{Vector{Float64}}`: (keyword) Carrier wave (angular) frequencies of size Cfreq[Nctrl]
- `Rfreq::Array{Float64,1}`: (keyword) Rotational (regular) frequencies for each control Hamiltonian; size Rfreq[Nctrl]
- `Hconst::Array{Float64,2}`: (keyword) Time-independent part of the Hamiltonian matrix of size Ntot × Ntot
- `Hsym_ops:: Array{Array{Float64,2},1}`: (keyword) Array of symmetric control Hamiltonians, each of size Ntot × Ntot
- `Hanti_ops:: Array{Array{Float64,2},1}`: (keyword) Array of anti-symmetric control Hamiltonians, each of size Ntot × Ntot
- `Hunc_ops:: Array{Array{Float64,2},1}`: (keyword) Array of uncoupled control Hamiltonians, each of size Ntot × Ntot
- `wmatScale::Float64 = 1.0`: (keyword) Scaling factor for suppressing guarded energy levels
- `objFuncType::Int64 = 1`  # 1 = objective function include infidelity and leakage
                            # 2 = objective function only includes infidelity... no leakage in obj function or constraint
                            # 3 = objective function only includes infidelity; leakage treated as inequality constraint
- `leak_ubound::Float64 = 1.0e-3`  : The upper bound on the leakage inequality constraint (See examples/cnot2-leakieq-setup.jl )
- `linear_solver::lsolver_object = lsolver_object()` : The linear solver object used to solve the implicit & adjoint system
- `use_sparse::Bool = false`: (keyword) Set to true to sparsify all Hamiltonian matrices
- `dVds::Array{Complex{Float64},2}`: (keyword) Matrix holding the complex-valued matrix dV/ds of size Ntot x Ne (for continuation)
"""
mutable struct objparams
    Nosc   ::Int64          # number of oscillators in the coupled quantum systems
    N      ::Int64          # total number of essential levels
    Nguard ::Int64          # total number of extra levels
    Ne     ::Array{Int64,1} # essential levels for each oscillator
    Ng     ::Array{Int64,1} # guard levels for each oscillator
    Nt     ::Array{Int64,1} # total # levels for each oscillator
    T      ::Float64        # final time

    nsteps       ::Int64    # Number of time steps
    Uinit        ::Array{Float64,2} # initial condition for each essential state: Should be a basis
    # Utarget      ::Array{Complex{Float64},2}
    Utarget_r      ::Array{Float64,2}
    Utarget_i      ::Array{Float64,2}
    use_bcarrier ::Bool
#    Nfreq        ::Int64 # number of frequencies
#    Cfreq        ::Array{Float64,2} # Carrier wave frequencies of dim Cfreq[seg,freq]
    NfreqTot     ::Int64
    Nfreq        ::Vector{Int64} # number of carrier frequencies in each control function
    Cfreq        ::Vector{Vector{Float64}} # Pointer to carrier wave frequencies of dim Cfreq[ctrl]
    kpar         ::Int64   # element of gradient to test
    tik0         ::Float64
#    tik1         ::Float64

    # Drift Hamiltonian
    Hconst ::MyRealMatrix     # time-independent part of the Hamiltonian (assumed symmetric)
   
    # Control Hamiltonians
    Hsym_ops  ::Vector{MyRealMatrix}   # Symmetric control Hamiltonians
    Hanti_ops ::Vector{MyRealMatrix}   # Anti-symmetric control Hamiltonians
    Hunc_ops  ::Vector{MyRealMatrix}   # Uncoupled control Hamiltonians

    Ncoupled :: Int64 # Number of coupled Hamiltonians.
    Nunc     :: Int64 # Number of uncoupled Hamiltonians.
    isSymm   :: BitArray{1} # Array to track symmetry of Hunc_ops entries

    Ident ::MyRealMatrix
    wmat  ::Diagonal{Float64,Array{Float64,1}} # Weights for discouraging guard level population 

    # Matrix of forbidden states for leakage penalty 
    forb_states  ::Array{ComplexF64,2}
    forb_weights ::Vector{Float64}
    wmat_real    ::WeightMatrix
    wmat_imag    ::WeightMatrix

    # Type of fidelity
    pFidType    ::Int64
    globalPhase ::Float64

    # Optimization problem formulation
    objFuncType ::Int64   # 1 = objective function include infidelity and leakage
                          # 2 = objective function only includes infidelity... no leakage in obj function or constraint
                          # 3 = objective function only includes infidelity; leakage treated as inequality constraint                            
    leak_ubound ::Float64 # The upper bound on the leakage inequality constraint

    #Store information from last computation
    last_leak       ::Float64
    last_infidelity ::Float64
    last_pcof       ::Array{Float64,1}
    last_leak_grad  ::Array{Float64,1}
    last_infidelity_grad::Array{Float64,1}

    # Convergence history variables
    saveConvHist  ::Bool;
    objHist       ::Array{Float64,1}
    primaryHist   ::Array{Float64,1}
    secondaryHist ::Array{Float64,1}
    dualInfidelityHist  ::Array{Float64,1}

    #Linear solver object to solve linear system in timestepping
    linear_solver ::lsolver_object

    objThreshold :: Float64
    traceInfidelityThreshold :: Float64
    lastTraceInfidelity :: Float64
    lastLeakIntegral :: Float64    

    usingPriorCoeffs :: Bool
    priorCoeffs ::Array{Float64,1}

    quiet:: Bool # quiet mode?

    Rfreq::Array{Float64,1}

    save_pcof_hist:: Bool
    pcof_hist:: Array{Vector{Float64},1}

    dVds_r      ::Array{Float64,2}
    dVds_i      ::Array{Float64,2}

    sv_type:: Int64

    # temporary storage for time stepping, to be allocated later (by setup_ipopt_problem)
    wa:: working_arrays

    nCoeff :: Int64 # Length of the control vector
    
    freq01::   Vector{Float64} 
    self_kerr:: Vector{Float64}
    couple_coeff:: Vector{Float64}
    couple_type:: Int64

# Regular arrays
    function objparams(Ne::Array{Int64,1}, Ng::Array{Int64,1}, T::Float64, nsteps::Int64;
                       Uinit::Array{Float64,2}, Utarget::Array{Complex{Float64},2}, # keyword args w/o default values (must be assigned)
                       Cfreq::Vector{Vector{Float64}}, Rfreq::Array{Float64,1}, Hconst::Array{Float64,2},
                       Hsym_ops:: Array{Array{Float64,2},1} = Array{Float64,2}[], # keyword args with default values
                       Hanti_ops:: Array{Array{Float64,2},1} = Array{Float64,2}[],
                       Hunc_ops:: Array{Array{Float64,2},1} = Array{Float64,2}[],
                       forb_states:: Array{ComplexF64,2} = Array{ComplexF64}(undef,0,2),
                       forb_weights:: Vector{Float64} = Float64[],
                       objFuncType:: Int64 = 1, leak_ubound:: Float64=1.0e-3,
                       wmatScale::Float64 = 1.0, use_sparse::Bool = false, use_custom_forbidden::Bool = false,
                       linear_solver::lsolver_object = lsolver_object(nrhs=prod(Ne)), msb_order::Bool = true,
                       dVds::Array{ComplexF64,2}= Array{ComplexF64}(undef,0,0), nCoeff::Int, freq01::Vector{Float64} = Vector{Float64}[], self_kerr::Vector{Float64} = Vector{Float64}[], couple_coeff::Vector{Float64} = Vector{Float64}[], couple_type::Int64 = 0)
        pFidType = 2
        Nosc   = length(Ne) # number of subsystems
        N      = prod(Ne)
        Ntot   = prod(Ne+Ng)
        Nguard = Ntot-N
        # Nfreq  = size(Cfreq,2)
        Nctrl = length(Cfreq)
        Ncoupled = length(Hsym_ops)
        Nanti  = length(Hanti_ops)
        Nunc   = length(Hunc_ops)

        Nctrl = Ncoupled + Nunc # Number of control Hamiltonians
        
        @assert(Ncoupled==Nctrl)
        @assert(length(Rfreq) >= Nctrl)
        
        if(Nunc > 0)
            throw(ArgumentError("Uncoupled Hamiltonians are currently not supported.\n"))
        end

        # Check size of Uinit, Utarget
        tz = ( Ntot, N )
        @assert( size(Uinit) == tz)
        @assert( size(Utarget) == tz)
        #println("Passed size compatibility tests")

        @assert(Ncoupled == Nanti)
        # Exit if there are any uncoupled controls

        Nfreq = Vector{Int64}(undef,Nctrl) # setup Nfreq vector from Cfreq
        for c = 1:Nctrl
            Nfreq[c] = length(Cfreq[c])
        end
        NfreqTot = sum(Nfreq)
        println("objparams: NfreqTot = ", NfreqTot)
    
        # Track symmetries of uncoupled Hamiltonian terms
        if Nunc > 0
            isSymm = BitArray(undef, Nunc)
            for i=1:Nunc
                if(issymmetric(Hunc_ops[i]))
                    isSymm[i] = true 
                elseif(norm(Hunc_ops[i] + Hunc_ops[i]') < 1e-15)
                    isSymm[i] = false
                else 
                    throw(ArgumentError("Uncoupled Hamiltonian is not symmetric or anti-symmetric. This functionality is not currently supported.\n"))
                end
            end
        else
            isSymm = BitArray(undef, Nunc)
        end
        
        # Set default Tikhonov parameter
        tik0 = 0.01

        # By default, test the first parameter for gradient correctness
        kpar = 1

        # By default use B-splines with carrier waves
        use_bcarrier = true

        # Weights in the W matrix for discouraging population of guarded states
        wmat = wmatScale.*Juqbox.wmatsetup(Ne, Ng, msb_order)

        # Build weighting matrices if there are user-specified forbidden states
        if use_custom_forbidden

            if size(forb_states,1) != Ntot
                throw(ArgumentError("Forbidden states array is an incorrect size. Make sure guard 
                    levels are accounted for!\n"))
            end
            wmat_real = zeros(Ntot,Ntot)
            wmat_imag = zeros(Ntot,Ntot)
            for k = 1:size(forb_states,2)
                weight_loc = forb_weights[k]
                for j = 1:size(forb_states,1)
                    f_loc = conj(forb_states[j,k])
                    @fastmath @inbounds @simd for i = 1:size(forb_states,1)
                        val = f_loc*forb_states[i,k]
                        wmat_real[i,j] += weight_loc*real(val)
                        wmat_imag[i,j] += weight_loc*imag(val)
                    end
                end
            end
        else 
            forb_states = zeros(1,1)
            wmat_real = copy(wmat)
            wmat_imag = Diagonal(zeros(Ntot))
            forb_weights = zeros(1)
        end

        # By default save convergence history
        saveConvHist = true

        # Check for consistency in coupled controls 
        for i = 1:Ncoupled
            L = LinearAlgebra.tril(Hsym_ops[i] + Hanti_ops[i]) # tril forms the lower triangular part of a matrix, in this case (a+a^† ) + (a - a^†) = 2 a, which is upper triangular
            if(norm(L) > eps(1.0))
                println("WARNING: Control Hamiltonian #", i, " may be inconsistently defined because H_sym+H_anti has a lower triangular part.")
            end
        end

        quiet = false

        objThreshold = 0.0
        traceInfidelityThreshold = 0.0
        usingPriorCoeffs = false
        priorCoeffs = [] # zeros(0)

        if use_sparse
            #println("Info: converting Hamiltonian matrices to sparse format")
            Ident = sparse(Matrix{Float64}(I, Ntot, Ntot))
            Hconst = sparse(Hconst)
            if  Ncoupled == 0
                Hsym_ops1 = SparseMatrixCSC{Float64,Int64}[]
                Hanti_ops1 = SparseMatrixCSC{Float64,Int64}[]
            elseif Ncoupled == 1
                Hsym_ops1 = [sparse(Hsym_ops[1])]
                Hanti_ops1 = [sparse(Hanti_ops[1])]
            elseif Ncoupled == 2
                Hsym_ops1 = [sparse(Hsym_ops[1]), sparse(Hsym_ops[2])]
                Hanti_ops1 = [sparse(Hanti_ops[1]), sparse(Hanti_ops[2])]
            elseif Ncoupled == 3
                Hsym_ops1 = [sparse(Hsym_ops[1]), sparse(Hsym_ops[2]), sparse(Hsym_ops[3])]
                Hanti_ops1 = [sparse(Hanti_ops[1]), sparse(Hanti_ops[2]), sparse(Hanti_ops[3])]
            else
                throw(ArgumentError("Sparsification of Hamiltonians only implemented for 0, 1, 2 & 3 elements.\n"))
            end
            
            for q=1:Ncoupled
                dropzeros!(Hsym_ops1[q])
                dropzeros!(Hanti_ops1[q])
            end

            if Nunc == 0
                Hunc_ops1 = SparseMatrixCSC{Float64,Int64}[]
            elseif Nunc == 1
                Hunc_ops1 = [sparse(Hunc_ops[1])]
            elseif Nunc == 2
                Hunc_ops1 = [sparse(Hunc_ops[1]), sparse(Hunc_ops[2])]
            elseif Nunc == 3
                Hunc_ops1 = [sparse(Hunc_ops[1]), sparse(Hunc_ops[2]), sparse(Hunc_ops[3])]
            else
                throw(ArgumentError("Sparsification of Hamiltonians only implemented for 1, 2 & 3 elements.\n"))
            end
            
            for q=1:Nunc
                dropzeros!(Hunc_ops1[q])
            end
        else
            Ident = Matrix{Float64}(I, Ntot, Ntot)
            Hsym_ops1 = Hsym_ops
            Hanti_ops1 = Hanti_ops
            Hunc_ops1 = Hunc_ops
        end
        
        if length(dVds) == 0
            my_dVds = copy(Utarget) # make a copy to be safe
            my_sv_type = 1
        else
            @assert(size(dVds) == size(Utarget))
            my_dVds = dVds
            my_sv_type = 2
        end

        wa = working_arrays(N, Ntot, convert(MyRealMatrix, Hconst), convert(Vector{MyRealMatrix}, Hsym_ops1), convert(Vector{MyRealMatrix}, Hanti_ops1), convert(Vector{MyRealMatrix}, Hunc_ops1), isSymm, pFidType, objFuncType, nCoeff)


        # sv_type is used for continuation. Only change this if you know what you are doing
        new(
             Nosc, N, Nguard, Ne, Ng, Ne+Ng, T, nsteps, Uinit, real(Utarget), imag(Utarget), 
             use_bcarrier, NfreqTot, Nfreq, Cfreq, kpar, tik0, Hconst, Hsym_ops1, 
             Hanti_ops1, Hunc_ops1, Ncoupled, Nunc, isSymm, Ident, wmat, 
             forb_states, forb_weights, wmat_real, wmat_imag, pFidType, 0.0,
             objFuncType, leak_ubound,
             0.0,0.0,zeros(0),zeros(0),zeros(0),saveConvHist,
             zeros(0), zeros(0), zeros(0), zeros(0), 
             linear_solver, objThreshold, traceInfidelityThreshold, 0.0, 0.0, 
             usingPriorCoeffs, priorCoeffs, quiet, Rfreq, false, [],
             real(my_dVds), imag(my_dVds), my_sv_type, wa, nCoeff,
             freq01, self_kerr, couple_coeff, couple_type # Add some checks for these ones!
            )

    end

end # mutable struct objparams


"""
    objf = traceobjgrad(pcof0, params[, verbose = false, evaladjoint = true])

Perform a forward and/or adjoint Schrödinger solve to evaluate the objective
function and/or gradient.
 
# Arguments
- `pcof0::Array{Float64,1}`: Array of parameter values defining the controls
- `param::objparams`: Struct with problem definition
- `verbose::Bool = false`: Run simulation with additional terminal output and store state history.
- `evaladjoint::Bool = true`: Solve the adjoint equation and calculate the gradient of the objective function.
"""
function traceobjgrad(pcof0::Array{Float64,1},  params::objparams, verbose::Bool = false, evaladjoint::Bool = true)
    wa = params.wa
    order  = 2
    N      = params.N    
    Nguard = params.Nguard  
    T      = params.T

    # Utarget = params.Utarget # Never used

    nsteps = params.nsteps
    tik0   = params.tik0

    H0 = params.Hconst
    Ng = params.Ng
    Ne = params.Ne
    
    Nt   = params.Nt # vector
    Ntot = N + Nguard # scalar

    Ncoupled  = params.Ncoupled # Number of symmetric control Hamiltonians. We currently assume that the number of anti-symmetric Hamiltonians is the same
    Nunc  = params.Nunc # Number of uncoupled control functions.
    Nosc  = params.Nosc
    Nfreq = params.Nfreq
    Nsig  = 2*(Ncoupled + Nunc) # Only uses for regular B-splines

    linear_solver = params.linear_solver    

    # Reference pre-allocated working arrays
    K0 = wa.K0
    S0 = wa.S0
    K05 = wa.K05
    S05 = wa.S05
    K1 = wa.K1
    S1 = wa.S1
    vtargetr = params.Utarget_r
    vtargeti = params.Utarget_i
    # vtargetr = wa.vtargetr
    # vtargeti = wa.vtargeti

    # New variables to accomodate continuation. By default dVds = Utarget
    dVds_r = params.dVds_r
    dVds_i = params.dVds_i
    # tmp
    # println("dVds")

    lambdar = wa.lambdar
    lambdar0 = wa.lambdar0
    lambdai = wa.lambdai
    lambdai0 = wa.lambdai0
    lambdar05 = wa.lambdar05
    lambdar_nfrc  = wa.lambdar_nfrc
    lambdar0_nfrc = wa.lambdar0_nfrc
    lambdai_nfrc  = wa.lambdai_nfrc
    lambdai0_nfrc = wa.lambdai0_nfrc
    lambdar05_nfrc= wa.lambdar05_nfrc
    κ₁ = wa.κ₁
    κ₂ = wa.κ₂
    ℓ₁ = wa.ℓ₁
    ℓ₂ = wa.ℓ₂
    rhs = wa.rhs
    gr0 = wa.gr0
    gi0 = wa.gi0
    gr1 = wa.gr1
    gi1 = wa.gi1
    hr0 = wa.hr0
    hi0 = wa.hi0
    hi1 = wa.hi1
    hr1 = wa.hr1
    vr = wa.vr
    vi = wa.vi
    vi05 = wa.vi05
    vr0 = wa.vr0
    vfinalr = wa.vfinalr # temporary storage for final time-stepped solution
    vfinali = wa.vfinali
    gr = wa.gr
    gi = wa.gi
    gradobjfadj = wa.gradobjfadj 
    tr_adj = wa.tr_adj


    # primary fidelity type
    pFidType = params.pFidType  

    if pFidType == 3
        nCoeff = size(pcof0,1)
        pcof = zeros(nCoeff-1)
        pcof[:] = pcof0[1:nCoeff-1] # these are for the B-spline coefficients
        params.globalPhase = pcof0[nCoeff]
    else
        pcof = pcof0
    end

    Psize = size(pcof,1) #must provide separate coefficients for real,imaginary, and uncoupled parts of the control fcn
    #
    #
    if Psize%2 != 0
        error("pcof must have an even number of elements, not ", Psize)
    end
    if params.use_bcarrier
        # NOTE: Nsig  = 2*(Ncoupled + Nunc)
        # D1 = div(Psize, Nsig*Nfreq)  # 
        # Psize = D1*Nsig*Nfreq # active part of the parameter array
        D1 = div(Psize, 2*params.NfreqTot)  # 
        Psize = 2*D1*params.NfreqTot # active part of the parameter array
    else
        # NOTE: Nsig  = 2*(Ncoupled + Nunc)
        D1 = div(Psize, Nsig)
        Psize = D1*Nsig # active part of the parameter array
    end
    
    tinv ::Float64 = 1.0/T
    
    # Parameters used for the gradient
    kpar = params.kpar

    if verbose
        println("Vector dim Ntot =", Ntot , ", Guard levels Nguard = ", Nguard , ", Param dim, Psize = ", Psize, ", Spline coeffs per func, D1= ", D1, ", Nsteps = ", nsteps, " Tikhonov coeff: ", tik0)
    end
    
    Ident = params.Ident

    # coefficients for penalty style #2 (wmat is a Diagonal matrix)
    # wmat = params.wmat
    
    wmat_real = params.wmat_real
    wmat_imag = params.wmat_imag

    # Here we can choose what kind of control function expansion we want to use
    if (params.use_bcarrier)
        # FMG FIX
        splinepar = bcparams(T, D1, params.Cfreq, pcof) # Assumes Nunc = 0
    else
    # the old bsplines is the same as the bcarrier with Cfreq = 0
        splinepar = splineparams(T, D1, Nsig, pcof)   # parameters for B-splines
    end

    # it is up to the user to estimate the number of time steps
    dt ::Float64 = T/nsteps

    gamma, stages = getgamma(order)

    if verbose
        println("Final time: ", T, ", number of time steps: " , nsteps , ", time step: " , dt )
    end
    
    #real and imaginary part of initial condition
    copy!(vr,params.Uinit)
    vi   .= 0.0

    # initialize temporaries
    vi05 .= 0.0
    vr0  .= 0.0

    # Zero out working arrays
    κ₁   .= 0.0
    κ₂   .= 0.0
    ℓ₁   .= 0.0
    ℓ₂   .= 0.0
    rhs  .= 0.0

    gr0  .= 0.0
    gi0  .= 0.0
    gr1  .= 0.0
    gi1  .= 0.0
    hr0  .= 0.0
    hi0  .= 0.0
    hi1  .= 0.0
    hr1  .= 0.0
    gr   .= 0.0
    gi   .= 0.0
    
    if verbose
        usaver = zeros(Float64,Ntot,N,nsteps+1)
        usavei = zeros(Float64,Ntot,N,nsteps+1)
        usaver[:,:,1] = vr # the rotation to the lab frame is the identity at t=0
        usavei[:,:,1] = -vi

        #to compute gradient with forward method
        if evaladjoint
            wr   = zeros(Float64,Ntot,N) 
            wi   = zeros(Float64,Ntot,N) 
            wr1  = zeros(Float64,Ntot,N) 
            wi05 = zeros(Float64,Ntot,N) 
            objf_alpha1 = 0.0
        end
    end

    #initialize variables for time stepping
    t     ::Float64 = 0.0
    step  :: Int64 = 0
    objfv ::Float64 = 0.0


    # Forward time stepping loop
    for step in 1:nsteps

        forbidden0 = tinv*penalf2aTrap(vr, wmat_real)
        # Störmer-Verlet
        for q in 1:stages
            copy!(vr0,vr)
            t0  = t
            
            # Update K and S matrices
            # general case
            KS!(K0, S0, t, params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq) 
            KS!(K05, S05, t + 0.5*dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq) 
            KS!(K1, S1, t + dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq) 
            
            # Take a step forward and accumulate weight matrix integral. Note the √2 multiplier is to account
            # for the midpoint rule in the numerical integration of the imaginary part of the signal.
            @inbounds t = step!(t, vr, vi, vi05, dt*gamma[q], K0, S0, K05, S05, K1, S1, Ident, κ₁, κ₂, ℓ₁, ℓ₂, rhs,linear_solver)

            forbidden = tinv*penalf2a(vr, vi05, wmat_real)  
            forbidden_imag1 = tinv*penalf2imag(vr0, vi05, wmat_imag)
            objfv = objfv + gamma[q]*dt*0.5*(forbidden0 + forbidden - 2.0*forbidden_imag1)

            # Keep prior value for next step (FG: will this work for multiple stages?)
            forbidden0 = forbidden

            # compute component of the gradient for verification of adjoint method
            if evaladjoint && verbose
                # compute the forcing for (wr, wi)
                fgradforce!(params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc,
                            params.isSymm, vr0, vi05, vr, t-dt, dt, splinepar, kpar, gr0, gr1, gi0, gi1, gr, gi)

                copy!(wr1,wr)

                @inbounds step_fwdGrad!(t0, wr1, wi, wi05, dt*gamma[q],
                                        gr0, gi0, gr1, gi1, K0, S0, K05, S05, K1, S1, Ident, κ₁, κ₂, ℓ₁, ℓ₂, rhs,linear_solver) 
                
                # Real part of forbidden state weighting
                forbalpha0 = tinv*penalf2grad(vr0, vi05, wr, wi05, wmat_real)
                forbalpha1 = tinv*penalf2grad(vr, vi05, wr1, wi05, wmat_real)

                # Imaginary part of forbidden state weighting
                forbalpha2 = tinv*penalf2grad(wi05, vi05, vr0, wr, wmat_imag)                

                copy!(wr,wr1)
                # accumulate contribution from the leak term
                objf_alpha1 = objf_alpha1 + gamma[q]*dt*0.5*2.0*(forbalpha0 + forbalpha1 + forbalpha2) 

            end  # evaladjoint && verbose
        end # Stromer-Verlet
        
        if verbose
            # rotated frame
            usaver[:,:, step + 1] = vr
            usavei[:,:, step + 1] = -vi
        end
    end #forward time stepping loop

if pFidType == 1
    scomplex1 = tracefidcomplex(vr, -vi, vtargetr, vtargeti)
    primaryobjf = 1+tracefidabs2(vr, -vi, vtargetr, vtargeti) - 2*real(scomplex1*exp(-1im*params.globalPhase)) # global phase angle 
elseif pFidType == 2
    primaryobjf = (1.0-tracefidabs2(vr, -vi, vtargetr, vtargeti)) # insensitive to global phase angle
elseif pFidType == 3 || pFidType == 4
    rotTarg = exp(1im*params.globalPhase)*(vtargetr + im*vtargeti)
    primaryobjf = (1.0 - tracefidreal(vr, -vi, real(rotTarg), imag(rotTarg)) )
end

secondaryobjf = objfv
objfv = primaryobjf + secondaryobjf

if evaladjoint && verbose
    # salpha1 = tracefidcomplex(wr, -wi, vtargetr, vtargeti)
    salpha1 = tracefidcomplex(wr, -wi, dVds_r, dVds_i)
    scomplex1 = tracefidcomplex(vr, -vi, vtargetr, vtargeti)
    if pFidType==1
        primaryobjgrad = 2*real(conj( scomplex1 - exp(1im*params.globalPhase) )*salpha1)
    elseif pFidType == 2
        primaryobjgrad = - 2*real(conj(scomplex1)*salpha1)
    elseif pFidType == 3 || pFidType == 4
        rotTarg = exp(1im*params.globalPhase)*(vtargetr + im*vtargeti)
        # grad wrt the control function 
        primaryobjgrad = - tracefidreal(wr, -wi, real(rotTarg), imag(rotTarg))
        # grad wrt the global phase
        primObjGradPhase = - tracefidreal(vr, -vi, real(im.*rotTarg), imag(im.*rotTarg))
    end
    objf_alpha1 = objf_alpha1 + primaryobjgrad
end  

vfinalr = copy(vr)
vfinali = copy(-vi)


traceInfidelity = 1.0 - tracefidabs2(vfinalr, vfinali, vtargetr, vtargeti)

if evaladjoint

    if verbose
        dfdp = objf_alpha1
    end  

    if (params.use_bcarrier)
        # gradSize = (2*Ncoupled+Nunc)*Nfreq*D1
        gradSize = params.NfreqTot*2*D1
    else
        gradSize = Nsig*D1
    end


    # initialize array for storing the adjoint gradient so it can be returned to the calling function/program
    leakgrad = zeros(0);
    infidelgrad = zeros(0);
    gradobjfadj[:] .= 0.0    
    t = T
    dt = -dt

    
    # println("traceobjgrad(): eval_adjoint: sv_type = ", params.sv_type) # tmp
    if params.sv_type == 1 || params.sv_type == 2 # regular case
        # println("scomplex #1 (vtarget)")
        scomplex0 = tracefidcomplex(vr, -vi, vtargetr, vtargeti)
    elseif params.sv_type == 3 # term2 for d/ds(grad(G))
        # println("scomplex #3 (dVds)")
        scomplex0 = tracefidcomplex(vr, -vi, dVds_r, dVds_i)
    #     println("Unknown sv_type = ", params.sv_type)
    end

    if pFidType == 1
        scomplex0 = exp(1im*params.globalPhase) - scomplex0
    end


    # Set initial condition for adjoint variables
    # Note (vtargetr, vtargeti) needs to be changed to dV/ds for continuation applications
    # By default, dVds = vtarget
    if params.sv_type == 1 # regular case
        # println("init_adjoint #1 (dVds)")
        init_adjoint!(pFidType, params.globalPhase, N, scomplex0, lambdar, lambdar0, lambdar05, lambdai, lambdai0,
                    vtargetr, vtargeti)
    elseif params.sv_type == 2 # term1 for d/ds(grad(G))
        # println("init_adjoint #2 (dVds)")
        init_adjoint!(pFidType, params.globalPhase, N, scomplex0, lambdar, lambdar0, lambdar05, lambdai, lambdai0,
                    dVds_r, dVds_i)               
    elseif params.sv_type == 3 # term2 for d/ds(grad(G))
        init_adjoint!(pFidType, params.globalPhase, N, scomplex0, lambdar, lambdar0, lambdar05, lambdai, lambdai0,
                    vtargetr, vtargeti)
    end

    #Initialize adjoint variables without forcing
    if params.objFuncType != 1
        lambdar_nfrc  .= lambdar
        lambdar0_nfrc .= lambdar0
        lambdai_nfrc  .= lambdai
        lambdai0_nfrc .= lambdai0
        lambdar05_nfrc.= lambdar05
        infidelgrad = zeros(gradSize);
    end

    #Backward time stepping loop
    for step in nsteps-1:-1:0

        # Forcing for the real part of the adjoint variables in first PRK "stage"
        mul!(hr0, wmat_real, vr, tinv, 0.0)

        #loop over stages
        for q in 1:stages
            t0 = t
            copy!(vr0,vr)
            
            # update K and S
            # Since t is negative we have that K0 is K^{n+1}, K05 = K^{n-1/2}, 
            # K1 = K^{n} and similarly for S.
            # general case
            KS!(K0, S0, t, params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq) 
            KS!(K05, S05, t + 0.5*dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq) 
            KS!(K1, S1, t + dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq) 


            # Integrate state variables backwards in time one step
            @inbounds t = step!(t, vr, vi, vi05, dt*gamma[q], K0, S0, K05, S05, K1, S1, Ident, κ₁, κ₂, ℓ₁, ℓ₂, rhs,linear_solver)

            # Forcing for adjoint equations (real part of forbidden state penalty)
            mul!(hi0,wmat_real,vi05,tinv,0.0)
            mul!(hr1,wmat_real,vr,tinv,0.0)

            # Forcing for adjoint equations (imaginary part of forbidden state penalty)
            mul!(hr1,wmat_imag,vi05,tinv,1.0)
            copy!(hi1,hi0)
            mul!(hi1,wmat_imag,vr,-tinv,1.0)

            # evolve lambdar, lambdai
            temp = t0
            @inbounds temp = step!(temp, lambdar, lambdai, lambdar05, dt*gamma[q], hr0, hi0, hr1, hi1, 
                                    K0, S0, K05, S05, K1, S1, Ident, κ₁, κ₂, ℓ₁, ℓ₂, rhs,linear_solver)

            # Accumulate gradient
            adjoint_grad_calc!(params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, vr0, vi05, vr, 
                                lambdar0, lambdar05, lambdai, lambdai0, t0, dt,splinepar, gr, gi, tr_adj) 
            axpy!(gamma[q]*dt,tr_adj,gradobjfadj)
            
            # save for next stage
            copy!(lambdai0,lambdai)
            copy!(lambdar0,lambdar)

            #Do adjoint step to compute infidelity grad (without forcing)
            if params.objFuncType != 1
                temp = t0
                @inbounds temp = step_no_forcing!(temp, lambdar_nfrc, lambdai_nfrc, lambdar05_nfrc, dt*gamma[q], 
                                                K0, S0, K05, S05, K1, S1, Ident, κ₁, κ₂, ℓ₁, ℓ₂, rhs,linear_solver)

                # Accumulate gradient
                adjoint_grad_calc!(params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, vr0, vi05, vr, 
                                    lambdar0_nfrc, lambdar05_nfrc, lambdai_nfrc, lambdai0_nfrc, t0, dt,splinepar, gr, gi, tr_adj) 
                axpy!(gamma[q]*dt,tr_adj,infidelgrad)
                
                # save for next stage
                copy!(lambdai0_nfrc,lambdai_nfrc)
                copy!(lambdar0_nfrc,lambdar_nfrc)    
            end

        end #for stages
    end # for step (backward time stepping loop)

    primObjGradPhase=0.0
    if pFidType == 3
        # gradient wrt the global phase
        rotTarg = exp(1im*params.globalPhase)*(vtargetr + im*vtargeti)
        # grad wrt the global phase
        primObjGradPhase = - tracefidreal(vfinalr, vfinali, real(im.*rotTarg), imag(im.*rotTarg))
        # all components of the gradient
        totalgrad = zeros(Psize+1) # allocate array to return the gradient
        totalgrad[1:Psize] = gradobjfadj[:]
        totalgrad[Psize+1] = primObjGradPhase
        # totalgrad = zeros(Psize) # allocate array to return the gradient
        # totalgrad[:] = gradobjfadj[:]
    else
        totalgrad = zeros(Psize) # allocate array to return the gradient
        totalgrad[:] = gradobjfadj[:]
    end

    if params.objFuncType != 1
        leakgrad = zeros(size(totalgrad));
        if pFidType == 3 
            push!(infidelgrad,primObjGradPhase) 
        end
        leakgrad .= totalgrad - infidelgrad
    else
        #This is needed because when params.objFuncType == 1, 
        #We assume that the infidelgrad stores the totalgrad in ipopt interface
        infidelgrad = totalgrad 
    end
   
end # if evaladjoint

if verbose
    tikhonovpenalty = tikhonov_pen(pcof, params)

    println("Total objective func: ", objfv+tikhonovpenalty)
    println("Primary objective func: ", primaryobjf, " Guard state penalty: ", secondaryobjf, " Tikhonov penalty: ", tikhonovpenalty)
    if evaladjoint
        println("Norm of adjoint gradient = ", norm(gradobjfadj))

        if kpar <= Psize
            dfdp = dfdp + gr[kpar] # add in the tikhonov term

            println("Forward integration of total gradient[kpar=", kpar, "]: ", dfdp);
            println("Adjoint integration of total gradient[kpar=", kpar, "]: ", gradobjfadj[kpar]);
            println("\tAbsolute Error in gradients is : ", abs(dfdp - gradobjfadj[kpar]))
            println("\tRelative Error in gradients is : ", abs((dfdp - gradobjfadj[kpar])/norm(gradobjfadj)))
            println("\tPrimary grad = ", primaryobjgrad, " Tikhonov penalty grad = ", gr[kpar], " Guard state grad = ", dfdp - gr[kpar] - primaryobjgrad )
        else
            println("The gradient with respect to the phase angle is computed analytically and not by solving the adjoint equation")
        end
        if pFidType == 3 || pFidType == 4
            println("\tPrimary grad wrt phase = ", primObjGradPhase)
        end
    end
    
    nlast = 1 + nsteps
    println("Unitary test 1, error in length of propagated state vectors:")
    println("Col |   (1 - |psi|)")
    Vnrm ::Float64 = 0.0
    for q in 1:N
        Vnrm = usaver[:,q,nlast]' * usaver[:,q,nlast] + usavei[:,q,nlast]' * usavei[:,q,nlast]
        Vnrm = sqrt(Vnrm)
        println("  ", q, " |  ", 1.0 - Vnrm)
    end

    # output primary objective function (infidelity at final time)
    fidelityrot = tracefidcomplex(vfinalr, vfinali, vtargetr, vtargeti) # vfinali = -vi
    mfidelityrot = abs(fidelityrot)^2
    println("Final trace infidelity = ", 1.0 - mfidelityrot, " trace fidelity = ", mfidelityrot)
    if params.pFidType == 3
        println(" global phase = ",  params.globalPhase)
    end
    
    if params.usingPriorCoeffs
    println("Relative difference from prior: || pcof-prior || / || pcof || = ", norm(pcof - params.priorCoeffs) / norm(pcof) )
    end
    
    
    # Also output L2 norm of last energy level
    if Ntot>N
        #       normlastguard = zeros(N)
        forbLev = identify_forbidden_levels(params)
        maxLev = zeros(Ntot)
        for lev in 1:Ntot
            maxpop = zeros(N)
            if forbLev[lev]
                for q in 1:N
                    maxpop[q] = maximum( abs.(usaver[lev, q, :]).^2 + abs.(usavei[lev, q, :]).^2 )
                end
                maxLev[lev] = maximum(maxpop)
                println("Row = ", lev, " is a forbidden level, max population = ", maxLev[lev])
            end #if
        end
        println("Max population over all forbidden levels = ", maximum(maxLev))
    else
        println("No forbidden levels in this simulation");
    end
    
end #if verbose




# return to calling routine (the order of the return arguments is somewhat inconsistent. At least add a description in the docs)
if verbose && evaladjoint
    return objfv, totalgrad, usaver+1im*usavei, mfidelityrot, dfdp, wr1 - 1im*wi
elseif verbose
    println("Returning from traceobjgrad with objfv, unitary history, fidelity")
    return objfv, usaver+1im*usavei, mfidelityrot
elseif evaladjoint
    return objfv, totalgrad, primaryobjf, secondaryobjf, traceInfidelity, infidelgrad, leakgrad
else
    return objfv, primaryobjf, secondaryobjf
end #if
end

"""
    change_target!(params, new_Utarget)

Update the unitary target in the objparams object. 
 
# Arguments
- `param::objparams`: Object holding the problem definition
- `new_Utarget::Array{ComplexF64,2}`: New unitary target as a two-dimensional complex-valued array (matrix) of dimension Ntot x N
"""
function change_target!(params::objparams, new_Utarget::Array{ComplexF64,2} )
    Ntot = params.N + params.Nguard
    tz = ( Ntot, params.N )
    # Check size of new_Utarget
    @assert( size(new_Utarget) == tz)
    #println("change_target: Passed size compatibility test")
    my_target = copy(new_Utarget) # make a copy to be safe
    params.Utarget_r = real(my_target)
    params.Utarget_i = imag(my_target)
    if params.sv_type == 1
        params.dVds_r = real(my_target)
        params.dVds_i = imag(my_target)
    end
end

"""
    set_adjoint_Sv_type!(params, new_sv_type)

For continuation only: update the sv_type in the objparams object.  
 
# Arguments
- `param::objparams`: Object holding the problem definition
- `new_sv_type:: Int64`: New value for sv_type. Must be 1, 2, or 3.
"""
function set_adjoint_Sv_type!(params::objparams, new_sv_type::Int64 = 1)
    @assert( new_sv_type == 1 || new_sv_type == 2 || new_sv_type == 3)
    #println("change_target: Passed size compatibility test")
    params.sv_type = new_sv_type
end

function setup_prior!(params::objparams, priorFile::String)

    # read a prior parameter vector from a JLD2 file, assume that the number of parameters is
    # compatible between the current and prior parameter vector
    
    prior_pcof = load(priorFile, "pcof")
    println("Length of prior_pcof = ", length(prior_pcof) )

    params.priorCoeffs = prior_pcof
    params.usingPriorCoeffs = true
end

"""
    wmat = wmatsetup(Ne, Ng[, msb_order])

Build the default positive semi-definite weighting matrix W to calculate the 
leakage into higher energy forbidden states
 
# Arguments
- `Ne::Array{Int64,1}`: Number of essential energy levels for each subsystem
- `Ng::Array{Int64,1}`: Number of guard energy levels for each subsystem
- `msb_order::Bool`: Ordering of the subsystems within the state vector (default is true)
"""
function wmatsetup(Ne::Array{Int64,1}, Ng::Array{Int64,1}, msb_order::Bool = true)
    Nt = Ne + Ng
    Ndim = length(Ne)
    @assert(Ndim == 1 || Ndim == 2 || Ndim ==3)
    
    Ntot = prod(Nt)
    w = zeros(Ntot)
    coeff = 1.0

    # reset temp variables
    temp = zeros(length(Ne))

    if sum(Ng) > 0
        nForb = 0 # number of states with the highest index in at least one dimension
        
        if msb_order # Classical Juqbox ordering
            if Ndim == 1
                fact = 0.1
                for q in 0:Ng[1]-1
                    w[Ntot-q] = fact^q
                end
                nForb = 1
                coeff = 1.0
            elseif Ndim == 2
                fact = 1e-3 # for more emphasis on the "forbidden" states. Old value: 0.1
                q = 0 # element in the array 'w'

                for i2 = 1:Nt[2]
                    for i1 = 1:Nt[1]
                        q += 1
                        # initialize temp variables
                        temp[1] = 0.0
                        temp[2] = 0.0
                        if i1 <= Ne[1] && i2 <= Ne[2]
                            w[q] = 0.0
                        else
                            # determine and assign the largest penalty
                            if i1 > Ne[1]   #only included if at a guard level
                                temp[1] = fact^(Nt[1]-i1)
                            end
                            if i2 > Ne[2]   #only included if at a guard level
                                temp[2] = fact^(Nt[2]-i2)
                            end
                            if i1 == Nt[1] || i2 == Nt[2]
                                nForb += 1 
                            end

                            forbFact=1.0
                            w[q] = forbFact*maximum(temp)
            
                        end # if guard level
                    end # for i1
                end # for i2

                # normalize by the number of entries with w=1
                coeff = 1.0/nForb # was 1/nForb
            elseif Ndim == 3
                fact = 1e-3 #  0.1 # for more emphasis on the "forbidden" states. Old value: 0.1
                nForb = 0 # number of states with the highest index in at least one dimension
                q = 0
                for i3 = 1:Nt[3]
                    for i2 = 1:Nt[2]
                        for i1 = 1:Nt[1]
                            q += 1
                            # initialize temp variables
                            temp1 = 0.0
                            temp2 = 0.0
                            temp3 = 0.0
                            if i1 <= Ne[1] && i2 <= Ne[2] && i3 <= Ne[3]
                                w[q] = 0.0
                            else
                                # determine and assign the largest penalty
                                if i1 > Ne[1]   #only included if at a guard level
                                    temp1 = fact^(Nt[1]-i1)
                                end
                                if i2 > Ne[2]   #only included if at a guard level
                                    temp2 = fact^(Nt[2]-i2)
                                end
                                if i3 > Ne[3]   #only included if at a guard level
                                    temp3 = fact^(Nt[3]-i3)
                                end

                                forbFact=1.0
                                w[q] = forbFact*max(temp1, temp2, temp3)

                                if i1 == Nt[1] || i2 == Nt[2] || i3 == Nt[3]
                                    nForb += 1
                                end

                            end # if
                        end # for
                    end # for
                end # for

                # normalize by the number of entries with w=1
                coeff = 10.0/nForb # was 1/nForb
            end # if ndim == 3
        else # msb_order = false
            if Ndim == 1
                fact = 0.1
                for q in 0:Ng[1]-1
                    w[Ntot-q] = fact^q
                end
                nForb = 1
                coeff = 1.0
            elseif Ndim == 2
                fact = 1e-3 # for more emphasis on the "forbidden" states. Old value: 0.1
                q = 0 # element in the array 'w'

                for i1 = 1:Nt[1]
                    for i2 = 1:Nt[2]
                        q += 1
                        # initialize temp variables
                        temp[1] = 0.0
                        temp[2] = 0.0
                        if i1 <= Ne[1] && i2 <= Ne[2]
                            w[q] = 0.0
                        else
                            # determine and assign the largest penalty
                            if i1 > Ne[1]   #only included if at a guard level
                                temp[1] = fact^(Nt[1]-i1)
                            end
                            if i2 > Ne[2]   #only included if at a guard level
                                temp[2] = fact^(Nt[2]-i2)
                            end
                            if i1 == Nt[1] || i2 == Nt[2]
                                nForb += 1 
                            end

                            forbFact=1.0
                            w[q] = forbFact*maximum(temp)
            
                        end # if guard level
                    end # for i1
                end # for i2

                # normalize by the number of entries with w=1
                coeff = 1.0/nForb # was 1/nForb
            elseif Ndim == 3
                fact = 1e-3 #  0.1 # for more emphasis on the "forbidden" states. Old value: 0.1
                nForb = 0 # number of states with the highest index in at least one dimension
                q = 0

                for i1 = 1:Nt[1]
                    for i2 = 1:Nt[2]
                        for i3 = 1:Nt[3]
                            q += 1
                            # initialize temp variables
                            temp1 = 0.0
                            temp2 = 0.0
                            temp3 = 0.0
                            if i1 <= Ne[1] && i2 <= Ne[2] && i3 <= Ne[3]
                                w[q] = 0.0
                            else
                                # determine and assign the largest penalty
                                if i1 > Ne[1]   #only included if at a guard level
                                    temp1 = fact^(Nt[1]-i1)
                                end
                                if i2 > Ne[2]   #only included if at a guard level
                                    temp2 = fact^(Nt[2]-i2)
                                end
                                if i3 > Ne[3]   #only included if at a guard level
                                    temp3 = fact^(Nt[3]-i3)
                                end

                                forbFact=1.0
                                w[q] = forbFact*max(temp1, temp2, temp3)

                                if i1 == Nt[1] || i2 == Nt[2] || i3 == Nt[3]
                                    nForb += 1
                                end

                            end # if
                        end # for
                    end # for
                end # for

                # normalize by the number of entries with w=1
                coeff = 10.0/nForb # was 1/nForb
            end # if ndim == 3
        end # lsb ordering
        # println("wmatsetup: Number of forbidden states = ", nForb, " scaling coeff = ", coeff)
    end # if sum(Ng) > 0

    wmat = coeff * Diagonal(w) # turn vector into diagonal matrix
    return wmat
end

# Matrices for the Hamiltonian in rotation frame
"""
    omega1[, omega2, omega3] = setup_rotmatrices(Ne, Ng, fund_freq)

Build diagonal rotation matrices based on the |0⟩to |1⟩ transition frequency in each sub-system.

 
# Arguments
- `Ne::Array{Int64,1}`: Number of essential energy levels for each subsystem
- `Ng::Array{Int64,1}`: Number of guard energy levels for each subsystem
- `fund_freq::Array{Float64}`: Transitions frequency [GHz] for each subsystem
"""
function setup_rotmatrices(Ne::Array{Int64,1}, Ng::Array{Int64,1}, fund_freq::Array{Float64})
    Nosc = length(Ne)
    @assert(Nosc >= 1 && Nosc <=3)

    if Nosc==1 # 1 oscillator
        Ntot = Ne[1] + Ng[1]
        omega1 = 2*pi*fund_freq[1]*Array(collect(0:Ntot-1))
        return omega1
    elseif Nosc==2 # 2 oscillators
        Nt1 = Ne[1] + Ng[1]
        Nt2 = Ne[2] + Ng[2]
# Note: The ket psi = ji> = e_j kron e_i.
# We order the elements in the vector psi such that i varies the fastest with i in [1,Nt1] and j in [1,Nt2]
# The matrix amat = I kron a1 acts on alpha in psi = beta kron alpha
# The matrix bmat = a2 kron I acts on beta in psi = beta kron alpha
        I1 = Array{Float64, 2}(I, Nt1, Nt1)
        I2 = Array{Float64, 2}(I, Nt2, Nt2)

        num1 = Diagonal(collect(0:Nt1-1))
        num2 = Diagonal(collect(0:Nt2-1))

        Na = Diagonal(kron(I2, num1))
        Nb = Diagonal(kron(num2, I1))

        # rotation matrices
        wa = diag(Na)
        wb = diag(Nb)

        omega1 = 2*pi*fund_freq[1]*wa
        omega2 = 2*pi*fund_freq[2]*wb 

        return omega1, omega2
    elseif Nosc==3 # 3 coupled quantum systems
        Nt1 = Ne[1] + Ng[1]
        Nt2 = Ne[2] + Ng[2]
        Nt3 = Ne[3] + Ng[3]
# Note: The ket psi = kji> = e_k kron (e_j kron e_i).
# We order the elements in the vector psi such that i varies the fastest with i in [1,Nt1], j in [1,Nt2] and k in [1,Nt3]
# The matrix amat = I kron I kron a1 acts on alpha in psi = gamma kron beta kron alpha
# The matrix bmat = I kron a2 kron I acts on beta in psi = gamma kron beta kron alpha
# The matrix cmat = a3 kron I kron I acts on gamma in psi = gamma kron beta kron alpha
        I1 = Array{Float64, 2}(I, Nt1, Nt1)
        I2 = Array{Float64, 2}(I, Nt2, Nt2)
        I3 = Array{Float64, 2}(I, Nt3, Nt3)

        num1 = Diagonal(collect(0:Nt1-1))
        num2 = Diagonal(collect(0:Nt2-1))
        num3 = Diagonal(collect(0:Nt3-1))

        N1 = Diagonal(kron(I3, I2, num1))
        N2 = Diagonal(kron(I3, num2, I1))
        N3 = Diagonal(kron(num3, I2, I1))

        # rotation matrices
        w1 = diag(N1)
        w2 = diag(N2)
        w3 = diag(N3)

        omega1 = 2*pi*fund_freq[1]*w1
        omega2 = 2*pi*fund_freq[2]*w2 
        omega3 = 2*pi*fund_freq[3]*w3 

        return omega1, omega2, omega3 # Put them in an Array to make the return type uniform?
    end
end

#------------------------------------------------------------
"""
    zero_start_end!(params, D1, minCoeff, maxCoeff)

Force the control functions to start and end at zero by setting zero bounds for the first two and last 
two parameters in each B-spline segment.
 
# Arguments
- `params:: objparams`: Struct containing problem definition.
- `D1:: Int64`: Number of basis functions in each segment.
- `minCoeff:: Vector{Float64}`: Lower parameter bounds to be modified
- `maxCoeff:: Vector{Float64}`: Upper parameter bounds to be modified
"""
function zero_start_end!(params::objparams, D1:: Int64, minCoeff:: Array{Float64,1}, maxCoeff:: Array{Float64,1} )
    @assert(D1 >= 5) # Need at least 5 parameters per B-spline segment
    
    Nfreq = params.Nfreq
    Ncoupled = params.Ncoupled
    Nunc = params.Nunc
    # nCoeff = 2*Ncoupled*Nfreq*D1
    NfreqTot = sum(Nfreq) 
    nCoeff = 2*D1*NfreqTot

#    @printf("Ncoupled = %d, Nfreq = %d, D1 = %d, nCoeff = %d\n", Ncoupled, Nfreq, D1, nCoeff)
    baseOffset = 0
    for c in 1:Ncoupled+Nunc  # We assume that either Nunc = 0 or Ncoupled = 0
        for f in 1:Nfreq[c]
            for q in 0:1
                # offset1 = 2*(c-1)*Nfreq*D1 + (f-1)*2*D1 + q*D1
                offset1 = baseOffset + (f-1)*2*D1 + q*D1
                # start
                minCoeff[ offset1 + 1] = 0.0
                minCoeff[ offset1 + 2] = 0.0
                maxCoeff[ offset1 + 1] = 0.0
                maxCoeff[ offset1 + 2] = 0.0
                # end
                offset2 = offset1+D1
                minCoeff[ offset2-1] = 0.0
                minCoeff[ offset2] = 0.0
                maxCoeff[ offset2-1] = 0.0
                maxCoeff[ offset2 ] = 0.0
            end
        end # for f
        baseOffset += 2*D1*Nfreq[c]
    end
end

#------------------------------------------------------------
"""
    zero_start_end!(Nctrl, Nfreq, D1, minCoeff, maxCoeff)

Force the control functions to start and end at zero by setting zero bounds for the first two and last 
two parameters in each B-spline segment.
 
# Arguments
- `Nctrl:: Int64`: Number of control Hamiltonians.
- `Nfreq:: Vector{Int64}`: Vector holding the number of carrier frequencies for each control
- `D1:: Int64`: Number of basis functions in each segment.
- `minCoeff:: Vector{Float64}`: Lower parameter bounds to be modified
- `maxCoeff:: Vector{Float64}`: Upper parameter bounds to be modified
"""
function zero_start_end!(Nctrl::Int64, Nfreq::Vector{Int64}, D1:: Int64, minCoeff:: Array{Float64,1}, maxCoeff:: Array{Float64,1} )
    @assert(D1 >= 5) # Need at least 5 parameters per B-spline segment
    @assert(Nctrl == length(Nfreq))
    @assert(sum(Nfreq) >= 1)

    #println("zero_start_end!: Nctrl = ", Nctrl, " Nfreq = ", Nfreq)

    baseOffset = 0
    for c in 1:Nctrl  # We assume that either Nunc = 0 or Ncoupled = 0
        for f in 1:Nfreq[c]
            for q in 0:1
                # offset1 = 2*(c-1)*Nfreq*D1 + (f-1)*2*D1 + q*D1
                offset1 = baseOffset + (f-1)*2*D1 + q*D1
                # start
                minCoeff[ offset1 + 1] = 0.0
                minCoeff[ offset1 + 2] = 0.0
                maxCoeff[ offset1 + 1] = 0.0
                maxCoeff[ offset1 + 2] = 0.0
                # end
                offset2 = offset1+D1
                minCoeff[ offset2-1] = 0.0
                minCoeff[ offset2] = 0.0
                maxCoeff[ offset2-1] = 0.0
                maxCoeff[ offset2 ] = 0.0
            end
        end # for f
        baseOffset += 2*D1*Nfreq[c]        
    end
end

#------------------------------------------------------------
"""
    minCoeff, maxCoeff = assign_thresholds_ctrl_freq(params, D1, maxAmp)

Build vector of parameter min/max constraints that can depend on the control function and carrier wave frequency, 
with `minCoeff = -maxCoeff`.
 
# Arguments
- `params:: objparams`: Struct containing problem definition.
- `D1:: Int64`: Number of basis functions in each segment.
- `maxAmp:: Matrix{Float64}`: `maxAmp[c,f]` is the maximum parameter value for ctrl `c` and frequency `f`
"""
function assign_thresholds_ctrl_freq(params::objparams, D1:: Int64, maxAmp:: Vector{Vector{Float64}})
    Nfreq = params.Nfreq
    Ncoupled = params.Ncoupled
    Nunc = params.Nunc
    @assert(Nunc == 0)

    NfreqTot = sum(Nfreq) 
    nCoeff = 2*D1*NfreqTot
    #nCoeff = 2*(Ncoupled+Nunc)*Nfreq*D1
    minCoeff = zeros(nCoeff) # Initialize storage
    maxCoeff = zeros(nCoeff)

#    @printf("Ncoupled = %d, Nfreq = %d, D1 = %d, nCoeff = %d\n", Ncoupled, Nfreq, D1, nCoeff)
    baseOffset = 0
    for c in 1:Ncoupled  # We assume that either Nunc = 0 or Ncoupled = 0
        for f in 1:Nfreq[c]
            # offset1 = 2*(c-1)*Nfreq*D1 + (f-1)*2*D1
            offset1 = baseOffset + (f-1)*2*D1
            minCoeff[offset1 + 1:offset1+2*D1] .= -maxAmp[c][f] # same for p(t) and q(t)
            maxCoeff[offset1 + 1:offset1+2*D1] .= maxAmp[c][f]
        end
        baseOffset += 2*D1*Nfreq[c]
    end
    return minCoeff, maxCoeff
end

# """
#     minCoeff, maxCoeff = assign_thresholds_freq(maxAmp, Ncoupled, Nfreq, D1)

# Build vector of frequency dependent min/max parameter constraints, with `minCoeff = -maxCoeff`, when
# there are no uncoupled control functions.
 
# # Arguments
# - `maxAmp::Array{Float64,1}`: Maximum parameter value for each frequency
# - `Ncoupled::Int64`: Number of coupled controls in the simulation
# - `Nfreq::Vector{Int64}`: Number of carrier wave frequencies used in the controls
# - `D1:: Int64`: Number of basis functions in each control function
# """
# function assign_thresholds_freq(maxAmp::Array{Float64,1}, Ncoupled::Int64, Nfreq::Vector{Int64}, D1::Int64)
#     NfreqTot = sum(Nfreq) 
#     # nCoeff = 2*Ncoupled*Nfreq*D1
#     nCoeff = 2*D1*NfreqTot
#     minCoeff = zeros(nCoeff) # Initialize storage
#     maxCoeff = zeros(nCoeff)
#     baseOffset = 0
# #    @printf("Ncoupled = %d, Nfreq = %d, D1 = %d, nCoeff = %d\n", Ncoupled, Nfreq, D1, nCoeff)
#     for c in 1:Ncoupled
#         for f in 1:Nfreq[c]
#             #offset1 = 2*(c-1)*Nfreq*D1 + (f-1)*2*D1
#             offset1 = baseOffset + (f-1)*2*D1
#             minCoeff[offset1 + 1:offset1+2*D1] .= -maxAmp[f] # same for p(t) and q(t)
#             maxCoeff[offset1 + 1:offset1+2*D1] .= maxAmp[f]
#         end
#         baseOffset += 2*D1*Nfreq[c]
#     end
#     return minCoeff, maxCoeff
# end

"""
    minCoeff, maxCoeff = assign_thresholds(params, D1, maxAmp)

Build vector of frequency independent min/max parameter constraints for each control function. Here, `minCoeff = -maxCoeff`.
 
# Arguments
- `params:: objparams`: Struct containing problem definition.
- `D1:: Int64`: Number of basis functions in each segment.
- `maxAmp:: Vector{Float64}`: `maxAmp[c]` is the maximum for ctrl function number `c`. Same bounds for p & q.
"""
function assign_thresholds(params::objparams, D1::Int64, maxAmp::Vector{Float64})
    Nfreq = params.Nfreq
    Ncoupled = params.Ncoupled
    Nunc = params.Nunc
    @assert(Nunc == 0)

    NfreqTot = params.NfreqTot
    nCoeff = 2*D1*NfreqTot
    #nCoeff = 2*(Ncoupled+Nunc)*Nfreq*D1
    minCoeff = zeros(nCoeff) # Initialize storage
    maxCoeff = zeros(nCoeff)

#    @printf("Ncoupled = %d, Nfreq = %d, D1 = %d, nCoeff = %d\n", Ncoupled, Nfreq, D1, nCoeff)
    baseOffset = 0
    for c in 1:Ncoupled  # We assume that either Nunc = 0 or Ncoupled = 0
        for f in 1:Nfreq[c]
            # offset1 = 2*(c-1)*Nfreq*D1 + (f-1)*2*D1
            offset1 = baseOffset + (f-1)*2*D1
            bound = maxAmp[c]/(sqrt(2)*Nfreq[c]) # Divide bounds equally between the carrier frequencies for each control
            minCoeff[offset1 + 1:offset1+2*D1] .= -bound # same for p(t) and q(t)
            maxCoeff[offset1 + 1:offset1+2*D1] .= bound
        end
        baseOffset += 2*D1*Nfreq[c]
    end
    return minCoeff, maxCoeff
end

# Initialize the adjoint variables in-place.
@inline function init_adjoint!(pFidType::Int64, globalPhase::Float64, N::Int64, scomplex0::Complex{Float64}, lambdar::Array{Float64,M},
                               lambdar0::Array{Float64,M}, lambdar05::Array{Float64,M},lambdai::Array{Float64,M}, lambdai0::Array{Float64,M},
                               vtargetr::Array{Float64,M}, vtargeti::Array{Float64,M}) where M
    if pFidType == 2
        rs = real(scomplex0)
        is = imag(scomplex0)
        for j = 1:size(lambdar,2)
            @fastmath @inbounds @simd for i = 1:size(lambdar,1)
                rtmp = (rs*vtargetr[i,j] + is*vtargeti[i,j])/N
                lambdar[i,j] = rtmp
                lambdar0[i,j] = rtmp
                lambdar05[i,j] = rtmp
                itmp = (is*vtargetr[i,j] - rs*vtargeti[i,j])/N
                lambdai[i,j] = itmp
                lambdai0[i,j] = itmp
            end
        end
    elseif pFidType == 3 || pFidType == 4
        rotTarg = exp(1im*globalPhase)*(vtargetr + im*vtargeti)
        rtargetr = real(rotTarg)
        rtargeti = imag(rotTarg)
        for j = 1:size(lambdar,2)
            @fastmath @inbounds @simd for i = 1:size(lambdar,1)
                rtmp = 0.5*rtargetr[i,j]/N
                lambdar[i,j] = rtmp
                lambdar0[i,j] = rtmp
                lambdar05[i,j] = rtmp
                itmp = - 0.5*rtargeti[i,j]/N
                lambdai[i,j] = itmp
                lambdai0[i,j] = itmp
            end
        end
    end
end

@inline function tracefidabs2(ur::Array{Float64,2}, ui::Array{Float64,2}, vtargetr::Array{Float64,2}, vtargeti::Array{Float64,2})
    N = size(vtargetr,2)
    # NOTE: ur = Re(v), ui = Im(v)

    # NOTE: this routine computes | tr((ur + i ui).dag * (vtr + i vtr)) | ^2 / N^2
    fidelity = (trace_operator(ur,vtargetr,ui,vtargeti)/N)^2
    fidelity += (trace_operator(ur,vtargeti,-ui,vtargetr)/N)^2
end

        
@inline function tracefidreal(ur::Array{Float64,2}, ui::Array{Float64,2}, vtargetr::Array{Float64,2}, vtargeti::Array{Float64,2})
    N = size(vtargetr,2)
    # fidreal = tr(ur' * vtargetr .+ ui' * vtargeti)/N; # NOTE: ui = Im(v)
    fidreal = trace_operator(ur,vtargetr,ui,vtargeti)
    fidreal = fidreal/N
end

@inline function tracefidcomplex(ur::Array{Float64,2}, ui::Array{Float64,2}, vtargetr::Array{Float64,2}, vtargeti::Array{Float64,2})
    N = size(vtargetr,2)
    # fid_cmplx = tr(ur' * vtargetr .+ ui' * vtargeti)/N + 1im*tr(ur' * vtargeti .- ui' * vtargetr)/N;
    fid_cmplx = trace_operator(ur,vtargetr,ui,vtargeti)
    fid_cmplx += 1im*(trace_operator(ur,vtargeti,-ui,vtargetr))
    fid_cmplx = fid_cmplx/N
end


# Helper function to evaluate tr(A'*B) without the need for full matrix multiplication
@inline function trace_operator(A::Array{Float64,2},B::Array{Float64,2})
    trace = 0.0

    for j = 1:size(A,2)
        @fastmath @inbounds @simd for i = 1:size(A,1)
            trace += A[i,j]*B[i,j]
        end
    end

    return trace
end

# Helper function to evaluate tr(A'*B + C'*D) without the need for full matrix multiplication
@inline function trace_operator(A::Array{Float64,2},B::Array{Float64,2},C::Array{Float64,2},D::Array{Float64,2})
    trace = 0.0

    for j = 1:size(A,2)
        @fastmath @inbounds @simd for i = 1:size(A,1)
            trace += A[i,j]*B[i,j] + C[i,j]*D[i,j]
        end
    end

    return trace
end

# Helper function to evaluate tr(A'*B*C)
@inline function adjoint_trace_operator!(A::Array{Float64,2},B::Array{Float64,2},C::Array{Float64,2})
    trace = 0.0

    # Without storage
    for j = 1:size(C,2)
        for i = 1:size(B,1)
            Btmp = 0.0
            @fastmath @inbounds @simd for k = 1:size(B,2)
                Btmp += B[i,k]*C[k,j]
            end
            trace += A[i,j]*Btmp
        end
    end



    return trace
end

# Helper function to evaluate tr(A'*B*C) (sparse version)
# FG: Note here that we should compute A'*B first since B is stored by columns. Then accumulate the trace
@inline function adjoint_trace_operator!(A::Array{Float64,2},B::SparseMatrixCSC{Float64,Int64},C::Array{Float64,2})

    # Here we use that trace is invariant to taking the transpose
    trace = 0.0
    for j = 1:size(A,2)
        for i = 1:size(A,1)
            len = B.colptr[i+1]-B.colptr[i]
            ind = B.colptr[i]
            mat_temp = 0.0
            @fastmath @inbounds @simd for k = 1:len
                tmp = ind + k -1
                row = B.rowval[tmp]
                mat_temp += A[row,j]*B.nzval[tmp]
            end
            trace += mat_temp*C[i,j]
        end
    end
    return trace

end

# anders version
@inline function penalf2a(vr::Array{Float64,N}, vi::Array{Float64,N},  wmat::Diagonal{Float64,Array{Float64,1}}) where N
    # f = tr( vr' * wmat * vr + vi' * wmat * vi);
    f = 0.0
    for j = 1:size(vr,2)
        @fastmath @inbounds @simd for i=1:size(vr,1)
            # f+= (vr[i,j]^2 + vi[i,j]^2)*wmat[i,i] 
            f+= (vr[i,j]^2 + 2.0*vi[i,j]^2)*wmat[i,i] # Account for the √2 in vi
        end
    end
    return f
end

# Version assuming weight matrix is not necessarily diagonal
@inline function penalf2a(vr::Array{Float64,N}, vi::Array{Float64,N},  wmat::Array{Float64,N}) where N
    # f = tr( vr' * wmat * vr + vi' * wmat * vi);
    f = 0.0
    for i = 1:size(wmat,2)
        for j = 1:size(vr,2)
            vr_loc = vr[i,j]
            vi_loc = vi[i,j]
            @fastmath @inbounds @simd for k=1:size(vr,1)
                f+= vr_loc*wmat[i,k]*vr[k,j] + 2.0*vi_loc*wmat[i,k]*vi[k,j]
            end
        end
    end
    return f
end

# FG : This is to collect only the trap rule portion of the objective function
@inline function penalf2aTrap(vr::Array{Float64,N}, wmat::Diagonal{Float64,Array{Float64,1}}) where N
    # f = tr( vr' * wmat * vr); #'
    f = 0.0
    for j = 1:size(vr,2)
        @fastmath @inbounds @simd for i=1:size(vr,1)
            f+= wmat[i,i]*vr[i,j]^2 
        end
    end
    return f
end

# Trap rule portion of the objective function, assumping weight matrix is not necessarily diagonal
@inline function penalf2aTrap(vr::Array{Float64,N}, wmat::Array{Float64,N}) where N
    # f = tr( vr' * wmat * vr); #'
    f = 0.0
    for i = 1:size(wmat,2)
        for j = 1:size(vr,2)
            v_loc = vr[i,j]
            @fastmath @inbounds @simd for k=1:size(vr,1)
                f+= v_loc*wmat[i,k]*vr[k,j] 
            end
        end
    end
    return f
end

# Compute weighting term due to imaginary component of forbidden state
@inline function penalf2imag(vr::Array{Float64,N}, vi::Array{Float64,N},  wmat::Array{Float64,2}) where N
    adjoint_trace_operator!(vi,wmat,vr)
end

# Compute imaginary weighting matrix component, do nothing if using usual weighting
@inline function penalf2imag(vr::Array{Float64,N}, vi::Array{Float64,N},  wmat::Diagonal{Float64,Array{Float64,1}}) where N
    return 0.0
end

# # Collect only the trap rule portion of the objective function. Here we use the input array of 
# # forbidden states
# @inline function penalf2aTrap(vr::Array{Float64,N}, forb_states::Array{Float64,2}) where N
#     # f = tr( vr' * wmat * vr); #'
#     f = 0.0
#     for j = 1:size(vr,2)
#         @fastmath @inbounds @simd for i=1:size(vr,1)
#             f+= wmat[i,i]*vr[i,j]^2 
#         end
#     end
#     return f
# end

function penalf2adj(vr::Array{Float64,N}, wmat::Diagonal{Float64,Array{Float64,1}}) where N
    f = wmat * vr;
    return f
end

@inline function penalf2adj!(vr::Array{Float64,N}, wmat::Diagonal{Float64,Array{Float64,1}}, tinv:: Float64, result::Array{Float64,N}) where N
    # result .= wmat*vr
    for j = 1:size(vr,2)
        @fastmath @inbounds @simd for i = 1:size(vr,1)
            result[i,j] = tinv*wmat[i,i]*vr[i,j]
        end
    end
end



@inline function penalf2grad(vr::Array{Float64,N}, vi::Array{Float64,N}, wr::Array{Float64,N}, wi::Array{Float64,N},  wmat::Diagonal{Float64,Array{Float64,1}}) where N
    # f = tr( wr' * wmat * vr ) + tr( wi' * wmat * vi);
    f = 0.0
    for j = 1:size(vr,2)
        @fastmath @inbounds @simd for i=1:size(vr,1)
            f+= (wr[i,j]*vr[i,j] + wi[i,j]*vi[i,j])*wmat[i,i]
        end
    end
    return f
end

# Accumulate penalty term without assuming weighting matrix is diagonal
@inline function penalf2grad(vr::Array{Float64,N}, vi::Array{Float64,N}, wr::Array{Float64,N}, wi::Array{Float64,N},  wmat::Array{Float64,N}) where N
    # f = tr( wr' * wmat * vr ) + tr( wi' * wmat * vi);
    f = 0.0
    for i = 1:size(wmat,2)
        for j = 1:size(vr,2)
            wr_loc = wr[i,j]
            wi_loc = wi[i,j]
            @fastmath @inbounds @simd for k=1:size(vr,1)
                f+= wr_loc*wmat[i,k]*vr[k,j] + wi_loc*wmat[i,k]*vi[k,j] 
            end
        end
    end    
    return f
end

@inline function tikhonov_pen(pcof::Array{Float64,1}, params ::objparams)
    Npar = size(pcof,1)

    # Tikhonov regularization
    if params.usingPriorCoeffs
        penalty0 = dot(pcof-params.priorCoeffs, pcof-params.priorCoeffs)
    else
        penalty0 = dot(pcof,pcof)
    end

    # Nseg = (2*params.Ncoupled+params.Nunc)*params.Nfreq
    # D1 = div(Npar,Nseg)
    # penalty1 = 0.0
    # This makes little sense when using smooth B-splines
    # for i = 1:Nseg
    #     offset = (i-1)*D1
    #     @fastmath @inbounds @simd for j = offset+2:offset+D1
    #         penalty1 += (pcof[j] - pcof[j-1])^2
    #     end
    # end

#    penalty = (params.tik0 * penalty0 + params.tik1 * penalty1)/Npar;

    penalty = (params.tik0 * penalty0)/Npar;
                                
    return penalty
end

@inline function tikhonov_grad!(pcof::Array{Float64,1}, params::objparams, pengrad::Array{Float64,1})  
    Npar = size(pcof,1)
    iNpar = 1.0/Npar

    # Tikhonov regularization
    pengrad[:] .= 0.0
    
    # Nseg = (2*params.Ncoupled+params.Nunc)*params.Nfreq
    # D1 = div(Npar,Nseg)
    # make little sense with smooth B-splines
    # for i = 1:Nseg
    #     Nstart = (i-1)*D1 + 1
    #     Nend = i*D1
    #     pengrad[Nstart] = 2.0*(pcof[Nstart] - pcof[Nstart+1])
    #     @fastmath @inbounds @simd for j = Nstart+1:Nstart+D1-2
    #         pengrad[j] = 2.0*( -pcof[j-1] + 2.0*pcof[j] - pcof[j+1] );
    #     end
    #     pengrad[Nend] = 2.0*(pcof[Nend] - pcof[Nend-1] );
    # end

    
    if params.usingPriorCoeffs
        @fastmath @inbounds @simd for i = 1:Npar
            pengrad[i] += (2.0*params.tik0*iNpar)* ( pcof[i] - params.priorCoeffs[i] )
        end
    else
        @fastmath @inbounds @simd for i = 1:Npar
            #        pengrad[i] *= iNpar*params.tik1
            pengrad[i] += (2.0*params.tik0*pcof[i]*iNpar)
        end
    end

end


function KS!(K::Array{Float64,N}, S::Array{Float64,N}, t::Float64, Hsym_ops::Array{MyRealMatrix,1}, Hanti_ops::Array{MyRealMatrix, 1},
             Hunc_ops::Array{MyRealMatrix, 1}, Nunc::Int64, isSymm::BitArray{1}, splinepar::BsplineParams, H0::Array{Float64,N}, Rfreq::Array{Float64,1}) where N

    # Isn't the H0 matrix always 2-dimensional? Why do we need to declare it as N-dimensional? 

    Ncoupled = splinepar.Ncoupled

    copy!(K,H0)
    S .= 0.0
    for q=1:Ncoupled # Assumes that Hanti_ops has the same length as Hsym_ops
        qs = (q-1)*2
        qa = qs+1
        pt = controlfunc(t,splinepar, qs)
        qt = controlfunc(t,splinepar, qa)
        axpy!(pt,Hsym_ops[q],K)
        axpy!(qt,Hanti_ops[q],S)
    end

#    offset = 2*Ncoupled-1
    offset = 2*Ncoupled
    for q=1:splinepar.Nunc  # Will not work for splineparams object
        qs = offset + (q-1)*2
        qa = qs+1
        pt = controlfunc(t,splinepar, qs)
        qt = controlfunc(t,splinepar, qa)

#        ft = controlfunc(t,splinepar, offset+q)
        ft = 2*( pt*cos(2*pi*Rfreq[q]*t) - qt*sin(2*pi*Rfreq[q]*t) )
        if(isSymm[q])
            axpy!(ft,Hunc_ops[q],K)
        else
            axpy!(ft,Hunc_ops[q],S)
        end
    end
    
end

# Sparse version
@inline function KS!(K::SparseMatrixCSC{Float64,Int64}, S::SparseMatrixCSC{Float64,Int64}, t::Float64, Hsym_ops::Array{MyRealMatrix,1}, Hanti_ops::Array{MyRealMatrix, 1},
             Hunc_ops::Array{MyRealMatrix, 1}, Nunc::Int64, isSymm::BitArray{1}, splinepar::BsplineParams, H0::SparseMatrixCSC{Float64,Int64}, Rfreq::Array{Float64,1})
    
    Ncoupled = splinepar.Ncoupled
    K.nzval .= 0.0
    accumulate_matrix!(K, H0, 1.0)

    S.nzval .= 0.0
    for q=1:Ncoupled # Assumes that Hanti_ops has the same length as Hsym_ops
        qs = (q-1)*2
        qa = qs+1
        pt = controlfunc(t,splinepar, qs)
        qt = controlfunc(t,splinepar, qa)
        accumulate_matrix!(K, Hsym_ops[q], pt)
        accumulate_matrix!(S, Hanti_ops[q], qt)
    end

#    offset = 2*Ncoupled-1
    offset = 2*Ncoupled
    for q=1:splinepar.Nunc # Will not work for splineparams object
        qs = offset + (q-1)*2
        qa = qs+1
        pt = controlfunc(t,splinepar, qs)
        qt = controlfunc(t,splinepar, qa)

#        ft = controlfunc(t,splinepar, offset+q)
        ft = 2*( pt*cos(2*pi*Rfreq[q]*t) - qt*sin(2*pi*Rfreq[q]*t) )
        if(isSymm[q])
            accumulate_matrix!(K, Hunc_ops[q], ft)
        else
            accumulate_matrix!(S, Hunc_ops[q], ft)
        end
        
    end
end

# Compute A + f*B → A for sparse matrices (assumes A has proper sparsity pattern)
@inline function accumulate_matrix!(A::SparseMatrixCSC{Float64,Int64}, B::SparseMatrixCSC{Float64,Int64}, f::Float64)

    for j = 1:B.n
        len = B.colptr[j+1]-B.colptr[j]
        ind = B.colptr[j]
        ind2 = A.colptr[j]
        for k = 1:len
            tmp = ind + k -1
            row = B.rowval[tmp]
            A[row,j] += f*B.nzval[tmp]
        end
    end
end

# splinefunc determines if the params are for the control function p_k or q_k
# NEW ordering: p-func are even and q-func are odd
# 0 - p1(t) for x-drive * Hsym_ops[1]
# 1 - q1(t) for y-drive * Hanti_ops[1]
# 2 - p2(t) for x-drive * Hsym_ops[2]
# 3 - q2(t) for y-drive * Hanti_ops[2]
# function for computing the splines
@inline controlfunc(t::Float64,splinepar::splineparams, splinefunc::Int64) = bspline2(t,splinepar, splinefunc)
@inline controlfunc(t::Float64, bcparams::bcparams, splinefunc::Int64) = bcarrier2(t, bcparams, splinefunc)

# function for computing the spline gradients
@inline controlfuncgrad!(t::Float64, splinepar::splineparams, splinefunc::Int64) = gradbspline2(t, splinepar, splinefunc)
@inline controlfuncgrad!(t::Float64,  bcparams::bcparams, splinefunc::Int64, g::Array{Float64,1}) = gradbcarrier2!(t, bcparams, splinefunc,g)

function rotmatrices!(t::Float64, domega::Array{Float64,1},rr::Array{Float64,2},ri::Array{Float64,2})
 for I in 1:length(domega)
  rr[I,I] = cos(domega[I]*t)
  ri[I,I] = -sin(domega[I]*t)
 end
end

# Compute the forcing to the forward gradient using the stage values of the state equation.
# Note: Only used for verification of the gradient
# Note2: This is for Störmer-Verlet, so all the stage values for v are exactly the same.
@inline function fgradforce!(Hsym_ops::Array{MyRealMatrix,1}, Hanti_ops::Array{MyRealMatrix, 1}, 
                            Hunc_ops::Array{MyRealMatrix,1}, Nunc::Int64, isSymm::BitArray{1},
                            vr0::Array{Float64,2}, vi05::Array{Float64,2}, vr1::Array{Float64,2}, 
                            t::Float64, dt::Float64, splinepar::BsplineParams, kpar::Int64,
                            fr0::Array{Float64,2}, fr1::Array{Float64,2},
                            fi0::Array{Float64,2}, fi1::Array{Float64,2},
                             gr::Array{Float64,1}, gi::Array{Float64,1})

    Ncoupled = length(Hsym_ops) # Ncoupled equals the number of terms in the Hsym_ops array

    # NEW ordering of the controlfunctions (p1, q1, p2, q2, etc.)

    fr0 .= 0.0
    fi0 .= 0.0
    fr1 .= 0.0
    fi1 .= 0.0

    # if kpar > splinepar.Ncoeff # testing the gradient wrt global phase
    #     return
    # end

    gr.= 0.0
    gi.= 0.0

    for q=1:Ncoupled
        qs = (q-1)*2
        qa = qs + 1

        controlfuncgrad!(t, splinepar, qs, gr)
        controlfuncgrad!(t, splinepar, qa, gi)

        # fr0 .= fr0 .+ gi[kpar].*Hanti_ops[q]*vr0  .- gr[kpar].*Hsym_ops[q]*vi05
        mul!(fr0, Hsym_ops[q], vi05, -gr[kpar], 1.0)
        mul!(fr0, Hanti_ops[q], vr0, gi[kpar], 1.0)

        controlfuncgrad!(t+0.5*dt, splinepar, qs, gr)
        controlfuncgrad!(t+0.5*dt, splinepar, qa, gi)


        # fi0 .= fi0 .+ gr[kpar].*Hsym_ops[q]*vr0 .+ gi[kpar].*Hanti_ops[q]*vi05
        mul!(fi0, Hsym_ops[q], vr0, gr[kpar], 1.0)
        mul!(fi0, Hanti_ops[q], vi05, gi[kpar], 1.0)

        # fi1 .= fi1 .+ gr[kpar].*Hsym_ops[q]*vr1 .+ gi[kpar].*Hanti_ops[q]*vi05
        mul!(fi1, Hsym_ops[q], vr1, gr[kpar], 1.0)
        mul!(fi1, Hanti_ops[q], vi05, gi[kpar], 1.0)

        controlfuncgrad!(t+dt, splinepar, qs, gr)
        controlfuncgrad!(t+dt, splinepar, qa, gi)


        # fr1 .= fr1 .+ gi[kpar].*Hanti_ops[q]*vr1  .- gr[kpar].*Hsym_ops[q]*vi05
        mul!(fr1, Hsym_ops[q], vi05, -gr[kpar], 1.0)
        mul!(fr1, Hanti_ops[q], vr1, gi[kpar], 1.0)
    end

    offset = 2*Ncoupled-1
    for q=1:Nunc
        qu = offset+q

        if(isSymm[q])
            controlfuncgrad!(t, splinepar, qu, gr)
            # fr0 .= fr0 .-  gr[kpar].*Hunc_ops[q]*vi05
            mul!(fr0, Hunc_ops[q], vi05, -gr[kpar], 1.0)

            controlfuncgrad!(t+0.5*dt, splinepar, qu, gr)
            # fi0 .= fi0 .+  gr[kpar].*Hunc_ops[q]*vr0
            mul!(fi0, Hunc_ops[q], vr0, gr[kpar], 1.0)

            # fi1 .= fi1 .+  gr[kpar].*Hunc_ops[q]*vr1
            mul!(fi1, Hunc_ops[q], vr1, gr[kpar], 1.0)

            controlfuncgrad!(t+dt, splinepar, qu, gr)
            # fr1 .= fr1 .-  gr[kpar].*Hunc_ops[q]*vi05
            mul!(fr1, Hunc_ops[q], vi05, -gr[kpar], 1.0)
        else 
            controlfuncgrad!(t, splinepar, qu, gi)

            # fr0 .= fr0 .+ gi[kpar].*Hunc_ops[q]*vr0
            mul!(fr0, Hunc_ops[q], vr0, gi[kpar], 1.0)

            controlfuncgrad!(t+0.5*dt, splinepar, qu, gi)

            # fi0 .= fi0 .+ gi[kpar].*Hunc_ops[q]*vi05
            mul!(fi0, Hunc_ops[q], vi05, gi[kpar], 1.0)

            # fi1 .= fi1 .+ gi[kpar].*Hunc_ops[q]*vi05
            mul!(fi1, Hunc_ops[q], vi05, gi[kpar], 1.0)

            controlfuncgrad!(t+dt, splinepar, qu, gi)

            # fr1 .= fr1 .+ gi[kpar].*Hunc_ops[q]*vr1 
            mul!(fr1, Hunc_ops[q], vr1, gi[kpar], 1.0)
        end
    end

end

# FG: It's possible to fuse some of these calls together to save some redundant matrix multiplations
# This routine calculates the contribution to the discrete gradient via the adjoint method for one time step (note that dt is negative)
@inline function adjoint_grad_calc!(Hsym_ops::Array{MyRealMatrix,1}, Hanti_ops::Array{MyRealMatrix, 1}, 
                                   Hunc_ops::Array{MyRealMatrix, 1}, Nunc::Int64, isSymm::BitArray{1}, vr0::Array{Float64,N},
                                   vi05::Array{Float64,N}, vr::Array{Float64,N}, lambdar0::Array{Float64,N}, lambdar05::Array{Float64,N},
                                   lambdai::Array{Float64,N}, lambdai0::Array{Float64,N}, t0::Float64, dt::Float64, 
                                   splinepar::BsplineParams,gr::Array{Float64,1}, gi::Array{Float64,1}, grad_step::Array{Float64,1}) where N

    # # NEW ordering: p-func are even, q-func are odd. p1(t) goes with Hsym_ops[1], q1(t) with Hanti_ops[1], etc
    Ncoupled = length(Hsym_ops)
    Npar = splinepar.Ncoeff # get the number of parameters = size of the gradient, from the spline object
    grad_step .= 0.0

    for q=1:Ncoupled
        qs = (q-1)*2
        qa = qs+1
        controlfuncgrad!(t0, splinepar, qs, gr)
        controlfuncgrad!(t0, splinepar, qa, gi)

        # grad_step .= grad_step .- adjoint_trace_operator!(vr0,Hanti_ops[q],lambdar05)*gi
        trace_tmp = adjoint_trace_operator!(vr0,Hanti_ops[q],lambdar05)
        axpy!(-trace_tmp, gi, grad_step)
        trace_tmp = adjoint_trace_operator!(vi05,Hsym_ops[q],lambdar05)
        # grad_step .= grad_step .- trace_tmp*gr
        axpy!(-trace_tmp, gr, grad_step)


        controlfuncgrad!(t0+dt, splinepar, qs, gr)
        controlfuncgrad!(t0+dt, splinepar, qa, gi)
        # grad_step .= grad_step .- adjoint_trace_operator!(vr,Hanti_ops[q],lambdar05)*gi # fuse with first call to adadjoint_trace_operator?
        # grad_step .= grad_step .- trace_tmp*gr
        axpy!(-trace_tmp, gr, grad_step)
        trace_tmp = adjoint_trace_operator!(vr,Hanti_ops[q],lambdar05)
        # grad_step .= grad_step .- adjoint_trace_operator!(vr,Hanti_ops[q],lambdar05)*gi # fuse with first call to adadjoint_trace_operator?
        axpy!(-trace_tmp, gi, grad_step)


        controlfuncgrad!(t0+0.5*dt, splinepar, qs, gr)
        controlfuncgrad!(t0+0.5*dt, splinepar, qa, gi)
        # grad_step .= grad_step .+ adjoint_trace_operator!(vr,Hsym_ops[q],lambdai)*gr
        trace_tmp = adjoint_trace_operator!(vr,Hsym_ops[q],lambdai)
        axpy!(trace_tmp, gr, grad_step)

        # grad_step .= grad_step .+ adjoint_trace_operator!(vr0,Hsym_ops[q],lambdai0)*gr
        trace_tmp = adjoint_trace_operator!(vr0,Hsym_ops[q],lambdai0)
        axpy!(trace_tmp, gr, grad_step)

        # grad_step .= grad_step .- adjoint_trace_operator!(vi05,Hanti_ops[q],lambdai)*gi
        trace_tmp = adjoint_trace_operator!(vi05,Hanti_ops[q],lambdai)
        axpy!(-trace_tmp, gi, grad_step)

        # grad_step .= grad_step .- adjoint_trace_operator!(vi05,Hanti_ops[q],lambdai0)*gi
        trace_tmp = adjoint_trace_operator!(vi05,Hanti_ops[q],lambdai0)
        axpy!(-trace_tmp, gi, grad_step)
    end

    # Collect contribution from uncoupled control functions
    offset = 2*Ncoupled-1
    for q=1:Nunc
        qu = offset+q
        if(isSymm[q])
            controlfuncgrad!(t0, splinepar, qu, gr)
            tmp = adjoint_trace_operator!(vi05,Hunc_ops[q],lambdar05)
            axpy!(-tmp, gr, grad_step)
            controlfuncgrad!(t0+dt, splinepar, qu, gr)
            axpy!(-tmp, gr, grad_step)
            controlfuncgrad!(t0+0.5*dt, splinepar, qu, gr)
            tmp = adjoint_trace_operator!(vr,Hunc_ops[q],lambdai)
            tmp += adjoint_trace_operator!(vr0,Hunc_ops[q],lambdai0)
            axpy!(tmp, gr, grad_step)
        else
            controlfuncgrad!(t0, splinepar, qu, gi)
            tmp = adjoint_trace_operator!(vr0,Hunc_ops[q],lambdar05)
            axpy!(-tmp, gi, grad_step)

            controlfuncgrad!(t0+dt, splinepar, qu, gi)

            tmp = adjoint_trace_operator!(vr,Hunc_ops[q],lambdar05)
            axpy!(-tmp, gi, grad_step)

            controlfuncgrad!(t0+0.5*dt, splinepar, qu, gi)

            tmp = adjoint_trace_operator!(vi05,Hunc_ops[q],lambdai)
            axpy!(-tmp, gi, grad_step)

            tmp = adjoint_trace_operator!(vi05,Hunc_ops[q],lambdai0)
            axpy!(-tmp, gi, grad_step)

        end
    end

end

# Calls to KS! need to be updated
function eval_forward(U0::Array{Float64,2}, pcof0::Array{Float64,1}, params::objparams; nsteps::Int64=0, 
                        saveEndOnly::Bool=true, saveEvery::Int64=1, verbose::Bool = false, order::Int64=2, stages::Int64=0)  
    N = params.N  

    Nguard = params.Nguard  
    T = params.T
    # Use params' nsteps by default, but user can provide a specific value instead
    if nsteps == 0
        nsteps = params.nsteps
    end

    H0 = params.Hconst

    Ntot = N + Nguard
    pcof = pcof0

    # We have 2*Ncoupled ctrl functions
    Ncoupled = params.Ncoupled
    Nunc = params.Nunc
    Nfreq = params.Nfreq
    NfreqTot = params.NfreqTot
    Nsig = 2*(Ncoupled + Nunc)

    @assert(Nunc==0)
    
    linear_solver = params.linear_solver    

    Psize = size(pcof,1) #must provide separate coefficients for the real and imaginary parts of the control fcn
    if Psize%2 != 0 || Psize < 6
        error("pcof must have an even number of elements >= 6, not ", Psize)
    end
    if params.use_bcarrier
        # D1 = div(Psize, Nsig*Nfreq)  # 
        # Psize = D1*Nsig*Nfreq # active part of the parameter array
        D1 = div(Psize, 2*NfreqTot)  # 
        Psize = 2*D1*NfreqTot # active part of the parameter array
    else
        D1 = div(Psize, Nsig)
        Psize = D1*Nsig # active part of the parameter array
    end
    
    tinv ::Float64 = 1.0/T
    
    if verbose
        println("Vector dim Ntot =", Ntot , ", Guard levels Nguard = ", Nguard , ", Param dim, Psize = ", Psize, ", Spline coeffs per func, D1= ", D1, ", Nsteps = ", nsteps)
    end
    
    zeromat = zeros(Float64,Ntot,N) 

    # Here we can choose what kind of control function expansion we want to use
    if (params.use_bcarrier)
        splinepar = bcparams(T, D1, params.Cfreq, pcof)
    else
        splinepar = splinepar(T, D1, Nsig, pcof)   # parameters for B-splines
    end

    # it is up to the user to estimate the number of time steps
    dt ::Float64 = T/nsteps

    gamma, used_stages = getgamma(order, stages)

    if verbose
        println("Final time: ", T, ", number of time steps: " , nsteps , ", time step: " , dt )
    end

    # the basis for the initial data as a matrix
    Ident = params.Ident

    # Note: Initial condition is supplied as an argument

    #real and imaginary part of initial condition
    vr   = U0[:,:]
    vi   = zeros(Float64,Ntot,N)
    vi05 = zeros(Float64,Ntot,N)

    if nsteps%saveEvery != 0
        error("nsteps must be divisible by saveEvery. nsteps=$nsteps, saveEvery=$saveEvery")
    end

    if !saveEndOnly # Only allocate solution memory for entire timespan if necessary
        usaver = zeros(Float64, Ntot, N, nsteps÷saveEvery +1)
        usavei = zeros(Float64, Ntot, N, nsteps÷saveEvery +1)
        usaver[:,:,1] = vr # the rotation to the lab frame is the identity at t=0
        usavei[:,:,1] = -vi
    end

    # Preallocate WHAT ABOUT SPARSE FORMAT!
    K0   = zeros(Float64,Ntot,Ntot)
    S0   = zeros(Float64,Ntot,Ntot)
    K05  = zeros(Float64,Ntot,Ntot)
    S05  = zeros(Float64,Ntot,Ntot)
    K1   = zeros(Float64,Ntot,Ntot)
    S1   = zeros(Float64,Ntot,Ntot)
    κ₁   = zeros(Float64,Ntot,N)
    κ₂   = zeros(Float64,Ntot,N)
    ℓ₁   = zeros(Float64,Ntot,N)
    ℓ₂   = zeros(Float64,Ntot,N)
    rhs  = zeros(Float64,Ntot,N)

    #initialize variables for time stepping
    t       ::Float64 = 0.0
    step    :: Int64 = 0

    KS!(K0, S0, t, params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq) 
    # Forward time stepping loop
    for step in 1:nsteps

        # Störmer-Verlet
        for q in 1:used_stages
            
            # Update K and S matrices
            KS!(K0, S0, t, params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq)
            KS!(K05, S05, t + 0.5*dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq)
            KS!(K1, S1, t + dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq)

            # Take a step forward and accumulate weight matrix integral. Note the √2 multiplier is to account
            # for the midpoint rule in the numerical integration of the imaginary part of the signal.
            # @inbounds t, vr, vi, vi05 = step(t, vr, vi, dt*gamma[q], K0, S0, K05, S05, K1, S1, Ident)
            @inbounds t = step!(t, vr, vi, vi05, dt*gamma[q], K0, S0, K05, S05, K1, S1, Ident, κ₁, κ₂, ℓ₁, ℓ₂, rhs,linear_solver)

            # Keep prior value for next step (FG: will this work for multiple stages?)

        end # Stromer-Verlet
        
        # rotated frame
        if (!saveEndOnly) && (step % saveEvery) == 0 
            usaver[:,:, step÷saveEvery + 1] = vr
            usavei[:,:, step÷saveEvery + 1] = -vi
        end

    end #forward time stepping loop

    if verbose
        println("Unitary test:")
        println(" Column   1 - Vnrm")
        Vnrm ::Float64 = 0.0
        for q in 1:N
            Vnrm = vr[:,q]' * vr[:,q] + vi[:,q]' * vi[:,q]
            Vnrm = sqrt(Vnrm)
            println(q, " | ", 1.0 - Vnrm)
        end
    end #if verbose

    # return to calling routine

    if saveEndOnly
        return vr - im*vi
    else
        return usaver + im*usavei
    end

end

# Estimate the number of terms used in the Neumann series linear solve during timestepping. 
# Both coupled and uncoupled terms present.
# FMG: This will work but appears to be pessimistic. One can use fewer terms, perhaps a better estimate can be found.
# TODO: Make maxpar and maxunc keyword arguments to simplify calling when maxpar=Float64[]?
"""
    estimate_Neumann!(tol, params, maxpar[, maxunc])

Estimate the number of terms needed by the Neumann series approach for solving the linear system
during the implicit steps of the Störmer-Verlet scheme. See also neumann!
 
# Arguments
- `tol:: Float64`: Error tolerance in inverting implicit SV term
- `params:: objparams`: Struct containing problem definition
- `maxpar:: Array{Float64,1}`: Maximum parameter value for each coupled control
- `maxunc:: Array{Float64,1}`: (optional) Maximum parameter value for each uncoupled controls
"""
function estimate_Neumann!(tol::Float64, params::objparams, maxpar::Array{Float64,1}, maxunc::Array{Float64,1}=Float64[])
    nsteps = params.nsteps
    k = Float64(params.T/nsteps)
    if(params.Ncoupled > 0)
        @assert(length(maxpar) >= params.Ncoupled)

        S = 0.5*k*maxpar[1]*params.Hanti_ops[1]
        for j = 2:length(params.Hanti_ops)
            axpy!(0.5*k*maxpar[j],params.Hanti_ops[j],S)
        end
    end

    if(params.Nunc > 0)
        @assert(length(maxunc) >= params.Nunc)

        if(!@isdefined S)
            S = zeros(size(params.Hunc_ops[1]))
        end
        for j = 1:params.Nunc
            if(!params.isSymm[j])
                axpy!(0.5*k*maxunc[j],params.Hunc_ops[j],S)     
            end
        end
    end

    # If in sparse mode, cast to full matrix for norm estimation
    if(typeof(S) ==  SparseMatrixCSC{Float64, Int64})
         S = Array(S)
    end

    normS = opnorm(S)
    nterms = ceil(Int64,log(tol)/log(normS))-1
    if(nterms > 0)
        params.linear_solver.max_iter = nterms
        recreate_linear_solver_closure!(params.linear_solver)
    end
    # return nterms
end

# unified calculation of the time step, merging previous cases into one routine
"""
    nsteps = calculate_timestep(T, H0; Hsym_ops=[], Hanti_ops=[], Hunc_ops=[], maxCoupled=[], maxUnc=[], Pmin=40)

Estimate the number of time steps needed for the simulation, when there are uncoupled controls.
 
# Arguments
- `T:: Float64`: Gate duration = Final simulation time
- `H0::Matrix{Float64}`: Time-independent part of the Hamiltonian matrix
- `Hsym_ops:: Vector{Matrix{Float64}}`: (Optional kw-arg) Array of symmetric control Hamiltonians
- `Hanti_ops:: Vector{Matrix{Float64}}`: (Optional kw-arg) Array of anti-symmetric control Hamiltonians
- `Hunc_ops:: Vector{Matrix{Float64}}`: (Optional kw-arg) Array of uncoupled control Hamiltonians
- `maxCoupled:: Vector{Float64}`: (Optional kw-arg) Maximum amplitude for each control function
- `maxUnc:: Vector{Float64}`: (Optional kw-arg) Maximum control amplitude for each uncoupled control Hamiltonian
- `Pmin:: Int64`: (Optional kw-arg) Number of time steps per shortest period (assuming a slowly varying Hamiltonian).
"""
function calculate_timestep(T::Float64, H0::Matrix{Float64}; Hsym_ops::Vector{Matrix{Float64}}=Matrix{Float64}[], Hanti_ops::Vector{Matrix{Float64}}=Matrix{Float64}[], Hunc_ops::Vector{Matrix{Float64}}=Matrix{Float64}[], maxCoupled::Vector{Float64}=Float64[], maxUnc::Vector{Float64}=Float64[], Pmin::Int64=40)

    Ncoupled = length(Hsym_ops)
    Nunc = length(Hunc_ops)
    @assert(length(maxCoupled)>= Ncoupled)
    @assert(length(maxUnc)>= Nunc)

    K1 = copy(H0) # system Hamiltonian

    # Coupled control Hamiltonians
    for i = 1:Ncoupled
        K1 += maxCoupled[i].*Hsym_ops[i] + 1im*maxCoupled[i].*Hanti_ops[i]
    end

    # Uncoupled control Hamiltonians
    for i = 1:Nunc
        if(issymmetric(Hunc_ops[i]))
            K1 += maxUnc[i]*Hunc_ops[i]
        elseif(norm(Hunc_ops[i]+Hunc_ops[i]') < 1e-14)
            K1 += 1im*maxUnc[i].*Hunc_ops[i]
        else 
            throw(ArgumentError("Uncoupled Hamiltonians must currently be either symmetric or anti-symmetric.\n"))
        end
    end

    # Estimate time step
    lamb = eigvals(Array(K1))
    maxeig = maximum(abs.(lamb)) 
    mineig = minimum(abs.(lamb)) 

    samplerate1 = maxeig*Pmin/(2*pi)
    nsteps = ceil(Int64, T*samplerate1)

    # NOTE: The above estimate does not account for quickly varying signals or a large number of splines. 
    # Double check at least 2-3 points per spline to resolve control function.

    return nsteps
end

# sparse array case:
"""
    nsteps = calculate_timestep(T, H0; Hsym_ops=[], Hanti_ops=[], Hunc_ops=[], maxCoupled=[], maxUnc=[], Pmin=40)

Estimate the number of time steps needed for the simulation, when there are uncoupled controls.
 
# Arguments
- `T:: Float64`: Gate duration = Final simulation time
- `H0::SparseMatrixCSC{Float64,Int64}`: Time-independent part of the Hamiltonian matrix
- `Hsym_ops:: Vector{SparseMatrixCSC{Float64,Int64}}`: (Optional kw-arg) Array of symmetric control Hamiltonians
- `Hanti_ops:: Vector{SparseMatrixCSC{Float64,Int64}}`: (Optional kw-arg) Array of anti-symmetric control Hamiltonians
- `Hunc_ops:: Vector{SparseMatrixCSC{Float64,Int64}}`: (Optional kw-arg) Array of uncoupled control Hamiltonians
- `maxCoupled:: Vector{Float64}`: (Optional kw-arg) Maximum control amplitude for each sym/anti-sym control Hamiltonian
- `maxUnc:: Vector{Float64}`: (Optional kw-arg) Maximum control amplitude for each uncoupled control Hamiltonian
- `Pmin:: Int64`: (Optional kw-arg) Number of time steps per shortest period (assuming a slowly varying Hamiltonian).
"""
function calculate_timestep(T::Float64, H0::SparseMatrixCSC{Float64,Int64}; Hsym_ops::Vector{SparseMatrixCSC{Float64,Int64}}=SparseMatrixCSC{Float64,Int64}[], Hanti_ops::Vector{SparseMatrixCSC{Float64,Int64}}=SparseMatrixCSC{Float64,Int64}[], Hunc_ops::Vector{SparseMatrixCSC{Float64,Int64}}=SparseMatrixCSC{Float64,Int64}[], maxCoupled::Vector{Float64}=Float64[], maxUnc::Vector{Float64}=Float64[], Pmin::Int64=40)

    Ncoupled = length(Hsym_ops)
    Nunc = length(Hunc_ops)
    @assert(length(maxCoupled)>= Ncoupled)
    @assert(length(maxUnc)>= Nunc)

    K1 = copy(H0) # system Hamiltonian
    # Typecasting issue for sparse arrays
    if(typeof(H0) == SparseMatrixCSC{Float64,Int64})
        K1 = SparseMatrixCSC{ComplexF64,Int64}(K1)
    end

    # Coupled control Hamiltonians
    for i = 1:Ncoupled
        K1 += maxCoupled[i].*Hsym_ops[i] + 1im*maxCoupled[i].*Hanti_ops[i]
    end

    # Uncoupled control Hamiltonians
    for i = 1:Nunc
        if(issymmetric(Hunc_ops[i]))
            K1 += maxUnc[i]*Hunc_ops[i]
        elseif(norm(Hunc_ops[i]+Hunc_ops[i]') < 1e-14)
            K1 += 1im*maxUnc[i].*Hunc_ops[i]
        else 
            throw(ArgumentError("Uncoupled Hamiltonians must currently be either symmetric or anti-symmetric.\n"))
        end
    end

    # Estimate time step
    lamb = eigvals(Array(K1))
    maxeig = maximum(abs.(lamb)) 
    mineig = minimum(abs.(lamb)) 

    samplerate1 = maxeig*Pmin/(2*pi)
    nsteps = ceil(Int64, T*samplerate1)

    # NOTE: The above estimate does not account for quickly varying signals or a large number of splines. 
    # Double check at least 2-3 points per spline to resolve control function.

    return nsteps
end


# Preallocate K and S matrices, not relying on params
function ks_alloc(Ntot:: Int64, Hconst::MyRealMatrix, Hsym_ops::Vector{MyRealMatrix}, Hanti_ops::Vector{MyRealMatrix}, Hunc_ops::Vector{MyRealMatrix}, isSymm::BitArray{1})
    # Ntot = prod(params.Nt)
    # establish the non-zero pattern for sparse storage
    if typeof(Hconst) == SparseMatrixCSC{Float64, Int64}
        K0 = copy(Hconst)
        S0 = spzeros(size(Hconst,1),size(Hconst,2))
        Ncoupled = length(Hsym_ops)
        for q=1:Ncoupled
            K0 += Hsym_ops[q]
            S0 += Hanti_ops[q]
        end
        Nunc = length(Hunc_ops)
        for q=1:Nunc
            if(isSymm[q])
                K0 = K0 + Hunc_ops[q]
            else
                S0 = S0 + Hunc_ops[q]
            end
        end
        K05 = copy(K0)
        K1  = copy(K0)
        S05 = copy(S0)
        S1  = copy(S0)
    else
        K0   = zeros(Float64,Ntot,Ntot)
        S0   = zeros(Float64,Ntot,Ntot)
        K05  = zeros(Float64,Ntot,Ntot)
        S05  = zeros(Float64,Ntot,Ntot)
        K1   = zeros(Float64,Ntot,Ntot)
        S1   = zeros(Float64,Ntot,Ntot)
    end
    # vtargetr = real(params.Utarget)
    # vtargeti = imag(params.Utarget)
    #return K0,S0,K05,S05,K1,S1,vtargetr,vtargeti
    return K0,S0,K05,S05,K1,S1
end

# Working arrays for timestepping
function time_step_alloc(Ntot::Int64, N::Int64)
    lambdar   = zeros(Float64,Ntot,N) 
    lambdar0  = zeros(Float64,Ntot,N) 
    lambdai   = zeros(Float64,Ntot,N) 
    lambdai0  = zeros(Float64,Ntot,N) 
    lambdar05 = zeros(Float64,Ntot,N) 
    κ₁        = zeros(Float64,Ntot,N)
    κ₂        = zeros(Float64,Ntot,N)
    ℓ₁        = zeros(Float64,Ntot,N)
    ℓ₂        = zeros(Float64,Ntot,N)
    rhs       = zeros(Float64,Ntot,N)
    gr0       = zeros(Float64,Ntot,N)
    gi0       = zeros(Float64,Ntot,N)
    gr1       = zeros(Float64,Ntot,N)
    gi1       = zeros(Float64,Ntot,N)
    hr0       = zeros(Float64,Ntot,N)
    hi0       = zeros(Float64,Ntot,N)
    hi1       = zeros(Float64,Ntot,N)
    hr1       = zeros(Float64,Ntot,N)
    vr        = zeros(Float64,Ntot,N)
    vi        = zeros(Float64,Ntot,N)
    vi05      = zeros(Float64,Ntot,N)
    vr0       = zeros(Float64,Ntot,N)
    vfinalr   = zeros(Float64,Ntot,N) 
    vfinali   = zeros(Float64,Ntot,N)
    return lambdar,lambdar0,lambdai,lambdai0,lambdar05,κ₁,κ₂,ℓ₁,
           ℓ₂,rhs,gr0,gi0,gr1,gi1,hr0,hi0,hi1,hr1,vr,vi,vi05,vr0,vfinalr,vfinali
end

function grad_alloc(Nparams::Int64)
    gr = zeros(Nparams)
    gi = zeros(Nparams)
    gradobjfadj = zeros(Nparams)
    tr_adj = zeros(Nparams)
    return gr, gi, gradobjfadj, tr_adj
end

# setup the initial conditions
"""
    u_init = initial_cond(Ne, Ng, msb_order=true)

Setup a basis of canonical unit vectors that span the essential Hilbert space, setting all guard levels to zero
 
# Arguments
- `Ne:: Array{Int64}`: Array holding the number of essential levels in each system
- `Ng:: Array{Int64}`: Array holding the number of guard levels in each system
- `msb_order:: Bool`: Most Significant Bit (MSB) ordering: true
"""
function initial_cond(Ne::Vector{Int64}, Ng::Vector{Int64}, msb_order::Bool = true)
    Nt = Ne + Ng
    Ntot = prod(Nt)
    @assert length(Nt) <= 3 "ERROR: initial_cond(): only length(Nt) <= 3 is implemented"
    NgTot = sum(Ng)
    N = prod(Ne)
    Ident = Matrix{Float64}(I, Ntot, Ntot)
    U0 = Ident[1:Ntot,1:N] # initial guess

    #adjust initial guess if there are ghost points
    if msb_order
        if length(Nt) == 3
            if NgTot > 0
                col = 0
                m = 0
                for k3 in 1:Nt[3]
                    for k2 in 1:Nt[2]
                        for k1 in 1:Nt[1]
                            m += 1
                            # is this a guard level?
                            guard = (k1 > Ne[1]) || (k2 > Ne[2]) || (k3 > Ne[3])
                            if !guard
                                col = col+1
                                U0[:,col] = Ident[:,m]
                            end # if ! guard
                        end #for
                    end # for
                end # for            
            end # if  
        elseif length(Nt) == 2
            if NgTot > 0
                # build up a basis for the essential states
                col = 0
                m = 0
                for k2 in 1:Nt[2]
                    for k1 in 1:Nt[1]
                        m += 1
                        # is this a guard level?
                        guard = (k1 > Ne[1]) || (k2 > Ne[2])
                        if !guard
                            col += 1
                            U0[:,col] = Ident[:,m]
                        end # if ! guard
                    end # for
                end # for
            end # if
        end
    else
        if length(Nt) == 3
            if NgTot > 0
                col = 0
                m = 0
                for k1 in 1:Nt[1]
                    for k2 in 1:Nt[2]
                        for k3 in 1:Nt[3]    
                            m += 1
                            # is this a guard level?
                            guard = (k1 > Ne[1]) || (k2 > Ne[2]) || (k3 > Ne[3])
                            if !guard
                                col = col+1
                                U0[:,col] = Ident[:,m]
                            end # if ! guard
                        end #for
                    end # for
                end # for            
            end # if  
        elseif length(Nt) == 2
            if NgTot > 0
                # build up a basis for the essential states
                col = 0
                m = 0
                for k1 in 1:Nt[1]
                    for k2 in 1:Nt[2]
                        m += 1
                        # is this a guard level?
                        guard = (k1 > Ne[1]) || (k2 > Ne[2])
                        if !guard
                            col += 1
                            U0[:,col] = Ident[:,m]
                        end # if ! guard
                    end # for
                end # for
            end # if
        end
    end
    return U0
end

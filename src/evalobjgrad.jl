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

    function working_arrays(N:: Int64, Ntot:: Int64, Hconst::MyRealMatrix, Hsym_ops::Vector{MyRealMatrix}, Hanti_ops::Vector{MyRealMatrix}, Hunc_ops::Vector{MyRealMatrix}, isSymm::BitArray{1}, objFuncType::Int64, nCoeff::Int64)
        #N = params.N
        #Ntot = N + params.Nguard

        # K0,S0,K05,S05,K1,S1,vtargetr,vtargeti = KS_alloc(params)
        # ks_alloc(Ntot:: Int64, Hconst::MyRealMatrix, Hsym_ops::Vector{MyRealMatrix}, Hanti_ops::Vector{MyRealMatrix}, Hunc_ops::Vector{MyRealMatrix}, isSymm::BitArray{1})
        K0, S0, K05, S05, K1, S1 = ks_alloc(Ntot, Hconst, Hsym_ops, Hanti_ops, Hunc_ops, isSymm)

        lambdar,lambdar0,lambdai,lambdai0,lambdar05,κ₁,κ₂,ℓ₁,ℓ₂,rhs,gr0,gi0,gr1,gi1,hr0,hi0,hi1,hr1,vr,vi,vi05,vr0,vfinalr,vfinali = time_step_alloc(Ntot,N)

        gr, gi, gradobjfadj, tr_adj = grad_alloc(nCoeff)

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
                        Hconst=Hconst,
                        w_diag_mat=w_diag_mat,
                        nCoeff=nCoeff,
                        Hsym_ops=Hsym_ops,
                        Hanti_ops=Hanti_ops, 
                        Hunc_ops=Hunc_ops,
                        objFuncType=objFuncType,
                        leak_ubound=leak_ubound,
                        linear_solver = lsolver_object(),
                        use_sparse = use_sparse,
                        dVds = dVds,
                        freq01::Vector{Float64} = Vector{Float64}[],
                        self_kerr::Vector{Float64} = Vector{Float64}[],
                        couple_coeff::Vector{Float64} = Vector{Float64}[],
                        couple_type::Int64 = 0,
                        zeroCtrlBC::Bool = true)

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
- `Uinit::Array{ComplexF64,2}`: (keyword) Matrix holding the initial conditions for the solution matrix of size Uinit[Ntot, Ness]
- `Utarget::Array{ComplexF64,2}`: (keyword) Matrix holding the target gate matrix of size Uinit[Ntot, Ness]
- `Cfreq::Vector{Vector{Float64}}`: (keyword) Carrier wave (angular) frequencies of size Cfreq[Nctrl]
- `Rfreq::Array{Float64,1}`: (keyword) Rotational (regular) frequencies for each control Hamiltonian; size Rfreq[Nctrl]
- `Hconst::Array{Float64,2}`: (keyword) Time-independent part of the Hamiltonian matrix of size Ntot × Ntot
- `Hsym_ops:: Array{Array{Float64,2},1}`: (keyword) Array of symmetric control Hamiltonians, each of size Ntot × Ntot
- `Hanti_ops:: Array{Array{Float64,2},1}`: (keyword) Array of anti-symmetric control Hamiltonians, each of size Ntot × Ntot
- `Hunc_ops:: Array{Array{Float64,2},1}`: (keyword) Array of uncoupled control Hamiltonians, each of size Ntot × Ntot
- `w_diag_mat::Diagonal{Float64,Array{Float64,1}}`: (keyword) Diagonal matrix with weights for suppressing guarded energy levels
- `nCoeff::Int`: (keyword) length of the control vector (pcof)
- `objFuncType::Int64 = 1`  # 1 = objective function include infidelity and leakage
                            # 2 = objective function only includes infidelity... no leakage in obj function or constraint
                            # 3 = objective function only includes infidelity; leakage treated as inequality constraint
- `leak_ubound::Float64 = 1.0e-3`  : The upper bound on the leakage inequality constraint (See examples/cnot2-leakieq-setup.jl )
- `linear_solver::lsolver_object = lsolver_object()` : The linear solver object used to solve the implicit & adjoint system
- `use_sparse::Bool = false`: (keyword) Set to true to sparsify all Hamiltonian matrices
- `dVds::Array{Complex{Float64},2}`: (keyword) Matrix holding the complex-valued matrix dV/ds of size Ntot x Ne (for continuation)
- `freq01::Vector{Float64} = Vector{Float64}[]`: Transition frequencies
- `self_kerr::Vector{Float64} = Vector{Float64}[]`: Self-Kerr coefficients
- `couple_coeff::Vector{Float64} = Vector{Float64}[]`: Coupling coefficients
- `couple_type::Int64 = 0`: Coupling type (1 = cross-Kerr, 2 = dispersive)
- `zeroCtrlBC::Bool = true`: Boundary condition for the B-splines
"""
mutable struct objparams
    Nosc   ::Int64          # number of oscillators in the coupled quantum systems
    N      ::Int64          # total number of essential levels
    Nguard ::Int64          # total number of guard levels
    Ntot   ::Int64          # total number of levels (essential + guard)
    Ne     ::Array{Int64,1} # essential levels for each oscillator
    Ng     ::Array{Int64,1} # guard levels for each oscillator
    Nt     ::Array{Int64,1} # total # levels for each oscillator
    T      ::Float64        # final time

    nsteps::Int64    # Number of time steps

    Uinit_r       ::Array{Float64,2} # Real part of initial condition for each essential state
    Uinit_i       ::Array{Float64,2} # Imaginary part of the above

    Utarget_r     ::Array{Float64,2}
    Utarget_i     ::Array{Float64,2}

    use_bcarrier  ::Bool
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
    pFidType    ::Int64 # default is 2 = std. trace infidelity
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

    # temporary storage for time stepping
    wa:: working_arrays

    nCoeff :: Int64 # Total length of the design vector (pcof). nCoeff = nAlpha + (nTimeIntervals - 1) * nWinit
    D1:: Int64 # Number of B-spline coefficients in each segment of the control functions
    nAlpha:: Int64 # length of the parameter vector (alpha) within pcof
    nWinit:: Int64 # length of each block of initial condition matrices within pcof

    # Hamiltonian coefficients
    freq01::   Vector{Float64} 
    self_kerr:: Vector{Float64}
    couple_coeff:: Vector{Float64}
    couple_type:: Int64

    msb_order:: Bool # false: Least significant bit ordering of state vector:| i1 i2 i3 i4 >
    zeroCtrlBC:: Bool # Boundary conditions for the control functions

    # variables for the direct method
    nTimeIntervals:: Int64 # number of time intervals in [0, T]
    T0int:: Vector{Float64} # Vector of size nTimeIntervals holding the starting time for each interval
    Tsteps:: Vector{Int64} # Number of time steps in each interval

    Lmult_r:: Vector{Matrix{Float64}} # Vector of size (nTimeIntervals-1) for the Lagrange multiplier coefficients (real)
    Lmult_i:: Vector{Matrix{Float64}} # Vector of size (nTimeIntervals-1) for the Lagrange multiplier coefficients (imag)

    gammaJump:: Float64 # Coefficient of the quadratic penalty term for jumps across time intervals

# constructor for regular arrays (full matrices)
    function objparams(Ne::Array{Int64,1}, Ng::Array{Int64,1}, T::Float64, nsteps::Int64;
                       Uinit::Array{ComplexF64,2}, Utarget::Array{Complex{Float64},2}, # keyword args w/o default values (must be assigned)
                       Cfreq::Vector{Vector{Float64}}, Rfreq::Array{Float64,1}, Hconst::Array{Float64,2},
                       w_diag_mat::Diagonal{Float64,Array{Float64,1}}, nCoeff::Int64, D1::Int64,
                       # keyword args with default values
                       Hsym_ops:: Array{Array{Float64,2},1} = Array{Float64,2}[], Hanti_ops:: Array{Array{Float64,2},1} = Array{Float64,2}[],
                       Hunc_ops:: Array{Array{Float64,2},1} = Array{Float64,2}[],
                       forb_states:: Array{ComplexF64,2} = Array{ComplexF64}(undef,0,2),
                       forb_weights:: Vector{Float64} = Float64[],
                       objFuncType:: Int64 = 1, leak_ubound:: Float64=1.0e-3,
                       use_sparse::Bool = false, use_custom_forbidden::Bool = false,
                       linear_solver::lsolver_object = lsolver_object(nrhs=prod(Ne)), msb_order::Bool = true,
                       dVds::Array{ComplexF64,2}= Array{ComplexF64}(undef,0,0), freq01::Vector{Float64} = Vector{Float64}[], self_kerr::Vector{Float64} = Vector{Float64}[], couple_coeff::Vector{Float64} = Vector{Float64}[], couple_type::Int64 = 0, zeroCtrlBC::Bool = true, nTimeIntervals::Int64=1, gammaJump::Float64=0.1)
        pFidType = 2 # Std gate infidelity
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
        nAlpha = 2*D1*NfreqTot # length of control vector within pcof
        nWinit = 2*Ntot^2 # length of each initial condition within pcof

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
        wmat = w_diag_mat # wmatScale.*Juqbox.wmatsetup_old(Ne, Ng, msb_order)

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

        wa = working_arrays(N, Ntot, convert(MyRealMatrix, Hconst), convert(Vector{MyRealMatrix}, Hsym_ops1), convert(Vector{MyRealMatrix}, Hanti_ops1), convert(Vector{MyRealMatrix}, Hunc_ops1), isSymm, objFuncType, nCoeff)

        T0int = zeros(Float64, nTimeIntervals)
        Tsteps = zeros(Int64, nTimeIntervals)

        T0int[1] = 0.0 # first interval always starts from t=0.0
    
        # Allocate storage for Lagrange multipliers
        Lmult_r = Vector{Matrix{Float64}}(undef, nTimeIntervals-1)
        Lmult_i = Vector{Matrix{Float64}}(undef, nTimeIntervals-1)

        # Compute intermediate start times by rounding to the nearest time step
        if nTimeIntervals == 1
            Tsteps[1] = nsteps
        else
            dt = T/nsteps
            ds = 1.0/nTimeIntervals
            Tgoal = T*ds
            Tsteps[1] = round(Int64, Tgoal/dt)
            #Hdelta = im*log(Ubasis'*Utarget)
            for q = 2:nTimeIntervals
                ts_start = sum(Tsteps[1:q-1])
                T0int[q] = dt*ts_start
                s1 = q*ds
                ts_end = round(Int64, T*s1/dt)
                # println("q=", q, " ts_start=", ts_start, " ts_end=", ts_end)
                Tsteps[q] = max(1, ts_end - ts_start)
                # Allocate space for the Lagrange multipliers
                Lmult_r[q-1] = zeros(Ntot, Ntot)
                Lmult_i[q-1] = zeros(Ntot, Ntot)
            end
            # make sure the number of time steps sum up to nsteps
            if (sum(Tsteps) != nsteps)
                Tsteps[nTimeIntervals] = nsteps - Tsteps[nTimeIntervals - 1]
            end
        end
    
        # println("\nNorm real(Uinit), image(Uinit): ", norm(real(Uinit)), ", ", norm(imag(Uinit)), "\n")

        # sv_type is used for continuation. Only change this if you know what you are doing
        new(
             Nosc, N, Nguard, N+Nguard, Ne, Ng, Ne+Ng, T, nsteps, real(Uinit), imag(Uinit), real(Utarget), imag(Utarget), 
             use_bcarrier, NfreqTot, Nfreq, Cfreq, kpar, tik0, Hconst, Hsym_ops1, 
             Hanti_ops1, Hunc_ops1, Ncoupled, Nunc, isSymm, Ident, wmat, 
             forb_states, forb_weights, wmat_real, wmat_imag, pFidType, 0.0,
             objFuncType, leak_ubound,
             0.0,0.0,zeros(0),zeros(0),zeros(0),saveConvHist,
             zeros(0), zeros(0), zeros(0), zeros(0), 
             linear_solver, objThreshold, traceInfidelityThreshold, 0.0, 0.0, 
             usingPriorCoeffs, priorCoeffs, quiet, Rfreq, false, [],
             real(my_dVds), imag(my_dVds), my_sv_type, wa, nCoeff, D1, nAlpha, nWinit,
             freq01, self_kerr, couple_coeff, couple_type, # Add some checks for these ones!
             msb_order, zeroCtrlBC, nTimeIntervals, T0int, Tsteps, Lmult_r, Lmult_i, gammaJump
            )

    end

end # mutable struct objparams


"""
    tpl = traceobjgrad(pcof0, params[, verbose = false, evaladjoint = true])

Perform a forward and/or adjoint Schrödinger solve to evaluate the objective
function and/or gradient.
 
# Arguments
- `pcof0::Array{Float64,1}`: Array of parameter values defining the controls
- `param::objparams`: Struct with problem definition
- `verbose::Bool = false`: Run simulation with additional terminal output and store state history.
- `evaladjoint::Bool = true`: Solve the adjoint equation and calculate the gradient of the objective function.

# Return argument
The return argument `tpl` is a tuple with a content that depends on the input arguments `verbose` and `evaladjoint`. 
- `verbose=false`, `evaladjoint=false`: tpl[1] = objective, tpl[2] = infidelity, tpl[3] = leakage. 
- `verbose=false`, `evaladjoint=true`: tpl[1] = objective, tpl[2] = gradient, tpl[3] = infidelity, tpl[4] = leakage, tpl[5] = trace-fidelity, tpl[6] = infidelity-gradient, tpl[7] = leakage-gradient.  
- `verbose=true`, `evaladjoint=false`: tpl[1] = objective, tpl[2] = final-unitary, tpl[3] = trace-fidelity. 
- `verbose=true`, `evaladjoint=true`: tpl[1] = objective, tpl[2] = gradient, tpl[3] = final-unitary, tpl[4] = trace-fidelity, tpl[5] = dfdp, tpl[6] = wr1 - im*wi. 

"""
function traceobjgrad(pcof0::Array{Float64,1},  p::objparams, verbose::Bool = false, evaladjoint::Bool = true)
    order  = 2
    gamma, stages = getgamma(order)

    # shortcut to working_arrays object in p::objparams
    w = p.wa

    if verbose
        println("traceobjgrad: Vector dim Ntot =", p.Ntot , ", Guard levels Nguard = ", p.Nguard , ", Param dim, Psize = ", p.nAlpha, ", Spline coeffs per func, D1= ", p.D1, ", Nsteps = ", p.nsteps, " Tikhonov coeff: ", p.tik0)
    end

    # initializations start here
    alpha = pcof0[1:p.nAlpha] # extract the B-spline-coefficients

    # setup splinepar
    if p.use_bcarrier
        splinepar = bcparams(p.T, p.D1, p.Cfreq, alpha) # Assumes Nunc = 0
    else
        Nsig  = 2*(p.Ncoupled + p.Nunc) # Only uses for regular B-splines
        splinepar = splineparams(p.T, p.D1, Nsig, alpha)
    end
    
    tinv ::Float64 = 1.0/p.T

    # it is up to the user to estimate the number of time steps
    dt ::Float64 = p.T/p.nsteps

    if verbose
        println("Final time: ", p.T, ", number of time steps: " , p.nsteps , ", time step: " , dt )
    end
    
    #real and imaginary part of initial condition
    copy!(w.vr, p.Uinit_r)
    copy!(w.vi, -p.Uinit_i) # note the sign

    # Zero out working arrays
    initialize_working_arrays(w)
    
    if verbose
        usaver = zeros(Float64,p.Ntot,p.N,p.nsteps+1)
        usavei = zeros(Float64,p.Ntot,p.N,p.nsteps+1)
        usaver[:,:,1] = w.vr # the rotation to the lab frame is the identity at t=0
        usavei[:,:,1] = -w.vi

        #to compute gradient with forward method
        if evaladjoint
            wr   = zeros(Float64,p.Ntot,p.N) 
            wi   = zeros(Float64,p.Ntot,p.N) 
            wr1  = zeros(Float64,p.Ntot,p.N) 
            wi05 = zeros(Float64,p.Ntot,p.N) 
            objf_alpha1 = 0.0
        end
    end

    #initialize variables for time stepping
    t     ::Float64 = 0.0
    step  :: Int64 = 0
    objfv ::Float64 = 0.0


    # Forward time stepping loop
    for step in 1:p.nsteps

        forbidden0 = tinv*penalf2aTrap(w.vr, p.wmat_real)
        # Störmer-Verlet
        for q in 1:stages
            copy!(w.vr0,w.vr)
            t0  = t
            
            # Update K and S matrices
            # general case
            KS!(w.K0, w.S0, t, p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, splinepar, p.Hconst, p.Rfreq) 
            KS!(w.K05, w.S05, t + 0.5*dt*gamma[q], p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, splinepar, p.Hconst, p.Rfreq) 
            KS!(w.K1, w.S1, t + dt*gamma[q], p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, splinepar, p.Hconst, p.Rfreq) 
            
            # Take a step forward and accumulate weight matrix integral. Note the √2 multiplier is to account
            # for the midpoint rule in the numerical integration of the imaginary part of the signal.
            @inbounds t = step!(t, w.vr, w.vi, w.vi05, dt*gamma[q], w.K0, w.S0, w.K05, w.S05, w.K1, w.S1, p.Ident, w.κ₁, w.κ₂, w.ℓ₁, w.ℓ₂, w.rhs, p.linear_solver)

            forbidden = tinv*penalf2a(w.vr, w.vi05, p.wmat_real)  
            forbidden_imag1 = tinv*penalf2imag(w.vr0, w.vi05, p.wmat_imag)
            objfv = objfv + gamma[q]*dt*0.5*(forbidden0 + forbidden - 2.0*forbidden_imag1)

            # Keep prior value for next step (FG: will this work for multiple stages?)
            forbidden0 = forbidden

            # compute component of the gradient for verification of adjoint method
            if evaladjoint && verbose
                # compute the forcing for (wr, wi)
                fgradforce!(p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc,
                            p.isSymm, w.vr0, w.vi05, w.vr, t-dt, dt, splinepar, p.kpar, w.gr0, w.gr1, w.gi0, w.gi1, w.gr, w.gi)

                copy!(wr1,wr)

                @inbounds step_fwdGrad!(t0, wr1, wi, wi05, dt*gamma[q],
                                        w.gr0, w.gi0, w.gr1, w.gi1, w.K0, w.S0, w.K05, w.S05, w.K1, w.S1, p.Ident, w.κ₁, w.κ₂, w.ℓ₁, w.ℓ₂, w.rhs, p.linear_solver) 
                
                # Real part of forbidden state weighting
                forbalpha0 = tinv*penalf2grad(w.vr0, w.vi05, wr, wi05, p.wmat_real)
                forbalpha1 = tinv*penalf2grad(w.vr, w.vi05, wr1, wi05, p.wmat_real)

                # Imaginary part of forbidden state weighting
                forbalpha2 = tinv*penalf2grad(wi05, w.vi05, w.vr0, wr, p.wmat_imag)                

                copy!(wr,wr1)
                # accumulate contribution from the leak term
                objf_alpha1 = objf_alpha1 + gamma[q]*dt*0.5*2.0*(forbalpha0 + forbalpha1 + forbalpha2) 

            end  # evaladjoint && verbose
        end # Stromer-Verlet
        
        if verbose
            # rotated frame
            usaver[:,:, step + 1] = w.vr
            usavei[:,:, step + 1] = -w.vi
        end
    end # end forward time stepping loop

    if p.pFidType == 1
        scomplex1 = tracefidcomplex(w.vr, -w.vi, p.Utarget_r, p.Utarget_i)
        primaryobjf = 1+tracefidabs2(w.vr, -w.vi, p.Utarget_r, p.Utarget_i) - 2*real(scomplex1*exp(-1im*p.globalPhase)) # global phase angle 
    elseif p.pFidType == 2
        primaryobjf = (1.0-tracefidabs2(w.vr, -w.vi, p.Utarget_r, p.Utarget_i)) # insensitive to global phase angle
    elseif p.pFidType == 4
        rotTarg = exp(1im*p.globalPhase)*(p.Utarget_r + im*p.Utarget_i)
        primaryobjf = (1.0 - tracefidreal(w.vr, -w.vi, real(rotTarg), imag(rotTarg)) )
    end

    secondaryobjf = objfv
    objfv = primaryobjf + secondaryobjf

    if p.objFuncType == 1
        objfv += tikhonov_pen(alpha, p)
    end

    if evaladjoint && verbose
        salpha1 = tracefidcomplex(wr, -wi, p.dVds_r, p.dVds_i)
        scomplex1 = tracefidcomplex(w.vr, -w.vi, p.Utarget_r, p.Utarget_i)
        if p.pFidType==1
            primaryobjgrad = 2*real(conj( scomplex1 - exp(1im*p.globalPhase) )*salpha1)
        elseif p.pFidType == 2
            primaryobjgrad = - 2*real(conj(scomplex1)*salpha1)
        elseif p.pFidType == 4
            rotTarg = exp(1im*p.globalPhase)*(p.Utarget_r + im*p.Utarget_i)
            # grad wrt the control function 
            primaryobjgrad = - tracefidreal(wr, -wi, real(rotTarg), imag(rotTarg))
            # grad wrt the global phase
            primObjGradPhase = - tracefidreal(w.vr, -w.vi, real(im.*rotTarg), imag(im.*rotTarg))
        end
        objf_alpha1 = objf_alpha1 + primaryobjgrad
    end  

    w.vfinalr = copy(w.vr)
    w.vfinali = copy(-w.vi)

    traceFidelity = tracefidabs2(w.vfinalr, w.vfinali, p.Utarget_r, p.Utarget_i)

    if evaladjoint

        if verbose
            dfdp = objf_alpha1
        end  

        if (p.use_bcarrier)
            # gradSize = (2*p.Ncoupled+p.Nunc)*Nfreq*D1
            gradSize = p.NfreqTot*2*p.D1
        else
            gradSize = Nsig*p.D1
        end


        # initialize array for storing the adjoint gradient so it can be returned to the calling function/program
        leakgrad = zeros(0);
        infidelgrad = zeros(0);
        w.gradobjfadj[:] .= 0.0    
        t = p.T
        dt = -dt

        
        # println("traceobjgrad(): eval_adjoint: sv_type = ", p.sv_type) # tmp
        if p.sv_type == 1 || p.sv_type == 2 # regular case
            # println("scomplex #1 (vtarget)")
            scomplex0 = tracefidcomplex(w.vr, -w.vi, p.Utarget_r, p.Utarget_i)
        elseif p.sv_type == 3 # term2 for d/ds(grad(G))
            # println("scomplex #3 (dVds)")
            scomplex0 = tracefidcomplex(w.vr, -w.vi, p.dVds_r, p.dVds_i)
        #     println("Unknown sv_type = ", p.sv_type)
        end

        if p.pFidType == 1
            scomplex0 = exp(1im*p.globalPhase) - scomplex0
        end


        # Set initial condition for adjoint variables
        # Note (p.Utarget_r, p.Utarget_i) needs to be changed to dV/ds for continuation applications
        # By default, dVds = vtarget
        if p.sv_type == 1 # regular case
            # println("init_adjoint #1 (dVds)")
            init_adjoint!(p.pFidType, p.globalPhase, p.N, scomplex0, w.lambdar, w.lambdar0, w.lambdar05, w.lambdai, w.lambdai0,
                        p.Utarget_r, p.Utarget_i)
        elseif p.sv_type == 2 # term1 for d/ds(grad(G))
            # println("init_adjoint #2 (dVds)")
            init_adjoint!(p.pFidType, p.globalPhase, p.N, scomplex0, w.lambdar, w.lambdar0, w.lambdar05, w.lambdai, w.lambdai0,
                        p.dVds_r, p.dVds_i)               
        elseif p.sv_type == 3 # term2 for d/ds(grad(G))
            init_adjoint!(p.pFidType, p.globalPhase, p.N, scomplex0, w.lambdar, w.lambdar0, w.lambdar05, w.lambdai, w.lambdai0,
                        p.Utarget_r, p.Utarget_i)
        end

        #Initialize adjoint variables without forcing
        if p.objFuncType != 1
            w.lambdar_nfrc  .= w.lambdar
            w.lambdar0_nfrc .= w.lambdar0
            w.lambdai_nfrc  .= w.lambdai
            w.lambdai0_nfrc .= w.lambdai0
            w.lambdar05_nfrc.= w.lambdar05
            infidelgrad = zeros(gradSize);
        end

        #Backward time stepping loop
        for step in p.nsteps-1:-1:0

            # Forcing for the real part of the adjoint variables in first PRK "stage"
            mul!(w.hr0, p.wmat_real, w.vr, tinv, 0.0)

            #loop over stages
            for q in 1:stages
                t0 = t
                copy!(w.vr0,w.vr)
                
                # update K and S
                # Since t is negative we have that w.K0 is K^{n+1}, w.K05 = K^{n-1/2}, 
                # w.K1 = K^{n} and similarly for S.
                # general case
                KS!(w.K0, w.S0, t, p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, splinepar, p.Hconst, p.Rfreq) 
                KS!(w.K05, w.S05, t + 0.5*dt*gamma[q], p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, splinepar, p.Hconst, p.Rfreq) 
                KS!(w.K1, w.S1, t + dt*gamma[q], p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, splinepar, p.Hconst, p.Rfreq) 


                # Integrate state variables backwards in time one step
                @inbounds t = step!(t, w.vr, w.vi, w.vi05, dt*gamma[q], w.K0, w.S0, w.K05, w.S05, w.K1, w.S1, p.Ident, w.κ₁, w.κ₂, w.ℓ₁, w.ℓ₂, w.rhs, p.linear_solver)

                # Forcing for adjoint equations (real part of forbidden state penalty)
                mul!(w.hi0,p.wmat_real,w.vi05,tinv,0.0)
                mul!(w.hr1,p.wmat_real,w.vr,tinv,0.0)

                # Forcing for adjoint equations (imaginary part of forbidden state penalty)
                mul!(w.hr1,p.wmat_imag,w.vi05,tinv,1.0)
                copy!(w.hi1,w.hi0)
                mul!(w.hi1,p.wmat_imag,w.vr,-tinv,1.0)

                # evolve w.lambdar, w.lambdai
                temp = t0
                @inbounds temp = step!(temp, w.lambdar, w.lambdai, w.lambdar05, dt*gamma[q], w.hr0, w.hi0, w.hr1, w.hi1, 
                                        w.K0, w.S0, w.K05, w.S05, w.K1, w.S1, p.Ident, w.κ₁, w.κ₂, w.ℓ₁, w.ℓ₂, w.rhs, p.linear_solver)

                # Accumulate gradient
                adjoint_grad_calc!(p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, w.vr0, w.vi05, w.vr, 
                                    w.lambdar0, w.lambdar05, w.lambdai, w.lambdai0, t0, dt,splinepar, w.gr, w.gi, w.tr_adj) 
                axpy!(gamma[q]*dt,w.tr_adj,w.gradobjfadj)
                
                # save for next stage
                copy!(w.lambdai0,w.lambdai)
                copy!(w.lambdar0,w.lambdar)

                #Do adjoint step to compute infidelity grad (without forcing)
                if p.objFuncType != 1
                    temp = t0
                    @inbounds temp = step_no_forcing!(temp, w.lambdar_nfrc, w.lambdai_nfrc, w.lambdar05_nfrc, dt*gamma[q], 
                                                    w.K0, w.S0, w.K05, w.S05, w.K1, w.S1, p.Ident, w.κ₁, w.κ₂, w.ℓ₁, w.ℓ₂, w.rhs, p.linear_solver)

                    # Accumulate gradient
                    adjoint_grad_calc!(p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, w.vr0, w.vi05, w.vr, 
                                        w.lambdar0_nfrc, w.lambdar05_nfrc, w.lambdai_nfrc, w.lambdai0_nfrc, t0, dt,splinepar, w.gr, w.gi, w.tr_adj) 
                    axpy!(gamma[q]*dt,w.tr_adj,infidelgrad)
                    
                    # save for next stage
                    copy!(w.lambdai0_nfrc,w.lambdai_nfrc)
                    copy!(w.lambdar0_nfrc,w.lambdar_nfrc)    
                end

            end #for stages
        end # for step (backward time stepping loop)

        primObjGradPhase=0.0

        # totalgrad = zeros(Psize) # allocate array to return the gradient
        # totalgrad[:] = gradobjfadj[:] # deep copy        
        totalgrad = w.gradobjfadj # use a shallow copy for improved efficiency

        if p.objFuncType != 1
            leakgrad = zeros(size(totalgrad));
            leakgrad .= totalgrad - infidelgrad # deep copy
        else
            # add in Tikhonov gradient
            tikhonov_grad!(alpha, p, w.gr)  
            axpy!(1.0, w.gr, totalgrad)
        end
    
    end # if evaladjoint

    if verbose
        tikhonovpenalty = objfv - primaryobjf - secondaryobjf
        println("Total objective func: ", objfv)
        println("Primary objective func: ", primaryobjf, " Guard state penalty: ", secondaryobjf, " Tikhonov penalty: ", tikhonovpenalty)
        if evaladjoint
            println("Norm of adjoint gradient = ", norm(w.gradobjfadj))

            if p.kpar <= p.nAlpha
                dfdp = dfdp + w.gr[p.kpar] # add in the tikhonov term

                println("Forward integration of total gradient[kpar=", p.kpar, "]: ", dfdp);
                println("Adjoint integration of total gradient[kpar=", p.kpar, "]: ", w.gradobjfadj[p.kpar]);
                println("\tAbsolute Error in gradients is : ", abs(dfdp - w.gradobjfadj[p.kpar]))
                println("\tRelative Error in gradients is : ", abs((dfdp - w.gradobjfadj[p.kpar])/norm(w.gradobjfadj)))
                println("\tPrimary grad = ", primaryobjgrad, " Tikhonov penalty grad = ", w.gr[p.kpar], " Guard state grad = ", dfdp - w.gr[p.kpar] - primaryobjgrad )
            else
                println("The gradient with respect to the phase angle is computed analytically and not by solving the adjoint equation")
            end
            if p.pFidType == 4
                println("\tPrimary grad wrt phase = ", primObjGradPhase)
            end
        end
        
        nlast = 1 + p.nsteps
        println("Unitary test 1, error in length of propagated state vectors:")
        println("Col |   (1 - |psi|)")
        Vnrm ::Float64 = 0.0
        for q in 1:p.N
            Vnrm = usaver[:,q,nlast]' * usaver[:,q,nlast] + usavei[:,q,nlast]' * usavei[:,q,nlast]
            Vnrm = sqrt(Vnrm)
            println("  ", q, " |  ", 1.0 - Vnrm)
        end

        # output primary objective function (infidelity at final time)
        fidelityrot = tracefidcomplex(w.vfinalr, w.vfinali, p.Utarget_r, p.Utarget_i) # w.vfinali = -w.vi
        mfidelityrot = abs(fidelityrot)^2
        println("Final trace infidelity = ", 1.0 - mfidelityrot, " trace fidelity = ", mfidelityrot)
        
        if p.usingPriorCoeffs
        println("Relative difference from prior: || alpha-prior || / || alpha || = ", norm(alpha - p.priorCoeffs) / norm(alpha) )
        end
        
        
        # Also output L2 norm of last energy level
        if p.Ntot>p.N
            #       normlastguard = zeros(p.N)
            forbLev = identify_forbidden_levels(p)
            maxLev = zeros(p.Ntot)
            for lev in 1:p.Ntot
                maxpop = zeros(p.N)
                if forbLev[lev]
                    for q in 1:p.N
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
        return objfv, copy(totalgrad), usaver+1im*usavei, mfidelityrot, dfdp, wr1 - 1im*wi
    elseif verbose
        println("Returning from traceobjgrad with objfv, unitary history, fidelity")
        return objfv, usaver+1im*usavei, mfidelityrot
    elseif evaladjoint
        return objfv, copy(totalgrad), primaryobjf, secondaryobjf, traceFidelity, infidelgrad, leakgrad
    else
        return objfv, primaryobjf, secondaryobjf
    end 
end # function traceobjgrad

#########################################################
function initialize_working_arrays(w::working_arrays)
    w.vi05 .= 0.0
    w.vr0  .= 0.0

    w.κ₁   .= 0.0
    w.κ₂   .= 0.0
    w.ℓ₁   .= 0.0
    w.ℓ₂   .= 0.0
    w.rhs  .= 0.0

    w.gr0  .= 0.0
    w.gi0  .= 0.0
    w.gr1  .= 0.0
    w.gi1  .= 0.0
    w.hr0  .= 0.0
    w.hi0  .= 0.0
    w.hi1  .= 0.0
    w.hr1  .= 0.0
    w.gr   .= 0.0
    w.gi   .= 0.0
end

#########################################################
function get_Winit_index(p::objparams, kpar::Int64, verbose::Bool = false)
    if kpar < 1 || kpar > p.nCoeff
        if verbose
            println("kpar = ", kpar, " is out of bounds")
        end
        interv = -9999
        real_imag0 = -1
        row = -1
        col = -1
    elseif kpar <= p.nAlpha
        if verbose
            println("kpar = ", kpar, " corresponds to a B-spline coefficient")
        end
        interv = -9999
        real_imag0 = -1
        row = -1
        col = -1
    else
        kpar0 = kpar - 1
        inter0 = div(kpar0-p.nAlpha, p.nWinit)
        interv = inter0 + 1

        offset0 = (kpar0 - p.nAlpha) % p.nWinit
        nT2 = p.Ntot^2
        real_imag0 = div(offset0, nT2)
        offset0 -= real_imag0 * nT2
        
        j0 = div(offset0, p.Ntot)
        i0 = offset0 % p.Ntot 
        
        row = i0+1
        col = j0+1
        if verbose
            println("kpar = ", kpar, " corresponds to a Winit element in interval = ", interv, " real/imag = ", real_imag0, " matrix element = [", row, ", ", col, "]")
        end        
    end

    return interv, real_imag0, row, col
end

#########################################################
#
# Augmented Lagrange method with multiple time intervals
# Evaluate the infidelity and all continuity constraints
#
#########################################################
function lagrange_objgrad(pcof0::Array{Float64,1},  p::objparams, verbose::Bool = true, evaladjoint::Bool = false)
    ###############
    # TODO: add in the term <lambda, (Uend - W)>_F
    ###############

    # shortcut to working_arrays object in p::objparams
    w = p.wa

    if evaladjoint
        objf_grad = zeros(p.nCoeff) # allocate storage for the gradient
    end

    if verbose
        println("lagrange_obj_grad: Vector dim Ntot =", p.Ntot , ", Guard levels Nguard = ", p.Nguard , ", Param dim, Psize = ", p.nCoeff, ", Spline coeffs per func, D1= ", p.D1, " Tikhonov coeff: ", p.tik0)
        if evaladjoint
            if p.nCoeff > p.nAlpha
                println("Objective depends on W-initial conditions, nIntervals = ", p.nTimeIntervals)
                if p.kpar <= p.nAlpha
                    println("kpar = ", p.kpar, " corresponds to a B-spline coefficient")
                else
                    println("kpar = ", p.kpar, " corresponds to intermediate initial conditions")
                end
            else
                println("Objective does NOT depend on W, nIntervals = ", p.nTimeIntervals)
            end
        end
    end

    # initializations start here
    alpha = pcof0[1:p.nAlpha] # extract the B-spline-coefficients

    # setup splinepar
    if p.use_bcarrier
        splinepar = bcparams(p.T, p.D1, p.Cfreq, alpha) # Assumes Nunc = 0
    else
        Nsig  = 2*(p.Ncoupled + p.Nunc) # Only uses for regular B-splines
        splinepar = splineparams(p.T, p.D1, Nsig, alpha)
    end

    dt ::Float64 = p.T/p.nsteps # global time step

    if verbose
        println("Final time: ", p.T, ", total number of time steps: " , p.nsteps , ", time step: " , dt )
        println("lagrange_objgrad: length(pcof) =  ", length(pcof0), " nAlpha = ", p.nAlpha, " nWinit = ", p.nWinit)
    end
    
    # Zero out working arrays
    initialize_working_arrays(w)

    # Allocate storage for saving the unitary at the end of each time interval
    Uend_r = Matrix{Float64}(undef, p.Ntot, p.Ntot)
    Uend_i = Matrix{Float64}(undef, p.Ntot, p.Ntot)

    # Total objective
    objf = 0.0
    grad_kpar = 0.0

    eval1gradient = verbose && evaladjoint # for testing the adjoint gradient

    # Split the time stepping into independent tasks in each time interval
    for interval = 1:p.nTimeIntervals
        tEnd = p.T0int[interval] + p.Tsteps[interval]*dt # terminal time for this time interval

        if interval == 1
            # initial conditions from Uinit (fixed)
            Winit_r = p.Uinit_r
            Winit_i = p.Uinit_i
        else
            # initial conditions from pcof0 (determined by optimization)
            offc = p.nAlpha + (interval-2)*p.nWinit # for interval = 2 the offset should be nAlpha
            # println("offset 1 = ", offc)
            nMat = p.Ntot^2
            Winit_r = reshape(pcof0[offc+1:offc+nMat], p.Ntot, p.Ntot)
            offc += nMat
            # println("offset 2 = ", offc)
            Winit_i = reshape(pcof0[offc+1:offc+nMat], p.Ntot, p.Ntot)
        end

        # Evolve the state under Schroedinger's equation
        # NOTE: the S-V scheme treats the real and imaginary parts with different time integrators
        # First compute the solution operator for a basis of real initial conditions: I
        reInitOp = evolve_schroedinger(p, splinepar, p.T0int[interval], p.Uinit_r, p.Uinit_i, p.Tsteps[interval], eval1gradient)
        
        # Then a basis for purely imaginary initial conditions: iI
        imInitOp = evolve_schroedinger(p, splinepar, p.T0int[interval], p.Uinit_i, p.Uinit_r, p.Tsteps[interval], eval1gradient)
        
        # Now we can  account for the initial conditions for this time interval and easily calculate the gradient wrt Winit
        # Uend = (reInitop[1] + i*reInitOp[2]) * Winit_r + (imInitOp[1] + i*imInitOp[2]) * Winit_i
        Uend_r = (reInitOp[1] * Winit_r + imInitOp[1] * Winit_i) # real part of above expression
        Uend_i = (reInitOp[2] * Winit_r + imInitOp[2] * Winit_i) # imaginary part

        scomplex0 = tracefidcomplex(Uend_r, Uend_i, p.Utarget_r, p.Utarget_i) # scaled by 1/N

        if eval1gradient # test 1 component of the gradient
            # figure out if kpar corresponds to a B-spl coeff or a Winit matrix
            interval_kp, real_imag_kp, row_kp, col_kp = get_Winit_index(p, p.kpar, true)

            # Then account for the initial conditions for this time interval
            dUda_r = (reInitOp[3] * Winit_r + imInitOp[3] * Winit_i)
            dUda_i = (reInitOp[4] * Winit_r + imInitOp[4] * Winit_i)

            if interval < p.nTimeIntervals
                # gradient wrt alpha (B-spline coefficients)

                # get initial condition offset in pcof0 array
                offc = p.nAlpha + (interval-1)*p.nWinit # for interval = 1 the offset should be nAlpha
                nMat = p.Ntot^2
                Wend_r = reshape(pcof0[offc+1:offc+nMat], p.Ntot, p.Ntot)
                offc += nMat
                Wend_i = reshape(pcof0[offc+1:offc+nMat], p.Ntot, p.Ntot)

                # test quadratic jump gradient
                Cjump_r = Uend_r - Wend_r
                Cjump_i = Uend_i - Wend_i

                dCda_kpar = p.gammaJump*p.N*real(tracefidcomplex(Cjump_r, Cjump_i, dUda_r, dUda_i))

                # test Lagrange multiplier gradient
                dLda_kpar = -p.N*real(tracefidcomplex(p.Lmult_r[interval], p.Lmult_i[interval], dUda_r, dUda_i)) # Assumes interval = 1

                grad_kpar += dCda_kpar + dLda_kpar

                dFda_kpar = 0.0 # only the last interval contributes to the infidelity
                dFdW_kpar = 0.0 

                # gradient wrt Winit
                if interval == interval_kp # gradient wrt W^{(1)}
                    # dependence through Cjump^{interval} = U^{interval} - W^{interval_kp}
                    if real_imag_kp == 0 # real part
                        dCdW_kpar = -p.gammaJump*Cjump_r[row_kp, col_kp]
                        dLdW_kpar = p.Lmult_r[interval][row_kp, col_kp]
                    else # imaginary part
                        dCdW_kpar = -p.gammaJump*Cjump_i[row_kp, col_kp]
                        dLdW_kpar = p.Lmult_i[interval][row_kp, col_kp]
                    end
                elseif interval == interval_kp+1
                    # p = row_kp, q = col_kp
                    c_rq = Cjump_r[:, col_kp]
                    c_iq = Cjump_i[:, col_kp]
                    la_rq = p.Lmult_r[interval][:, col_kp]
                    la_iq = p.Lmult_i[interval][:, col_kp]
                    # dependence through initial condition Cjump^{interval} = U^{interval}(W^{interval_kp}) - W^{interval}
                    if real_imag_kp == 0 # real part
                        s_rp = reInitOp[1][:, row_kp]
                        s_ip = reInitOp[2][:, row_kp]

                    else # imaginary part
                        s_rp = imInitOp[1][:, row_kp]
                        s_ip = imInitOp[2][:, row_kp]
                    end
                    dCdW_kpar = p.gammaJump*( s_rp' * c_rq + s_ip' * c_iq)
                    dLdW_kpar = -( s_rp' * la_rq + s_ip' * la_iq)
                else
                    dCdW_kpar = 0.0
                    dLdW_kpar = 0.0
                end # if interval
                
                grad_kpar += dCdW_kpar + dLdW_kpar
            else # last interval
                # test infidelity gradient wrt control parameter p.kpar
                salpha1 = tracefidcomplex(dUda_r, dUda_i, p.Utarget_r, p.Utarget_i) # scaled by 1/N

                dFda_kpar = -2*real(conj(scomplex0)*salpha1) # gradient of infidelity NOTE: minus sign
                grad_kpar += dFda_kpar

                dCda_kpar = 0.0 # there is no quadratic penalty or lagrange multiplier term from the last interval
                dLda_kpar = 0.0
                dCdW_kpar = 0.0
                dLdW_kpar = 0.0

                # tmp this one is a bit complicated...
                if interval == interval_kp+1
                    # p = row_kp, q = col_kp
                    v_rq = p.Utarget_r[:, col_kp]
                    v_iq = p.Utarget_i[:, col_kp]
                    # dependence through initial condition Cjump^{interval} = U^{interval}(W^{interval_kp}) - W^{interval}
                    if real_imag_kp == 0 # real part
                        s_rp = reInitOp[1][:, row_kp]
                        s_ip = reInitOp[2][:, row_kp]

                    else # imaginary part
                        s_rp = imInitOp[1][:, row_kp]
                        s_ip = imInitOp[2][:, row_kp]
                    end
                    sW1 = (s_rp' * v_rq + s_ip' * v_iq) + im*( s_rp' * v_iq - s_ip' * v_rq )
                    dFdW_kpar = -2*real( conj(scomplex0) * sW1 )/p.N # Note: scomplex0 is scaled by 1/N
                else
                    dFdW_kpar = 0.0
                end
                
                grad_kpar += dFdW_kpar
            end
        end # if eval1gradient

        if interval < p.nTimeIntervals
            offc = p.nAlpha + (interval-1)*p.nWinit # for interval = 1 the offset should be nAlpha
            # println("offset 1 = ", offc)
            nMat = p.Ntot^2
            Wend_r = reshape(pcof0[offc+1:offc+nMat], p.Ntot, p.Ntot)
            offc += nMat
            # println("offset 2 = ", offc)
            Wend_i = reshape(pcof0[offc+1:offc+nMat], p.Ntot, p.Ntot)

            Cjump_r = Uend_r - Wend_r
            Cjump_i = Uend_i - Wend_i

            # Jump in state at the end of interval k equals C^k = Uend^k - Wend^k (Ntot x Ntot matrices)
            # evaluate continuity constraint (Frobenius norm squared of mismatch)
            cont_2 = 0.5*p.gammaJump * (norm( Cjump_r )^2 + norm( Cjump_i )^2)

            objf += cont_2 # accumulate contributions to the augemnted Lagrangian

            # println("Sizes: p.Lmult_r = ", size(p.Lmult_r[interval]), " Uend_r = ", size(Uend_r), " Wend_r = ", size(Wend_r))
            Lmult_cont = -p.N*real(tracefidcomplex(p.Lmult_r[interval], p.Lmult_i[interval], Cjump_r, Cjump_i))
            if verbose
                println("Interval # ", interval, " Continuity jump (norm2) = ", cont_2, " (Lagrange multiplier) x (continuity jump) = ", Lmult_cont)
            end
            
            objf += Lmult_cont # accumulate contributions to the augemnted Lagrangian

            if evaladjoint            
                # gradient wrt Winit^{(interval)} through Cjump = Uend - Winit
                ws_grad = zeros(p.nCoeff) # workspace
                # contribution to gradient wrt Winit^{(interval)} from Wend in Cjump and Lmult
                offc = p.nAlpha + (interval-1)*p.nWinit # for interval = 1 the offset should be nAlpha
                ws_grad[offc+1:offc+nMat] = vec(-p.gammaJump * Cjump_r + p.Lmult_r[interval]) # real part
                offc += nMat
                ws_grad[offc+1:offc+nMat] = vec(-p.gammaJump * Cjump_i + p.Lmult_i[interval]) # imaginary part
                
                objf_grad += ws_grad # accumulate total gradient

                if interval >= 2 # gradient wrt Winit^{(interval - 1)} through initial condition for (Uend_r, Uend_i)
                    offc_r = p.nAlpha + (interval-2)*p.nWinit # for interval = 2 the offset should be nAlpha
                    offc_i = offc_r + nMat
                    
                    # p = row, q = col
                    for col in 1:p.N
                        c_rq = Cjump_r[:, col]
                        c_iq = Cjump_i[:, col]
                        la_rq = p.Lmult_r[interval][:, col]
                        la_iq = p.Lmult_i[interval][:, col]
                        # dependence through initial condition 
                        # Cjump^{interval} = U^{interval}(W^{interval - 1}) - W^{interval}
                         
                        # real part: vectorize over 'row'
                        s_rp = reInitOp[1]
                        s_ip = reInitOp[2]
                        objf_grad[offc_r + 1: offc_r + p.N] += p.gammaJump*( s_rp' * c_rq + s_ip' * c_iq) - ( s_rp' * la_rq + s_ip' * la_iq) 
                        
                        # imaginary part: vectorize over row
                        s_rp = imInitOp[1]
                        s_ip = imInitOp[2]
                        objf_grad[offc_i + 1: offc_i + p.N] += p.gammaJump*( s_rp' * c_rq + s_ip' * c_iq) - ( s_rp' * la_rq + s_ip' * la_iq)

                        offc_r += p.N
                        offc_i += p.N
                    end # for col
                end # if interval >= 2

                # End gradient wrt Wend 

                # gradient wrt alpha (B-spline coefficients)

                # adjoint gradient of quadratic jump
                Amat_r = p.gammaJump*Cjump_r
                Amat_i = p.gammaJump*Cjump_i
                # Calculate gradients
                quadGrad = adjoint_gradient(p, splinepar, tEnd, p.Tsteps[interval], Uend_r, Uend_i, Amat_r, Amat_i)
                
                objf_grad += quadGrad # accumulate gradient

                # adjoint gradient of Lagrange mult.
                Amat_r = -p.Lmult_r[interval]
                Amat_i = -p.Lmult_i[interval]
                lagrangeGrad = adjoint_gradient(p, splinepar, tEnd, p.Tsteps[interval], Uend_r, Uend_i, Amat_r, Amat_i)
                
                objf_grad += lagrangeGrad # accumulate gradient

                if eval1gradient
                    println("kpar = ", p.kpar, " dLda_kpar = ", dLda_kpar, " dLda_adj = ", lagrangeGrad[p.kpar]," diff = ", dLda_kpar - lagrangeGrad[p.kpar])
                    println("kpar = ", p.kpar, " dCda_kpar = ", dCda_kpar, " dCda_adj = ", quadGrad[p.kpar]," diff = ", dCda_kpar - quadGrad[p.kpar])

                    println("dCdW_kpar = ", dCdW_kpar, " dLdW_kpar = ", dLdW_kpar, " dFdW_kpar = ", dFdW_kpar)
                    println("Fwd grad_kpar = ", grad_kpar, " adjoint_grad_kpar = ", objf_grad[p.kpar], " diff = ", grad_kpar - objf_grad[p.kpar])
                end
            end
        else # final time interval
            traceInfid = (1.0-tracefidabs2(Uend_r, Uend_i, p.Utarget_r, p.Utarget_i))
            
            objf += traceInfid
            
            if verbose
                println("Interval # ", interval, " Infidelity = ", traceInfid)
            end 

            if evaladjoint
                # Gradient of infidelity

                # Gradient wrt Winit^{interval -  1} through initial condition for Uend
                offc_r = p.nAlpha + (interval-2)*p.nWinit # for interval = 2 the offset should be nAlpha
                offc_i = offc_r + nMat

                # p = row, q = col
                for col in 1:p.N
                    v_rq = p.Utarget_r[:, col]
                    v_iq = p.Utarget_i[:, col]
                     
                    # real part: vectorize over 'row'
                    s_rp = reInitOp[1]
                    s_ip = reInitOp[2]
                    sW1 = (s_rp' * v_rq + s_ip' * v_iq) + im*( s_rp' * v_iq - s_ip' * v_rq )
                        
                    objf_grad[offc_r + 1: offc_r + p.N] += -2*real( conj(scomplex0) * sW1 )/p.N
                    
                    # imaginary part: vectorize over row
                    s_rp = imInitOp[1]
                    s_ip = imInitOp[2]
                    sW1 = (s_rp' * v_rq + s_ip' * v_iq) + im*( s_rp' * v_iq - s_ip' * v_rq )
                    
                    objf_grad[offc_i + 1: offc_i + p.N] += -2*real( conj(scomplex0) * sW1 )/p.N

                    offc_r += p.N
                    offc_i += p.N
                end # for col

                # gradient wrt alpha (B-spline coefficients)
                Amat = -2*conj(scomplex0) * (p.Utarget_r + im*p.Utarget_i)/p.N # for infidelity gradient
                # Calculate gradients
                infidGrad = adjoint_gradient(p, splinepar, tEnd, p.Tsteps[interval], Uend_r, Uend_i, real(Amat), imag(Amat))
                
                objf_grad += infidGrad # accumulate gradient

                if eval1gradient
                    println("kpar = ", p.kpar, " dFda_kpar = ", dFda_kpar)
                    println("kpar = ", p.kpar, " dFda_adj = ", infidGrad[p.kpar]," diff = ", dFda_kpar - infidGrad[p.kpar])

                    println("dCdW_kpar = ", dCdW_kpar, " dLdW_kpar = ", dLdW_kpar, " dFdW_kpar = ", dFdW_kpar)
                    println("Fwd grad_kpar = ", grad_kpar, " adjoint_grad_kpar = ", objf_grad[p.kpar], " diff = ", grad_kpar - objf_grad[p.kpar])
                end
            end
        end

    end # for interval...

    # tp = tikhonov_pen(alpha, p) # Tikhonov penalty
    # objf += tp
    # println("Tikhonov penalty = ", tp)

    if verbose
        println("lagrange_objgrad():, objf = ", objf)
    end

    if evaladjoint
        if eval1gradient
            println("kpar = ", p.kpar, " adjointGrad[kpar] = ", objf_grad[p.kpar], " Fwd grad_kpar = ", grad_kpar, " diff = ", grad_kpar - objf_grad[p.kpar])
        end
        return objf, copy(objf_grad) # ipopt interface requires a copy of objf_grad?
    else
        return objf
    end

end # function lagrange_objgrad

##################################################
function evolve_schroedinger(p::objparams, splinepar::BsplineParams, tStart::Float64, Winit_r::Matrix{Float64}, Winit_i::Matrix{Float64}, N_time_steps::Int64, eval1gradient::Bool = false)
    
    # shortcut to working_arrays object in p::objparams
    w = p.wa
    
    # Zero out working arrays
    initialize_working_arrays(w)
    
    if eval1gradient
        # for computing gradient with forward method
        wr   = zeros(Float64, p.Ntot, p.N) 
        wi   = zeros(Float64, p.Ntot, p.N) 
        wr1  = zeros(Float64, p.Ntot, p.N) 
        wi05 = zeros(Float64, p.Ntot, p.N) 
    end

    tinv = 1.0/p.T

    # global time step (the same in all intervals)
    dt = p.T/p.nsteps

    #initialize variables for time stepping
    t = tStart
    step = 0
    objfv = 0.0

    # assign initial conditions
    copy!(w.vr, Winit_r)
    copy!(w.vi,-Winit_i) # note the negative sign

    # Forward time stepping loop
    for step in 1:N_time_steps

        forbidden0 = tinv*penalf2aTrap(w.vr, p.wmat_real)
        # Störmer-Verlet
        copy!(w.vr0, w.vr)
        t0 = t # starting time for this time step (needed to evaluate gradient)
        
        # Update K and S matrices
        # general case
        KS!(w.K0, w.S0, t, p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, splinepar, p.Hconst, p.Rfreq) 
        KS!(w.K05, w.S05, t + 0.5*dt*1.0, p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, splinepar, p.Hconst, p.Rfreq) 
        KS!(w.K1, w.S1, t + dt*1.0, p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, splinepar, p.Hconst, p.Rfreq) 
        
        # Take a step forward and accumulate weight matrix integral. Note the √2 multiplier is to account
        # for the midpoint rule in the numerical integration of the imaginary part of the signal.
        @inbounds t = step!(t, w.vr, w.vi, w.vi05, dt*1.0, w.K0, w.S0, w.K05, w.S05, w.K1, w.S1, p.Ident, w.κ₁, w.κ₂, w.ℓ₁, w.ℓ₂, w.rhs, p.linear_solver)

        forbidden = tinv*penalf2a(w.vr, w.vi05, p.wmat_real)  
        forbidden_imag1 = tinv*penalf2imag(w.vr0, w.vi05, p.wmat_imag)
        objfv = objfv + 1.0*dt*0.5*(forbidden0 + forbidden - 2.0*forbidden_imag1)

        # Keep prior value for next step (FG: will this work for multiple stages?)
        forbidden0 = forbidden

        # compute 1 component (p.kpar) of the gradient for verification of adjoint method
        if eval1gradient
            # compute the forcing for (wr, wi)
            fgradforce!(p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc,
                        p.isSymm, w.vr0, w.vi05, w.vr, t-dt, dt, splinepar, p.kpar, w.gr0, w.gr1, w.gi0, w.gi1, w.gr, w.gi)

            copy!(wr1,wr)

            @inbounds step_fwdGrad!(t0, wr1, wi, wi05, dt*1.0,
                                    w.gr0, w.gi0, w.gr1, w.gi1, w.K0, w.S0, w.K05, w.S05, w.K1, w.S1, p.Ident, w.κ₁, w.κ₂, w.ℓ₁, w.ℓ₂, w.rhs, p.linear_solver)               

            copy!(wr,wr1)
        end  # evaladjoint && verbose
        
    end # end forward time stepping loop

    # Final state is stored in (w.vr, -w.vi)

    if eval1gradient
        return copy(w.vr), -copy(w.vi), wr, -wi
    else
        return copy(w.vr), -copy(w.vi)
    end

end # function evolve_schroedinger

########################################
# Solve the adjoint Stormer-Verlet scheme, assuming NO leak term
########################################
function adjoint_gradient(p::objparams, splinepar::BsplineParams, tEnd::Float64, N_time_steps::Int64, Uend_r::Matrix{Float64}, Uend_i::Matrix{Float64}, Amat_r::Matrix{Float64}, Amat_i::Matrix{Float64})
    
    # shortcut to working_arrays object in p::objparams
    w = p.wa
    
    # initialize array for storing the adjoint gradient so it can be returned to the calling function/program
    w.gradobjfadj[:] .= 0.0
    
    # Terminal time
    t = tEnd

    # Terminal conditions
    copy!(w.vr, Uend_r)
    copy!(w.vi, -Uend_i) # note the sign

    # Zero out working arrays
    initialize_working_arrays(w) 

    # global time step (the same in all intervals)
    dt = - p.T/p.nsteps # going backwards

    # Set terminal conditions for adjoint variables
    set_adjoint_termial_cond!(Amat_r, -Amat_i, w.lambdar, w.lambdar0, w.lambdar05, w.lambdai, w.lambdai0)

    #Backwards time stepping loop
    for step in N_time_steps-1: -1: 0 # p.nsteps-1:-1:0

        t0 = t
        copy!(w.vr0, w.vr)
        
        # Assign K and S matrices, general case
        # Since dt is negative we have that w.K0 is K^{n+1}, w.K05 = K^{n-1/2}, w.K1 = K^{n} and similarly for S.
        KS!(w.K0, w.S0, t, p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, splinepar, p.Hconst, p.Rfreq) 
        KS!(w.K05, w.S05, t + 0.5*dt*1.0, p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, splinepar, p.Hconst, p.Rfreq) 
        KS!(w.K1, w.S1, t + dt*1.0, p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, splinepar, p.Hconst, p.Rfreq) 

        # Integrate state variables backwards in time one step (w.rhs is a working array, holding the right hand side during linear solves)
        @inbounds t = step!(t, w.vr, w.vi, w.vi05, dt*1.0, w.K0, w.S0, w.K05, w.S05, w.K1, w.S1, p.Ident, w.κ₁, w.κ₂, w.ℓ₁, w.ℓ₂, w.rhs, p.linear_solver)

        # evolve w.lambdar, w.lambdai (w.rhs is a working array, holding the right hand side during linear solves)
        temp = t0
        @inbounds temp = step!(temp, w.lambdar, w.lambdai, w.lambdar05, dt*1.0, w.hr0, w.hi0, w.hr1, w.hi1, w.K0, w.S0, w.K05, w.S05, w.K1, w.S1, p.Ident, w.κ₁, w.κ₂, w.ℓ₁, w.ℓ₂, w.rhs, p.linear_solver)

        # Accumulate gradient
        adjoint_grad_calc!(p.Hsym_ops, p.Hanti_ops, p.Hunc_ops, p.Nunc, p.isSymm, w.vr0, w.vi05, w.vr, w.lambdar0, w.lambdar05, w.lambdai, w.lambdai0, t0, dt, splinepar, w.gr, w.gi, w.tr_adj) 
        
        # axpy!(1.0*dt, w.tr_adj, w.gradobjfadj)
        axpy!(-0.5*dt, w.tr_adj, w.gradobjfadj) # Getting the factor in the adjoint terminal condition right

        # save for next stage
        copy!(w.lambdai0, w.lambdai)
        copy!(w.lambdar0, w.lambdar)

    end # for step (backward time stepping loop)    

    return copy(w.gradobjfadj) # do we need to copy the gradient before returning it?
end

##################################################

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
    NfreqTot = sum(Nfreq) # not used
    nAlpha = 2*D1*NfreqTot # not used

    @printf("zero_start_end!(): Ncoupled = %d, Nfreq = %d, D1 = %d, nAlpha = %d, nCoeff = %d\n", Ncoupled, Nfreq, D1, nAlpha, params.nCoeff)
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
    minCoeff, maxCoeff = assign_thresholds_ctrl_freq(params, maxAmp)

Build vector of parameter min/max constraints that can depend on the control function and carrier wave frequency, 
with `minCoeff = -maxCoeff`.
 
# Arguments
- `params:: objparams`: Struct containing problem definition.
- `maxAmp:: Matrix{Float64}`: `maxAmp[c,f]` is the maximum parameter value for ctrl `c` and frequency `f`
"""
function assign_thresholds_ctrl_freq(params::objparams, maxAmp:: Vector{Vector{Float64}})
    Nfreq = params.Nfreq
    Ncoupled = params.Ncoupled
    Nunc = params.Nunc
    @assert(Nunc == 0)

    D1 = params.D1
    nCoeff = params.nCoeff
    minCoeff = zeros(nCoeff) # Initialize storage
    maxCoeff = zeros(nCoeff)

    # @printf("assign_thresholds Ncoupled = %d, Nfreq = %d, D1 = %d, nAlpha = %d, nCoeff = %d\n", Ncoupled, Nfreq, D1, params.nAlpha, params.nCoeff)
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

    if params.nCoeff > params.nAlpha # Box constraints on the intermediate initial conditions
        wInitBound = 10.0 # Do we need a bound on these elements?
        maxCoeff[params.nAlpha+1: params.nCoeff] .= wInitBound
        minCoeff[params.nAlpha+1: params.nCoeff] .= -wInitBound
    end
    return minCoeff, maxCoeff
end


"""
    minCoeff, maxCoeff = assign_thresholds(params, maxAmp)

Build vector of frequency independent min/max parameter constraints for each control function. Here, `minCoeff = -maxCoeff`.
 
# Arguments
- `params:: objparams`: Struct containing problem definition.
- `maxAmp:: Vector{Float64}`: `maxAmp[c]` is the maximum for ctrl function number `c`. Same bounds for p & q.
"""
function assign_thresholds(params::objparams, maxAmp::Vector{Float64})
    Nfreq = params.Nfreq
    Ncoupled = params.Ncoupled
    Nunc = params.Nunc
    @assert(Nunc == 0)

    D1 = params.D1
    nCoeff = params.nCoeff
    minCoeff = zeros(nCoeff) # Initialize storage
    maxCoeff = zeros(nCoeff)

    #@printf("assign_thresholds: Ncoupled = %d, D1 = %d, nAlpha = %d, nCoeff = %d\n", Ncoupled, D1, params.nAlpha, params.nCoeff)

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

    if params.nCoeff > params.nAlpha # Box constraints on the intermediate initial conditions
        wInitBound = 1.1 # Do we need a bound on these elements?
        maxCoeff[params.nAlpha+1: params.nCoeff] .= wInitBound
        minCoeff[params.nAlpha+1: params.nCoeff] .= -wInitBound
    end

    return minCoeff, maxCoeff
end


# Based on input scomplex0 and (vtargetr, vtargeti):
# Initialize the adjoint variables (lambdar, lambdar0, lambdar05, lambdai, lambdai0) in-place.
function set_adjoint_termial_cond!(amat_r::Array{Float64,M}, amat_i::Array{Float64,M}, lambdar::Array{Float64,M},
    lambdar0::Array{Float64,M}, lambdar05::Array{Float64,M},lambdai::Array{Float64,M}, lambdai0::Array{Float64,M}) where M
    
    lambdar .= amat_r
    lambdar0 .= amat_r
    lambdar05 .= amat_r

    lambdai .= amat_i
    lambdai0 .= amat_i
end

# Based on input scomplex0 and (vtargetr, vtargeti):
# Initialize the adjoint variables (lambdar, lambdar0, lambdar05, lambdai, lambdai0) in-place.
@inline function init_adjoint!(pFidType::Int64, globalPhase::Float64, N::Int64, scomplex0::Complex{Float64}, lambdar::Array{Float64,M},
                               lambdar0::Array{Float64,M}, lambdar05::Array{Float64,M},lambdai::Array{Float64,M}, lambdai0::Array{Float64,M},
                               vtargetr::Array{Float64,M}, vtargeti::Array{Float64,M}) where M
    if pFidType == 2 # standard (trace infidelity)
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
    elseif pFidType == 4
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

    # Test: Only compute egenvalues of the system Hamiltonian. Then increase the sample rate by a factor > 1 to compensate for the control Hamiltonian: Tends to underestimate the number of time steps

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

    ctrlFactor = 1.2 # Heuristic, assuming that the total Hamiltonian is dominated by the system part.
    samplerate1 = ctrlFactor * maxeig*Pmin/(2*pi)
    nsteps = ceil(Int64, T*samplerate1)

    # NOTE: The above estimate does not account for quickly varying signals or a large number of splines. 
    # Double check at least 2-3 points per spline to resolve control function.

    return nsteps
end

# general case to be used after the params object has been defined
"""
    nsteps = calculate_timestep(params, maxCoupled, maxUnc=[], Pmin=40)

Estimate the number of time steps needed for the simulation, when there are uncoupled controls.
 
# Arguments
- `params:: objparams`: Simulation object
- `maxCoupled:: Vector{Float64}`: (kw-arg) Maximum control amplitude for each sym/anti-sym control Hamiltonian
- `maxUnc:: Vector{Float64}`: (Optional kw-arg) Maximum control amplitude for each uncoupled control Hamiltonian
- `Pmin:: Int64`: (Optional kw-arg) Number of time steps per shortest period (assuming a slowly varying Hamiltonian).
"""
function calculate_timestep(params::objparams;maxCoupled::Vector{Float64}, maxUnc::Vector{Float64}=Float64[], Pmin::Int64=40)

    T=params.T
    H0 = params.Hconst 
    Hsym_ops = params.Hsym_ops
    Hanti_ops = params.Hanti_ops
    Hunc_ops = params.Hunc_ops

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
        K1 += maxCoupled[i].*Hsym_ops[i] + im*maxCoupled[i].*Hanti_ops[i]
    end

    # Uncoupled control Hamiltonians
    for i = 1:Nunc
        if(issymmetric(Hunc_ops[i]))
            K1 += maxUnc[i]*Hunc_ops[i]
        elseif(norm(Hunc_ops[i]+Hunc_ops[i]') < 1e-14)
            K1 += im*maxUnc[i].*Hunc_ops[i]
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
    # Double check that at least 2-3 points per spline to resolve control function.

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
function initial_cond_old(Ne::Vector{Int64}, Ng::Vector{Int64}, msb_order::Bool = true)
    Nt = Ne + Ng
    Ntot = prod(Nt)
    @assert(length(Ne) == length(Ng))
    @assert length(Nt) <= 4 "ERROR: initial_cond(): only <= 4 sub-systems is implemented"
    NgTot = sum(Ng)
    N = prod(Ne)
    Ident = Matrix{Float64}(I, Ntot, Ntot)
    U0 = Ident[1:Ntot,1:N] # initial guess

    #adjust initial guess if there are ghost points
    if msb_order
        if length(Nt) == 4
            if NgTot > 0
                col = 0
                m = 0
                for k4 in 1:Nt[4]
                    for k3 in 1:Nt[3]
                        for k2 in 1:Nt[2]
                            for k1 in 1:Nt[1]
                                m += 1
                                # is this a guard level?
                                guard = (k1 > Ne[1]) || (k2 > Ne[2]) || (k3 > Ne[3]) || (k4 > Ne[4])
                                if !guard
                                    col = col+1
                                    U0[:,col] = Ident[:,m]
                                end # if ! guard
                            end #for
                        end # for
                    end # for k3
                end # for k4        
            end # if  
        elseif length(Nt) == 3
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
        # LSB ordering
        if length(Nt) == 4
            if NgTot > 0
                col = 0
                m = 0
                for k1 in 1:Nt[1]
                    for k2 in 1:Nt[2]
                        for k3 in 1:Nt[3]
                            for k4 in 1:Nt[4]    
                                m += 1
                                # is this a guard level?
                                guard = (k1 > Ne[1]) || (k2 > Ne[2]) || (k3 > Ne[3]) || (k4 > Ne[4])
                                if !guard
                                    col = col+1
                                    U0[:,col] = Ident[:,m]
                                end # if ! guard
                            end # for k4
                        end #for
                    end # for
                end # for            
            end # if  
        elseif length(Nt) == 3
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

# setup the initial conditions
"""
    u_init = initial_cond_general(is_ess, Ne, Ng)

Setup a basis of canonical unit vectors that span the essential Hilbert space, setting all guard levels to zero
 
# Arguments
- `is_ess:: Vector{Bool}`: Vector is_ess[j]=true if j corresponds to an essential level
- `Ne:: Array{Int64}`: Array holding the number of essential levels in each system
- `Ng:: Array{Int64}`: Array holding the number of guard levels in each system
"""
function initial_cond_general(is_ess::Vector{Bool}, Ne::Vector{Int64}, Ng::Vector{Int64})
    Nt = Ne + Ng
    Ntot = prod(Nt)
    @assert(length(Ne) == length(Ng))
    NgTot = sum(Ng)
    N = prod(Ne)
    Ident = Matrix{Float64}(I, Ntot, Ntot)
    U0 = Ident[1:Ntot,1:N] # initial guess

    #adjust initial guess if there are ghost points
    if NgTot > 0
        col = 0
        for j = 1:Ntot
            if is_ess[j]
                col += 1
                U0[:,col] = Ident[:,j]
            end
        end
    end

    return U0
end

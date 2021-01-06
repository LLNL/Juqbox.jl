mutable struct parameters
    Nosc   ::Int64          # number of oscillators in the coupled quantum systems
    N      ::Int64          # total number of essential levels
    Nguard ::Int64          # total number of extra levels
    Ne     ::Array{Int64,1} # essential levels for each oscillator
    Ng     ::Array{Int64,1} # guard levels for each oscillator
    Nt     ::Array{Int64,1} # total # levels for each oscillator
    T      ::Float64        # final time

    nsteps       ::Int64    # Number of time steps
    U0           ::Array{Float64,2} # initial condition for each essential state: Should be a basis
    utarget      ::Array{Complex{Float64},2}
    use_bcarrier ::Bool
    Nfreq        ::Int64 # number of frequencies
    om           ::Array{Float64,2} # om[seg,freq]
    kpar         ::Int64   # element of gradient to test
    tik0         ::Float64
#    tik1         ::Float64

    # Drift Hamiltonian
    H0 ::MyRealMatrix     # time-independent part of the Hamiltonian (assumed symmetric)
   
    # Control Hamiltonians
    Hsym_ops  ::Array{MyRealMatrix,1}   # Symmetric control Hamiltonians
    Hanti_ops ::Array{MyRealMatrix,1}   # Anti-symmetric control Hamiltonians
    Hunc_ops  ::Array{MyRealMatrix,1}   # Uncoupled control Hamiltonians

    Ncoupled :: Int64 # Number of coupled Hamiltonians.
    Nunc     :: Int64 # Number of uncoupled Hamiltonians.
    isSymm   :: BitArray{1} # Array to track symmetry of Hunc_ops entries

    Ident ::MyRealMatrix
    wmat  ::Diagonal{Float64,Array{Float64,1}} # Weights for discouraging guard level population 

    # Type of fidelity
    pFidType    ::Int64
    globalPhase ::Float64

    # Convergence history variables
    saveConvHist  ::Bool;
    objHist       ::Array{Float64,1}
    primaryHist   ::Array{Float64,1}
    secondaryHist ::Array{Float64,1}
    dualInfidelityHist  ::Array{Float64,1}

    # Number of terms in truncated Neumann series (not counting identity) 
    # to solve linear system in timestepping
    nNeumann :: Int64

    traceInfidelityThreshold :: Float64
    lastTraceInfidelity :: Float64
    lastLeakIntegral :: Float64    

    usingPriorCoeffs :: Bool
    priorCoeffs ::Array{Float64,1}

    quiet:: Bool # quiet mode?
    
    # Constructor for case with no coupled controls.
    function parameters(Ne::Array{Int64,1}, Ng::Array{Int64,1}, T::Float64, nsteps::Int64,
                        U0::Array{Float64,2}, utarget::Array{Complex{Float64},2}, 
                        om::Array{Float64,2}, H0::AbstractArray, Hunc_ops:: AbstractArray)
        Nosc   = length(Ne)
        N      = prod(Ne)
        Ntot   = prod(Ne+Ng)
        Nguard = Ntot-N
        Nfreq  = size(om,2)
        Ncoupled   = 0
        Nunc   = length(Hunc_ops)
    
        # Track symmetries of uncoupled Hamiltonian terms
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

        # Set default Tikhonov parameters to zero
        tik0 = 0.01
#        tik1 = 0.0

        # By default, test the first parameter for gradient correctness
        kpar = 1

        # By default use B-splines with carrier waves
        use_bcarrier = true

        # Weights in the W matrix for discouraging population of guarded states
        wmatScale = 1.0
        wmat = wmatScale.*Juqbox.wmatsetup(Ne, Ng)

        # By default save convergence history
        saveConvHist = true

        # Exit if uncoupled controls with more than 1 oscillator present
        if(Nunc > 0 && Nosc > 1)
            throw(ArgumentError("Uncoupled Hamiltonians for more than a single oscillator not currently supported.\n"))
        end
        
        # Appropriately type identity
        if(typeof(H0) == SparseMatrixCSC{Float64, Int64})
            Ident = sparse(Matrix{Float64}(I, Ntot, Ntot))
        else 
            Ident = Matrix{Float64}(I, Ntot, Ntot)
        end

        # Default number of Neumann series terms
        nNeumann = 3
        quiet = false
        new(Nosc, N, Nguard, Ne, Ng, Ne+Ng, T, nsteps, U0, utarget, use_bcarrier, Nfreq, om, kpar, tik0, H0, [], [], Hunc_ops, Ncoupled, Nunc, isSymm, Ident, wmat, 2, 0.0, saveConvHist, zeros(0), zeros(0), zeros(0), zeros(0), nNeumann, 0.0, 0.0, 0.0, false, zeros(0), quiet)
    end

    # FG: Streamlined constructor (with uncoupled controls)
    function parameters(Ne::Array{Int64,1}, Ng::Array{Int64,1}, T::Float64, nsteps::Int64,
                        U0::Array{Float64,2}, utarget::Array{Complex{Float64},2}, 
                        om::Array{Float64,2}, H0::Array{Float64,2}, Hsym_ops:: Array{Array{Float64,2},1},
                        Hanti_ops:: Array{Array{Float64,2},1}, Hunc_ops:: Array{Array{Float64,2},1})
        Nosc   = length(Ne)
        N      = prod(Ne)
        Ntot   = prod(Ne+Ng)
        Nguard = Ntot-N
        Nfreq  = size(om,2)
        Ncoupled   = length(Hsym_ops)
        Nanti  = length(Hanti_ops)
        Nunc   = length(Hunc_ops)
    

        # Check for consistency in coupled controls 
        for i = 1:Ncoupled
            L = LinearAlgebra.tril(Hsym_ops[i] + Hanti_ops[i])
            if(norm(L) > eps(1.0))
                throw(ArgumentError("Coupled Hamiltonian inconsistent with currently implemented rotating wave approximation. Please flip sign on Hanti_ops.\n"))
            end
        end

        # Track symmetries of uncoupled Hamiltonian terms
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

        # Set default Tikhonov parameters to zero
        tik0 = 0.01
#        tik1 = 0.0

        # By default, test the first parameter for gradient correctness
        kpar = 1

        # By default use B-splines with carrier waves
        use_bcarrier = true

        # Weights in the W matrix for discouraging population of guarded states
        wmatScale = 1.0
        wmat = wmatScale.*Juqbox.wmatsetup(Ne, Ng)

        # By default save convergence history
        saveConvHist = true

        @assert(Ncoupled == Nanti)

        # Exit if uncoupled controls with more than 1 oscillator present
        if(Nunc > 0 && Nosc > 1)
            throw(ArgumentError("Uncoupled Hamiltonians for more than a single oscillator not currently supported.\n"))
        end
        Ident = Matrix{Float64}(I, Ntot, Ntot)

        # Default number of Neumann series terms
        nNeumann = 3
        quiet = false
        new(Nosc, N, Nguard, Ne, Ng, Ne+Ng, T, nsteps, U0, utarget, use_bcarrier, Nfreq, om, kpar, tik0, H0, Hsym_ops, Hanti_ops, Hunc_ops, Ncoupled, Nunc, isSymm, Ident, wmat, 2, 0.0, saveConvHist, zeros(0), zeros(0), zeros(0), zeros(0), nNeumann, 0.0, 0.0, 0.0, false, zeros(0), quiet)
    end

    # FG: Streamlined constructor (without uncoupled controls)
    function parameters(Ne::Array{Int64,1}, Ng::Array{Int64,1}, T::Float64, nsteps::Int64,
                        U0::Array{Float64,2}, utarget::Array{Complex{Float64},2},
                        om::Array{Float64,2}, H0::Array{Float64,2}, Hsym_ops:: Array{Array{Float64,2},1},
                        Hanti_ops:: Array{Array{Float64,2},1}, wmatScale::Float64 = 1.0)
        pFidType = 2
        Nosc   = length(Ne)
        N      = prod(Ne)
        Ntot   = prod(Ne+Ng)
        Nguard = Ntot-N
        Nfreq  = size(om,2)
        Ncoupled   = length(Hsym_ops)
        Nanti  = length(Hanti_ops)
        Nunc   = 0

        # Set default Tikhonov parameters to zero
        tik0 = 0.01
#        tik1 = 0.0

        # By default, test the first parameter for gradient correctness
        kpar = 1

        # By default use B-splines with carrier waves
        use_bcarrier = true

        # Weights in the W matrix for discouraging population of guarded states
        wmat = wmatScale.*Juqbox.wmatsetup(Ne, Ng)

        # By default save convergence history
        saveConvHist = true

        @assert(Ncoupled == Nanti)

        # Check for consistency in coupled controls 
        for i = 1:Ncoupled
            L = LinearAlgebra.tril(Hsym_ops[i] + Hanti_ops[i])
            if(norm(L) > eps(1.0))
                throw(ArgumentError("Coupled Hamiltonian inconsistent with currently implemented rotating wave approximation. Please flip sign on Hanti_ops.\n"))
            end
        end

        # Exit if uncoupled controls with more than 1 oscillator present
        if(Nunc > 0 && Nosc > 1)
            throw(ArgumentError("Uncoupled Hamiltonians for more than a single oscillator not currently supported.\n"))
        end
        Ident = Matrix{Float64}(I, Ntot, Ntot)
        # Default number of Neumann series terms
        nNeumann = 3
        quiet = false
        new(Nosc, N, Nguard, Ne, Ng, Ne+Ng, T, nsteps, U0, utarget, use_bcarrier, Nfreq, om, kpar, tik0, H0, Hsym_ops, Hanti_ops, [], Ncoupled, Nunc, [], Ident, wmat, pFidType, 0.0, saveConvHist, zeros(0), zeros(0), zeros(0), zeros(0), nNeumann, 0.0, 0.0, 0.0, false, zeros(0), quiet)
    end

    # FG: Streamlined constructor (with uncoupled controls), Sparse
    function parameters(Ne::Array{Int64,1}, Ng::Array{Int64,1}, T::Float64, nsteps::Int64,
                        U0::Array{Float64,2}, utarget::Array{Complex{Float64},2},
                        om::Array{Float64,2}, H0::SparseMatrixCSC{Float64, Int64}, Hsym_ops:: Array{SparseMatrixCSC{Float64, Int64},1},
                        Hanti_ops:: Array{SparseMatrixCSC{Float64, Int64},1}, Hunc_ops:: Array{SparseMatrixCSC{Float64, Int64},1})
        Nosc   = length(Ne)
        N      = prod(Ne)
        Ntot   = prod(Ne+Ng)
        Nguard = Ntot-N
        Nfreq  = size(om,2)
        Ncoupled   = length(Hsym_ops)
        Nanti  = length(Hanti_ops)
        Nunc   = length(Hunc_ops)

        # Check for consistency in coupled controls 
        for i = 1:Ncoupled
            L = LinearAlgebra.tril(Hsym_ops[i] + Hanti_ops[i])
            if(norm(L) > eps(1.0))
                throw(ArgumentError("Coupled Hamiltonian inconsistent with currently implemented rotating wave approximation. Please flip sign on Hanti_ops.\n"))
            end
        end

        # Track symmetries of uncoupled Hamiltonian terms
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

        # Set default Tikhonov parameters to zero
        tik0 = 0.01
#        tik1 = 0.0

        # By default, test the first parameter for gradient correctness
        kpar = 1

        # By default use B-splines with carrier waves
        use_bcarrier = true

        # Weights in the W matrix for discouraging population of guarded states
        wmatScale = 1.0
        wmat = wmatScale.*Juqbox.wmatsetup(Ne, Ng)

        # By default save convergence history
        saveConvHist = true

        @assert(Ncoupled == Nanti)

        # Exit if uncoupled controls with more than 1 oscillator present
        if(Nunc > 0 && Nosc > 1)
            throw(ArgumentError("Uncoupled Hamiltonians for more than a single oscillator not currently supported.\n"))
        end
        Ident = sparse(Matrix{Float64}(I, Ntot, Ntot))
        # Default number of Neumann series terms
        nNeumann = 3
        quiet = false
        new(Nosc, N, Nguard, Ne, Ng, Ne+Ng, T, nsteps, U0, utarget, use_bcarrier, Nfreq, om, kpar, tik0, H0, Hsym_ops, Hanti_ops, Hunc_ops, Ncoupled, Nunc, isSymm, Ident, wmat, 2, 0.0, saveConvHist, zeros(0), zeros(0), zeros(0), zeros(0), nNeumann, 0.0, 0.0, 0.0, false, zeros(0), quiet)
    end

    # FG: Streamlined constructor (without uncoupled controls), Sparse
    function parameters(Ne::Array{Int64,1}, Ng::Array{Int64,1}, T::Float64, nsteps::Int64,
                        U0::Array{Float64,2}, utarget::Array{Complex{Float64},2},
                        om::Array{Float64,2}, H0::SparseMatrixCSC{Float64, Int64}, Hsym_ops::  Array{SparseMatrixCSC{Float64, Int64},1},
                        Hanti_ops::  Array{SparseMatrixCSC{Float64, Int64},1})
        Nosc   = length(Ne)
        N      = prod(Ne)
        Ntot   = prod(Ne+Ng)
        Nguard = Ntot-N
        Nfreq  = size(om,2)
        Ncoupled   = length(Hsym_ops)
        Nanti  = length(Hanti_ops)
        Nunc   = 0

        # Set default Tikhonov parameters to zero
        tik0 = 0.01
#        tik1 = 0.0

        # By default, test the first parameter for gradient correctness
        kpar = 1

        # By default use B-splines with carrier waves
        use_bcarrier = true

        # Weights in the W matrix for discouraging population of guarded states
        wmatScale = 1.0
        wmat = wmatScale.*Juqbox.wmatsetup(Ne, Ng)

        # By default save convergence history
        saveConvHist = true

        @assert(Ncoupled == Nanti)

        # Check for consistency in coupled controls 
        for i = 1:Ncoupled
            L = LinearAlgebra.tril(Hsym_ops[i] + Hanti_ops[i])
            if(norm(L) > eps(1.0))
                throw(ArgumentError("Coupled Hamiltonian inconsistent with currently implemented rotating wave approximation. Please flip sign on Hanti_ops.\n"))
            end
        end

        # Exit if uncoupled controls with more than 1 oscillator present
        if(Nunc > 0 && Nosc > 1)
            throw(ArgumentError("Uncoupled Hamiltonians for more than a single oscillator not currently supported.\n"))
        end
        Ident = sparse(Matrix{Float64}(I, Ntot, Ntot))
        # Default number of Neumann series terms
        nNeumann = 3
        quiet = false
        new(Nosc, N, Nguard, Ne, Ng, Ne+Ng, T, nsteps, U0, utarget, use_bcarrier, Nfreq, om, kpar, tik0, H0, Hsym_ops, Hanti_ops, [], Ncoupled, Nunc, [], Ident, wmat, 2, 0.0, saveConvHist, zeros(0), zeros(0), zeros(0), zeros(0), nNeumann, 0.0, 0.0, 0.0, false, zeros(0), quiet)
    end

    function parameters(Nosc::Int64, N::Int64, Nguard::Int64, Ne::Array{Int64,1}, Ng::Array{Int64,1}, T::Float64, nsteps::Int64,
                        U0::Array{Float64,2}, utarget::Array{Complex{Float64},2}, H0::Array{Float64,2}, Hsym_ops:: Array{Array{Float64,2},1},
                        Hanti_ops:: Array{Array{Float64,2},1}, Hunc_ops:: Array{Array{Float64,2},1}, wmat::Diagonal{Float64,Array{Float64,1}}, 
                        use_bcarrier::Bool, om::Array{Float64,2}, kpar::Int64, tik0::Float64, saveConvHist::Bool)
        Nfreq=size(om,2)
        Ntot = N+Nguard
        Ncoupled = length(Hsym_ops)
        Nanti = length(Hanti_ops)
        Nunc = length(Hunc_ops)
        
        # Check for consistency in coupled controls 
        for i = 1:Ncoupled
            L = LinearAlgebra.tril(Hsym_ops[i] + Hanti_ops[i])
            if(norm(L) > eps(1.0))
                throw(ArgumentError("Coupled Hamiltonian inconsistent with currently implemented rotating wave approximation. Please flip sign on Hanti_ops.\n"))
            end
        end

        # Track symmetries of uncoupled Hamiltonian terms
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
        @assert(Ncoupled == Nanti)

        # Exit if uncoupled controls with more than 1 oscillator present
        if(Nunc > 0 && Nosc >1)
            throw(ArgumentError("Uncoupled Hamiltonians for more than a single oscillator not currently supported.\n"))
        end
        Ident = Matrix{Float64}(I, Ntot, Ntot)
        # Default number of Neumann series terms
        nNeumann = 3
        quiet = false
        new(Nosc, N, Nguard, Ne, Ng, Ne+Ng, T, nsteps, U0, utarget, use_bcarrier, Nfreq, om, kpar, tik0, H0, Hsym_ops, Hanti_ops, Hunc_ops, Ncoupled, Nunc, isSymm, Ident, wmat, 2, 0.0, saveConvHist, zeros(0), zeros(0), zeros(0), zeros(0), nNeumann, 0.0, 0.0, 0.0, false, zeros(0), quiet)
    end

    function parameters(Nosc::Int64, N::Int64, Nguard::Int64, Ne::Array{Int64,1}, Ng::Array{Int64,1}, T::Float64, nsteps::Int64,
                        U0::Array{Float64,2}, utarget::Array{Complex{Float64},2}, H0::SparseMatrixCSC{Float64, Int64},
                        Hsym_ops:: Array{SparseMatrixCSC{Float64, Int64},1}, Hanti_ops:: Array{SparseMatrixCSC{Float64, Int64},1},
                        Hunc_ops:: Array{SparseMatrixCSC{Float64, Int64},1}, wmat::Diagonal{Float64,Array{Float64,1}}, 
                        use_bcarrier::Bool, om::Array{Float64,2}, kpar::Int64, tik0::Float64, saveConvHist::Bool)
        Nfreq=size(om,2)
        Ntot = N+Nguard
        Ncoupled = length(Hsym_ops)
        Nanti = length(Hanti_ops)
        
        # Track symmetries of uncoupled Hamiltonian termsNunc = length(Hunc_ops)
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
        @assert(Ncoupled == Nanti)

        # Check for consistency in coupled controls 
        for i = 1:Ncoupled
            L = LinearAlgebra.tril(Hsym_ops[i] + Hanti_ops[i])
            if(norm(L) > eps(1.0))
                throw(ArgumentError("Coupled Hamiltonian inconsistent with currently implemented rotating wave approximation. Please flip sign on Hanti_ops.\n"))
            end
        end

        # Exit if uncoupled controls with more than 1 oscillator present
        if(Nunc > 0 && Nosc > 1)
            throw(ArgumentError("Uncoupled Hamiltonians for more than a single oscillator not currently supported.\n"))
        end
        Ident = sparse(Matrix{Float64}(I, Ntot, Ntot)) # It needs to be sparse for the sparse time stepper to work
        # Default number of Neumann series terms
        nNeumann = 3
        quiet = false
        new(Nosc, N, Nguard, Ne, Ng, Ne+Ng, T, nsteps, U0, utarget, use_bcarrier, Nfreq, om, kpar, tik0, H0, Hsym_ops, Hanti_ops, Hunc_ops, Ncoupled, Nunc, isSymm, Ident, wmat, 2, 0.0, saveConvHist, zeros(0), zeros(0), zeros(0), zeros(0), nNeumann, 0.0, 0.0, 0.0, false, zeros(0), quiet)
    end


    # If no uncoupled control functions specified or needed, set Nunc=0
    function parameters(Nosc::Int64, N::Int64, Nguard::Int64, Ne::Array{Int64,1}, Ng::Array{Int64,1}, T::Float64, nsteps::Int64,
                        U0::Array{Float64,2}, utarget::Array{Complex{Float64},2}, H0::Array{Float64,2}, Hsym_ops:: Array{Array{Float64,2},1},
                        Hanti_ops:: Array{Array{Float64,2},1}, wmat::Diagonal{Float64,Array{Float64,1}}, 
                        use_bcarrier::Bool, om::Array{Float64,2}, kpar::Int64, tik0::Float64, saveConvHist::Bool)
        Nfreq=size(om,2)
        Ntot = N+Nguard
        Ncoupled = length(Hsym_ops)
        Nanti = length(Hanti_ops)
        Nunc = 0
        Hunc_ops = []
        @assert(Ncoupled == Nanti)
        # Check for consistency in coupled controls 
        for i = 1:Ncoupled
            L = LinearAlgebra.tril(Hsym_ops[i] + Hanti_ops[i])
            if(norm(L) > eps(1.0))
                throw(ArgumentError("Coupled Hamiltonian inconsistent with currently implemented rotating wave approximation. Please flip sign on Hanti_ops.\n"))
            end
        end
        isSymm = []
        Ident = Matrix{Float64}(I, Ntot, Ntot)
        # Default number of Neumann series terms
        nNeumann = 3
        quiet = false
        new(Nosc, N, Nguard, Ne, Ng, Ne+Ng, T, nsteps, U0, utarget, use_bcarrier, Nfreq, om, kpar, tik0, H0, Hsym_ops, Hanti_ops, Hunc_ops, Ncoupled, Nunc, isSymm, Ident, wmat, 2, 0.0, saveConvHist, zeros(0), zeros(0), zeros(0), zeros(0), nNeumann, 0.0, 0.0, 0.0, false, zeros(0), quiet)
    end

    function parameters(Nosc::Int64, N::Int64, Nguard::Int64, Ne::Array{Int64,1}, Ng::Array{Int64,1}, T::Float64, nsteps::Int64,
                        U0::Array{Float64,2}, utarget::Array{Complex{Float64},2}, H0::SparseMatrixCSC{Float64, Int64},
                        Hsym_ops:: Array{SparseMatrixCSC{Float64, Int64},1}, Hanti_ops:: Array{SparseMatrixCSC{Float64, Int64},1}, wmat::Diagonal{Float64,Array{Float64,1}}, 
                        use_bcarrier::Bool, om::Array{Float64,2}, kpar::Int64, tik0::Float64, saveConvHist::Bool)
        Nfreq=size(om,2)
        Ntot = N+Nguard
        Ncoupled = length(Hsym_ops)
        Nanti = length(Hanti_ops)
        Nunc = 0
        Hunc_ops = spzeros(0)
        @assert(Ncoupled == Nanti)
        # Check for consistency in coupled controls 
        for i = 1:Ncoupled
            L = LinearAlgebra.tril(Hsym_ops[i] + Hanti_ops[i])
            if(norm(L) > eps(1.0))
                throw(ArgumentError("Coupled Hamiltonian inconsistent with currently implemented rotating wave approximation. Please flip sign on Hanti_ops.\n"))
            end
        end

        isSymm = []
        Ident = sparse(Matrix{Float64}(I, Ntot, Ntot)) # It needs to be sparse for the sparse time stepper to work
        # Default number of Neumann series terms
        nNeumann = 3
        quiet = false
        new(Nosc, N, Nguard, Ne, Ng, Ne+Ng, T, nsteps, U0, utarget, use_bcarrier, Nfreq, om, kpar, tik0, H0, Hsym_ops, Hanti_ops, Hunc_ops, Ncoupled, Nunc, isSymm, Ident, wmat, 2, 0.0, saveConvHist, zeros(0), zeros(0), zeros(0), zeros(0), nNeumann, 0.0, 0.0, 0.0, false, zeros(0), quiet)
    end


end

# This struct holds all of the working arrays needed to call traceobjgrad. Preallocated for efficiency
mutable struct Working_Arrays
    # Hamiltonian matrices
    K0  ::MyRealMatrix
    K05 ::MyRealMatrix
    K1  ::MyRealMatrix
    S0  ::MyRealMatrix
    S05 ::MyRealMatrix
    S1  ::MyRealMatrix

    # Forward/Adjoint variables+stages
    vtargetr    ::Array{Float64,2}
    vtargeti    ::Array{Float64,2}
    lambdar     ::Array{Float64,2}
    lambdar0    ::Array{Float64,2}
    lambdai     ::Array{Float64,2}
    lambdai0    ::Array{Float64,2}
    lambdar05   ::Array{Float64,2}
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

    function Working_Arrays(params::parameters,Ntot::Int64,N::Int64,nCoeff::Int64)
        println("Deprecation warning: this Working_Array constructor will be removed in future versions of the Juqbox package")
        println("\t Instead use Working_Arrays(params::parameters, nCoeff::Int64)")
        K0,S0,K05,S05,K1,S1,vtargetr,vtargeti = KS_alloc(params)
        lambdar,lambdar0,lambdai,lambdai0,lambdar05,κ₁,κ₂,ℓ₁,ℓ₂,rhs,gr0,gi0,gr1,gi1,hr0,hi0,hr1,vr,vi,vi05,vr0,vfinalr,vfinali = time_step_alloc(Ntot,N)
        if params.pFidType == 3
            gr, gi, gradobjfadj, tr_adj = grad_alloc(nCoeff-1)
        else
            gr, gi, gradobjfadj, tr_adj = grad_alloc(nCoeff)
        end            
        new(K0,S0,K05,S05,K1,S1,vtargetr,vtargeti,lambdar,lambdar0,lambdai,lambdai0,lambdar05,κ₁,κ₂,ℓ₁,ℓ₂,rhs,gr0,gi0,gr1,gi1,hr0,hi0,hr1,vr,vi,vi05,vr0,vfinalr,vfinali,gr, gi, gradobjfadj, tr_adj)
    end

    function Working_Arrays(params::parameters, nCoeff::Int64)
        N = params.N
        Ntot = N + params.Nguard
        K0,S0,K05,S05,K1,S1,vtargetr,vtargeti = KS_alloc(params)
        lambdar,lambdar0,lambdai,lambdai0,lambdar05,κ₁,κ₂,ℓ₁,ℓ₂,rhs,gr0,gi0,gr1,gi1,hr0,hi0,hr1,vr,vi,vi05,vr0,vfinalr,vfinali = time_step_alloc(Ntot,N)
        if params.pFidType == 3
            gr, gi, gradobjfadj, tr_adj = grad_alloc(nCoeff-1)
        else
            gr, gi, gradobjfadj, tr_adj = grad_alloc(nCoeff)
        end            
        new(K0,S0,K05,S05,K1,S1,vtargetr,vtargeti,lambdar,lambdar0,lambdai,lambdai0,lambdar05,κ₁,κ₂,ℓ₁,ℓ₂,rhs,gr0,gi0,gr1,gi1,hr0,hi0,hr1,vr,vi,vi05,vr0,vfinalr,vfinali,gr, gi, gradobjfadj, tr_adj)
    end
    
end

function traceobjgrad(pcof0::Array{Float64,1},  params::parameters, wa::Working_Arrays, verbose::Bool = false, evaladjoint::Bool = true)
#    @assert(params.Nosc >= 1 && params.Nosc <=2) # Currently the only implemented cases
    order  = 2
    N      = params.N    
    Nguard = params.Nguard  
    T      = params.T

    utarget = params.utarget

    nsteps = params.nsteps
    tik0   = params.tik0

    H0 = params.H0
    Ng = params.Ng
    Ne = params.Ne
    
    Nt   = params.Nt # vector
    Ntot = N + Nguard # scalar

    Ncoupled  = params.Ncoupled # Number of symmetric control Hamiltonians. We currently assume that the number of anti-symmetric Hamiltonians is the same
    Nunc  = params.Nunc # Number of uncoupled control functions.
    Nosc  = params.Nosc
    Nfreq = params.Nfreq
    Nsig  = 2*Ncoupled + Nunc

    nNeumann = params.nNeumann

    # Reference pre-allocated working arrays
    K0 = wa.K0
    S0 = wa.S0
    K05 = wa.K05
    S05 = wa.S05
    K1 = wa.K1
    S1 = wa.S1
    vtargetr = wa.vtargetr
    vtargeti = wa.vtargeti
    lambdar = wa.lambdar
    lambdar0 = wa.lambdar0
    lambdai = wa.lambdai
    lambdai0 = wa.lambdai0
    lambdar05 = wa.lambdar05
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
    hr1 = wa.hr1
    vr = wa.vr
    vi = wa.vi
    vi05 = wa.vi05
    vr0 = wa.vr0
    vfinalr = wa.vfinalr
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
    if Psize%Nsig != 0 || Psize < 3*Nsig
        error("pcof must have an even number of elements >= ",3*Nsig,", not ", Psize)
    end
    if params.use_bcarrier
        D1 = div(Psize, Nsig*Nfreq)  # 
        Psize = D1*Nsig*Nfreq # active part of the parameter array
    else
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
    wmat = params.wmat
    
    # Here we can choose what kind of control function expansion we want to use
    if (params.use_bcarrier)
        # FMG FIX
        splinepar = bcparams(T, D1, Ncoupled, Nunc, params.om, pcof)
    else
    # the old bsplines is the same as the bcarrier with om = 0
        splinepar = splineparams(T, D1, Nsig, pcof)   # parameters for B-splines
    end

    # it is up to the user to estimate the number of time steps
    dt ::Float64 = T/nsteps

    gamma, stages = getgamma(order)

    if verbose
        println("Final time: ", T, ", number of time steps: " , nsteps , ", time step: " , dt )
    end
    
    #real and imaginary part of initial condition
    copy!(vr,params.U0)
    vi   .= 0.0
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

        forbidden0 = tinv*penalf2aTrap(vr, wmat)
        # Störmer-Verlet
        for q in 1:stages
            if evaladjoint && verbose
                t0  = t
                copy!(vr0,vr)
            end
            
            # Update K and S matrices
            # general case
            KS!(K0, S0, t, params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0) 
            KS!(K05, S05, t + 0.5*dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0) 
            KS!(K1, S1, t + dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0) 
            
            # Take a step forward and accumulate weight matrix integral. Note the √2 multiplier is to account
            # for the midpoint rule in the numerical integration of the imaginary part of the signal.
            @inbounds t = step!(t, nNeumann, vr, vi, vi05, dt*gamma[q], K0, S0, K05, S05, K1, S1, Ident, κ₁, κ₂, ℓ₁, ℓ₂, rhs)

            forbidden = tinv*penalf2a(vr, vi05, wmat)  
            objfv = objfv + gamma[q]*dt*0.5*(forbidden0 + forbidden)

            # Keep prior value for next step (FG: will this work for multiple stages?)
            forbidden0 = forbidden

            # compute component of the gradient for verification of adjoint method
            if evaladjoint && verbose
                # compute the forcing for (wr, wi)
                fgradforce!(params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc,
                            params.isSymm, vr0, vi05, vr, t-dt, dt, splinepar, kpar, gr0, gr1, gi0, gi1, gr, gi)

                copy!(wr1,wr)
                @inbounds step_fwdGrad!(t0, nNeumann, wr1, wi, wi05, dt*gamma[q],
                                                  gr0, gi0, gr1, gi1, K0, S0, K05, S05, K1, S1, Ident, κ₁, κ₂, ℓ₁, ℓ₂, rhs) 
                forbalpha0 = tinv*penalf2grad(vr0, vi05, wr, wi05, wmat)
                forbalpha1 = tinv*penalf2grad(vr, vi05, wr1, wi05, wmat)

                copy!(wr,wr1)
                objf_alpha1 = objf_alpha1 + gamma[q]*dt*0.5*2.0*(forbalpha0 + forbalpha1)

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
    salpha1 = tracefidcomplex(wr, -wi, vtargetr, vtargeti)
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

tikhonovpenalty = tikhonov_pen(pcof, params)

objfv = objfv .+ tikhonovpenalty

traceInfidelity = 1.0 - tracefidabs2(vfinalr, vfinali, vtargetr, vtargeti)

if evaladjoint

    if verbose
        dfdp = objf_alpha1
    end  

    if (params.use_bcarrier)
        gradSize = (2*Ncoupled+Nunc)*Nfreq*D1
    else
        gradSize = Nsig*D1
    end

    # initialize array for storing the adjoint gradient so it can be returned to the calling function/program
    totalgrad = zeros(gradSize, 1);
    gradobjfadj[:] .= 0.0
    
    t = T
    dt = -dt

    scomplex0 = tracefidcomplex(vr, -vi, vtargetr, vtargeti)

    if pFidType == 1
        scomplex0 = exp(1im*params.globalPhase) - scomplex0
    end

    # Set initial condition for adjoint variables
    init_adjoint!(pFidType, params.globalPhase, N, scomplex0, lambdar, lambdar0, lambdar05,lambdai, lambdai0,
                  vtargetr, vtargeti)

    #Backward time stepping loop
    for step in nsteps-1:-1:0

        # Forcing for the real part of the adjoint variables in first PRK "stage"
        penalf2adj!(vr,wmat,tinv,hr0)

        #loop over stages
        for q in 1:stages
            t0 = t
            copy!(vr0,vr)
            
            # update K and S
            # Since t is negative we have that K0 is K^{n+1}, K05 = K^{n-1/2}, 
            # K1 = K^{n} and similarly for S.
            # general case
            KS!(K0, S0, t, params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0) 
            KS!(K05, S05, t + 0.5*dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0) 
            KS!(K1, S1, t + dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0) 

            # Integrate state variables backwards in time one step
            @inbounds t = step!(t, nNeumann, vr, vi, vi05, dt*gamma[q], K0, S0, K05, S05, K1, S1, Ident, κ₁, κ₂, ℓ₁, ℓ₂, rhs)

            # Forcing for adjoint equations
            penalf2adj!(vi05, wmat,tinv, hi0)
            penalf2adj!(vr, wmat,tinv, hr1)

            # evolve lambdar, lambdai
            temp = t0
            @inbounds temp = step!(temp, nNeumann, lambdar, lambdai, lambdar05, dt*gamma[q], hr0, hi0, hr1, K0, S0, K05, S05, K1, S1, Ident, κ₁, κ₂, ℓ₁, ℓ₂, rhs)

            # Accumulate gradient
            adjoint_grad_calc!(params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, vr0, vi05, vr, lambdar0, lambdar05, lambdai, lambdai0, t0, dt,splinepar, gr, gi, tr_adj) 
            axpy!(gamma[q]*dt,tr_adj,gradobjfadj)
            
            # save for next stage
            copy!(lambdai0,lambdai)
            copy!(lambdar0,lambdar)
            copy!(hr0,hr1)

        end #for stages
    end # for step (backward time stepping loop)

    # Add in Tikhonov regularization gradient term
    tikhonov_grad!(pcof, params, gr)  
    axpy!(1.0,gr,gradobjfadj)

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
    
end # if evaladjoint

if verbose
    println("Total objective func: ", objfv)
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
    println("Unitary test:")
    println("Col   (1 - Vnrm)")
    Vnrm ::Float64 = 0.0
    for q in 1:N
        Vnrm = usaver[:,q,nlast]' * usaver[:,q,nlast] + usavei[:,q,nlast]' * usavei[:,q,nlast]
        Vnrm = sqrt(Vnrm)
        println("  ", q, " |     ", 1.0 - Vnrm)
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
        maxpop = zeros(N)
        for q in 1:N
#            normlastguard[q] = sqrt( (usaver[Ntot,q,:]' * usaver[Ntot,q,:] + usavei[Ntot,q,:]' * usavei[Ntot,q,:])/nsteps );
            maxpop[q] = maximum( abs.(usaver[Ntot,q,:]).^2 + abs.(usavei[Ntot,q,:]).^2 )
        end
#        println("L2 norm last guard level (max) = ", maximum(normlastguard))
        println("Max population of last guard level = ", maximum(maxpop))
    else
        println("No guard levels in this simulation");
    end
    
end #if verbose

# return to calling routine
if verbose && evaladjoint
    return objfv, totalgrad, usaver+1im*usavei, mfidelityrot, dfdp
elseif verbose
    println("Returning from traceobjgrad with objfv, unitary history, fidelity")
    return objfv, usaver+1im*usavei, mfidelityrot
elseif evaladjoint
    return objfv, totalgrad, primaryobjf, secondaryobjf, traceInfidelity
else
    return objfv
end #if
end

function setup_prior!(params:: parameters, priorFile::String)

    # read a prior parameter vector from a JLD2 file, assume that the number of parameters is
    # compatible between the current and prior parameter vector
    
    prior_pcof = load(priorFile, "pcof")
    println("Length of prior_pcof = ", length(prior_pcof) )

    params.priorCoeffs = prior_pcof
    params.usingPriorCoeffs = true
end

function wmatsetup(Ne::Array{Int64,1}, Ng::Array{Int64,1})
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

                        # additional weighting (ad hoc)
                        # if i1 == Nt1 && i2<=Ne2 
                        #   forbFact=100
                        # end
                        # if i2 == Nt2 && i1<=Ne1 
                        #   forbFact=100
                        # end

                        w[q] = forbFact*maximum(temp)
          
                    end # if guard level
                end # for i1
            end # for i2

            # normalize by the number of entries with w=1
            coeff = 10.0/nForb # was 1/nForb
        elseif Ndim == 3
            fact = 1e-3 # for more emphasis on the "forbidden" states. Old value: 0.1
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
                            # additional weighting (ad hoc)
                            # if i1 == Nt[1] && i2<=Ne[2] && i3<=Ne3
                            #   forbFact=100
                            # end
                            # if i2 == Nt[2] && i1<=Ne[1] && i3<=Ne3
                            #   forbFact=100
                            # end
                            if i3 == Nt[3] && i1<=Ne[1] && i2<=Ne[2]
                                forbFact=100
                            end

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

        # println("wmatsetup: Number of forbidden states = ", nForb, " scaling coeff = ", coeff)
    end # if sum(Ng) > 0
    wmat = coeff * Diagonal(w) # turn vector into diagonal matrix
    return wmat
end

# Matrices for the Hamiltonian in rotation frame
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
function assign_thresholds_old(maxpar, Ncoupled, Nfreq, D1)
    nCoeff = 2*Ncoupled*Nfreq*D1
    minCoeff = zeros(nCoeff) # Initialize storage
    maxCoeff = zeros(nCoeff)

#    @printf("Ncoupled = %d, Nfreq = %d, D1 = %d, nCoeff = %d\n", Ncoupled, Nfreq, D1, nCoeff)
    for c in 1:Ncoupled
        for f in 1:Nfreq
            offset1 = 2*(c-1)*Nfreq*D1 + (f-1)*2*D1
            minCoeff[ offset1 + 1:offset1+2*D1] .= -maxpar[c] # same for p(t) and q(t)
            maxCoeff[offset1 + 1:offset1+2*D1] .= maxpar[c]
        end
    end
    return minCoeff, maxCoeff
end

#------------------------------------------------------------
function assign_thresholds_freq(maxamp, Ncoupled, Nfreq, D1)
    nCoeff = 2*Ncoupled*Nfreq*D1
    minCoeff = zeros(nCoeff) # Initialize storage
    maxCoeff = zeros(nCoeff)

#    @printf("Ncoupled = %d, Nfreq = %d, D1 = %d, nCoeff = %d\n", Ncoupled, Nfreq, D1, nCoeff)
    for c in 1:Ncoupled
        for f in 1:Nfreq
            offset1 = 2*(c-1)*Nfreq*D1 + (f-1)*2*D1
            minCoeff[ offset1 + 1:offset1+2*D1] .= -maxamp[f] # same for p(t) and q(t)
            maxCoeff[offset1 + 1:offset1+2*D1] .= maxamp[f]
        end
    end
    return minCoeff, maxCoeff
end


# FMG: Account for both coupled/uncoupled controls
function assign_thresholds(params,D1,maxpar,maxpar_unc)
    Nfreq = params.Nfreq
    nCoeff =  (2*params.Ncoupled + params.Nunc)*Nfreq*D1
    minCoeff = zeros(nCoeff) # Initialize storage
    maxCoeff = zeros(nCoeff)

    for c in 1:params.Ncoupled
        for f in 1:Nfreq
            offset1 = 2*(c-1)*Nfreq*D1 + (f-1)*2*D1
            minCoeff[ offset1 + 1:offset1+2*D1] .= -maxpar[c] # same for p(t) and q(t)
            maxCoeff[offset1 + 1:offset1+2*D1] .= maxpar[c]
        end
    end

    offset0 = 2*params.Ncoupled*params.Nfreq*D1
    for c in 1:params.Nunc
        for f in 1:Nfreq
            offset1 = (c-1)*Nfreq*D1 + (f-1)*D1
            minCoeff[offset0+offset1+1:offset0+offset1+D1] .= -maxpar_unc[c]
            maxCoeff[offset0+offset1+1:offset0+offset1+D1] .= maxpar_unc[c]
        end
    end

    return minCoeff, maxCoeff
end

#------------------------------------------------------------

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


@inline function tikhonov_pen(pcof::Array{Float64,1}, params ::parameters)
    Npar = size(pcof,1)
    iNpar = 1.0/Npar
    Nseg = (2*params.Ncoupled+params.Nunc)*params.Nfreq
    D1 = div(Npar,Nseg)

    # Tikhonov regularization
    if params.usingPriorCoeffs
        penalty0 = dot(pcof-params.priorCoeffs, pcof-params.priorCoeffs)
    else
        penalty0 = dot(pcof,pcof)
    end

    penalty1 = 0.0
    # This makes little sense when using smooth B-splines
    # for i = 1:Nseg
    #     offset = (i-1)*D1
    #     @fastmath @inbounds @simd for j = offset+2:offset+D1
    #         penalty1 += (pcof[j] - pcof[j-1])^2
    #     end
    # end

#    penalty = (params.tik0 * penalty0 + params.tik1 * penalty1) * iNpar;

    penalty = (params.tik0 * penalty0) * iNpar;
                                
    return penalty
end

@inline function tikhonov_grad!(pcof::Array{Float64,1}, params:: parameters, pengrad::Array{Float64,1})  
    Npar = size(pcof,1)
    iNpar = 1.0/Npar
    Nseg = (2*params.Ncoupled+params.Nunc)*params.Nfreq
    D1 = div(Npar,Nseg)

    # Tikhonov regularization
    pengrad[:] .= 0.0
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
             Hunc_ops::Array{MyRealMatrix, 1}, Nunc::Int64, isSymm::BitArray{1}, splinepar::BsplineParams, H0::Array{Float64,N}) where N

    # NEW ordering:
    # rfeval = controlfunc(t,splinepar, 0)
    # ifeval = controlfunc(t,splinepar, 1)
    # rgeval = controlfunc(t,splinepar, 2)
    # igeval = controlfunc(t,splinepar, 3)
    
    # Ncoupled = length(Hsym_ops) # Nq equals the number of oscillators
    Ncoupled = splinepar.Ncoupled

    # K .= H0
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

    offset = 2*Ncoupled-1
    for q=1:Nunc
        ft = controlfunc(t,splinepar, offset+q)
        if(isSymm[q])
            axpy!(ft,Hunc_ops[q],K)
        else
            axpy!(ft,Hunc_ops[q],S)
        end
    end
    
end

# Sparse version
@inline function KS!(K::SparseMatrixCSC{Float64,Int64}, S::SparseMatrixCSC{Float64,Int64}, t::Float64, Hsym_ops::Array{MyRealMatrix,1}, Hanti_ops::Array{MyRealMatrix, 1},
             Hunc_ops::Array{MyRealMatrix, 1}, Nunc::Int64, isSymm::BitArray{1}, splinepar::BsplineParams, H0::SparseMatrixCSC{Float64,Int64})

    # NEW ordering:
    # rfeval = controlfunc(t,splinepar, 0)
    # ifeval = controlfunc(t,splinepar, 1)
    # rgeval = controlfunc(t,splinepar, 2)
    # igeval = controlfunc(t,splinepar, 3)
    
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

    offset = 2*Ncoupled-1
    for q=1:Nunc
        ft = controlfunc(t,splinepar, offset+q)
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
# 0 - p1(t) for x-drive # Hsym_ops[1]
# 1 - q1(t) for y-drive # Hanti_ops[1]
# 2 - p2(t) for x-drive # Hsym_ops[1]
# 3 - q2(t) for y-drive # Hanti_ops[1]
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
function eval_forward(U0::Array{Float64,2}, pcof0::Array{Float64,1}, params::parameters, saveAll:: Bool = false, verbose::Bool = false)  
    N = params.N  
    Q = 1 #one initial data, specified in U0[:,1] (currently assumed to be real)

    order = 2
    Nguard = params.Nguard  
    T = params.T
    nsteps = params.nsteps
    nNeumann = params.nNeumann
    H0 = params.H0

    Ntot = N + Nguard
    pcof = pcof0

    # We have 2*Ncoupled ctrl functions
    Ncoupled = params.Ncoupled
    Nunc = params.Nunc
    Nfreq = params.Nfreq
    Nsig = 2*Ncoupled + Nunc

    Psize = size(pcof,1) #must provide separate coefficients for the real and imaginary parts of the control fcn
    if Psize%2 != 0 || Psize < 6
        error("pcof must have an even number of elements >= 6, not ", Psize)
    end
    if params.use_bcarrier
        D1 = div(Psize, 2*Ncoupled*Nfreq) 
        Psize = 2*D1*Ncoupled*Nfreq #
    else
        D1 = div(Psize, 2*Ncoupled)
        Psize = 2*D1*Ncoupled # active part of the parameter array
    end
    
    tinv ::Float64 = 1.0/T
    
    if verbose
        println("Vector dim Ntot =", Ntot , ", Guard levels Nguard = ", Nguard , ", Param dim, Psize = ", Psize, ", Spline coeffs per func, D1= ", D1, ", Nsteps = ", nsteps)
    end
    
    zeromat = zeros(Float64,Ntot,N) 

    # Here we can choose what kind of control function expansion we want to use
    if (params.use_bcarrier)
        splinepar = bcparams(T, D1, Ncoupled, Nunc, params.om, pcof)
    else
        splinepar = splinepar(T, D1, Nsig, pcof)   # parameters for B-splines
    end

    # it is up to the user to estimate the number of time steps
    dt ::Float64 = T/nsteps

    gamma, stages = getgamma(order)

    if verbose
        println("Final time: ", T, ", number of time steps: " , nsteps , ", time step: " , dt )
    end

    # the basis for the initial data as a matrix
    Ident = params.Ident

    # Note: Initial condition is supplied as an argument

    #real and imaginary part of initial condition
    vr   = U0[:,Q:Q]
    vi   = zeros(Float64,Ntot,Q)
    vi05 = zeros(Float64,Ntot,Q)

    usaver = zeros(Float64,Ntot,Q,nsteps+1)
    usavei = zeros(Float64,Ntot,Q,nsteps+1)
    usaver[:,:,1] = vr # the rotation to the lab frame is the identity at t=0
    usavei[:,:,1] = -vi

    # Preallocate WHAT ABOUT SPARSE FORMAT!
    K0   = zeros(Float64,Ntot,Ntot)
    S0   = zeros(Float64,Ntot,Ntot)
    K05  = zeros(Float64,Ntot,Ntot)
    S05  = zeros(Float64,Ntot,Ntot)
    K1   = zeros(Float64,Ntot,Ntot)
    S1   = zeros(Float64,Ntot,Ntot)
    κ₁   = zeros(Float64,Ntot,Q)
    κ₂   = zeros(Float64,Ntot,Q)
    ℓ₁   = zeros(Float64,Ntot,Q)
    ℓ₂   = zeros(Float64,Ntot,Q)
    rhs   = zeros(Float64,Ntot,Q)

    #initialize variables for time stepping
    t       ::Float64 = 0.0
    step    :: Int64 = 0

    KS!(K0, S0, t, params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0) 
    # Forward time stepping loop
    for step in 1:nsteps

        # Störmer-Verlet
        for q in 1:stages
            
            # Update K and S matrices
            KS!(K0, S0, t, params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0)
            KS!(K05, S05, t + 0.5*dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0)
            KS!(K1, S1, t + dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0)

            # Take a step forward and accumulate weight matrix integral. Note the √2 multiplier is to account
            # for the midpoint rule in the numerical integration of the imaginary part of the signal.
            # @inbounds t, vr, vi, vi05 = step(t, vr, vi, dt*gamma[q], K0, S0, K05, S05, K1, S1, Ident)
            @inbounds t = step!(t, nNeumann, vr, vi, vi05, dt*gamma[q], K0, S0, K05, S05, K1, S1, Ident, κ₁, κ₂, ℓ₁, ℓ₂, rhs)

            # Keep prior value for next step (FG: will this work for multiple stages?)

        end # Stromer-Verlet
        
        # rotated frame
        usaver[:,:, step + 1] = vr
        usavei[:,:, step + 1] = -vi

    end #forward time stepping loop

    if verbose
        nlast = 1 + nsteps
        println("Unitary test:")
        println(" Column   1 - Vnrm")
        Vnrm ::Float64 = 0.0
        for q in 1:Q
            Vnrm = usaver[:,q,nlast]' * usaver[:,q,nlast] + usavei[:,q,nlast]' * usavei[:,q,nlast]
            Vnrm = sqrt(Vnrm)
            println(q, " | ", 1.0 - Vnrm)
        end
    end #if verbose

    # return to calling routine

    if saveAll
        return usaver + im*usavei
    else
        return usaver[:,:,end] + im*usavei[:,:,end]
    end

end

# Estimate the number of terms used in the Neumann series linear solve during timestepping. 
# FMG: This will work but appears to be pessimistic. One can use fewer terms, perhaps a 
# better estimate can be found.
# old version
function estimate_Neumann(tol::Float64, T::Float64, nsteps::Int64, maxpar::Array{Float64,1}, Hanti_ops::Array{Array{Float64,N},1}) where N
    k = Float64(T/nsteps)
    S = 0.5*k*maxpar[1]*Hanti_ops[1]
    for j = 2:length(Hanti_ops)
        axpy!(0.5*k*maxpar[j],Hanti_ops[j],S)
    end
    normS = opnorm(S)
    nterms = ceil(Int64,log(tol)/log(normS))-1
    return nterms
end

# old version
# Sparse version of above
function estimate_Neumann(tol::Float64, T::Float64, nsteps::Int64, maxpar::Array{Float64,1}, Hanti_ops::Array{SparseMatrixCSC{Float64,Int64},1})
    k = Float64(T/nsteps)
    S = 0.5*k*maxpar[1]*Hanti_ops[1]
    for j = 2:length(Hanti_ops)
        axpy!(0.5*k*maxpar[j],Hanti_ops[j],S)
    end
    S = Array(S) #In case input is sparse
    normS = opnorm(S)
    nterms = ceil(Int64,log(tol)/log(normS))-1
    return nterms
end

# new version
function estimate_Neumann!(tol::Float64, T::Float64, params::parameters, maxpar::Array{Float64,1})
    nsteps = params.nsteps
    k = Float64(T/nsteps)

    if(params.Ncoupled > 0 && params.Nunc == 0)
        # If only coupled Hamiltonian terms are present
        S = 0.5*k*maxpar[1]*params.Hanti_ops[1]
        for j = 2:length(params.Hanti_ops)
            axpy!(0.5*k*maxpar[j],params.Hanti_ops[j],S)
        end

        # If in sparse mode, cast to full matrix for norm estimation
        if(typeof(S) ==  SparseMatrixCSC{Float64, Int64})
            S = Array(S)
        end
    else 
        # If only uncoupled Hamiltonian terms are present
        S = zeros(size(params.Hunc_ops[1]))
        for j = 1:params.Nunc
            if(!params.isSymm[j])
                axpy!(0.5*k*maxpar[j],params.Hunc_ops[j],S)     
            end
        end
    end
    normS = opnorm(S)
    nterms = ceil(Int64,log(tol)/log(normS))-1
    if(nterms > 0)
        params.nNeumann = nterms
    end
    # return nterms
end


# Estimate the number of terms used in the Neumann series linear solve during timestepping. 
# Both coupled and uncoupled terms present.
function estimate_Neumann!(tol::Float64, T::Float64, params::parameters, maxpar::Array{Float64,1}, maxunc::Array{Float64,1})
    nsteps = params.nsteps
    k = Float64(T/nsteps)
    if(params.Ncoupled > 0)
        S = 0.5*k*maxpar[1]*params.Hanti_ops[1]
        for j = 2:length(params.Hanti_ops)
            axpy!(0.5*k*maxpar[j],params.Hanti_ops[j],S)
        end
    end

    if(params.Nunc > 0)
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
        params.nNeumann = nterms
    end
    # return nterms
end

# Function to estimate the number of time steps needed for the simulation. Coupled controls only.
function calculate_timestep(T::Float64, D1::Int64, H0::AbstractArray,Hsym_ops::AbstractArray,Hanti_ops::AbstractArray, maxpar::Array{Float64,1})
    K1 = copy(H0) 
    Ncoupled = length(Hsym_ops)

    # Coupled control Hamiltonians
    for i = 1:Ncoupled
        K1 = K1 + maxpar.*Hsym_ops[i] + 1im*maxpar.*Hanti_ops[i]
    end

    # Estimate time step
    lamb = eigvals(Array(K1))
    maxeig = maximum(abs.(lamb)) 
    mineig = minimum(abs.(lamb)) 
    Pmin = 40
    samplerate1 = maxeig*Pmin/(2*pi)
    nsteps = ceil(Int64, T*samplerate1)

    # The above estimate does not account for quickly varying signals or a large number of splines. 
    # Double check at least 2-3 points per spline to resolve control function.
    nsteps_pps = 3*(D1-2)
    nsteps = max(nsteps_pps,nsteps)

    return maxeig,nsteps
end

# Function to estimate the number of time steps needed for the simulation. Includes uncoupled controls.
function calculate_timestep(T::Float64, D1::Int64, H0::AbstractArray,Hsym_ops::AbstractArray,Hanti_ops::AbstractArray,
                            Hunc_ops::AbstractArray,maxpar::Array{Float64,1},max_flux::Array{Float64,1})
    K1 = copy(H0) 
    Ncoupled = length(Hsym_ops)
    Nunc = length(Hunc_ops)

    # Coupled control Hamiltonians
    for i = 1:Ncoupled
        K1 = K1 .+ maxpar[i].*Hsym_ops[i] .+ 1im*maxpar[i].*Hanti_ops[i]
    end

    # Typecasting issue for sparse arrays
    if(typeof(H0) == SparseMatrixCSC{Float64,Int64})
        K1 = SparseMatrixCSC{ComplexF64,Int64}(K1)
    end

    # Uncoupled control Hamiltonians
    for i = 1:Nunc
        if(issymmetric(Hunc_ops[i]))
            K1 = K1 .+ max_flux[i]*Hunc_ops[i]
        elseif(norm(Hunc_ops[i]+Hunc_ops[i]') < 1e-14)
            K1 .+= 1im*max_flux[i].*Hunc_ops[i]
        else 
            throw(ArgumentError("Uncoupled Hamiltonians for more than a single oscillator not currently supported.\n"))
        end
    end

    # Estimate time step
    lamb = eigvals(Array(K1))
    maxeig = maximum(abs.(lamb)) 
    mineig = minimum(abs.(lamb)) 
    Pmin = 40
    samplerate1 = maxeig*Pmin/(2*pi)
    nsteps = ceil(Int64, T*samplerate1)

    # The above estimate does not account for quickly varying signals or a large number of splines. 
    # Double check at least 2-3 points per spline to resolve control function.
    nsteps_pps = 3*(D1-2)
    nsteps = max(nsteps_pps,nsteps)

    return maxeig,nsteps
end

# Function to estimate the number of time steps needed for the simulation. Only uncoupled controls.
function calculate_timestep(T::Float64, D1::Int64, H0::AbstractArray, Hunc_ops::AbstractArray,max_unc::Array{Float64,1})
    K1 = copy(H0) 
    Nunc = length(Hunc_ops)

    # Typecasting issue for sparse arrays
    if(typeof(H0) == SparseMatrixCSC{Float64,Int64})
        K1 = SparseMatrixCSC{ComplexF64,Int64}(K1)
    end

    # Uncoupled control Hamiltonians
    for i = 1:Nunc
        if(issymmetric(Hunc_ops[i]))
            K1 = K1 .+ max_unc[i]*Hunc_ops[i]
        elseif(norm(Hunc_ops[i]+Hunc_ops[i]') < 1e-14)
            K1 .+= 1im*max_unc[i].*Hunc_ops[i]
        else 
            throw(ArgumentError("Uncoupled Hamiltonians for more than a single oscillator not currently supported.\n"))
        end
    end

    # Estimate time step
    lamb = eigvals(Array(K1))
    maxeig = maximum(abs.(lamb)) 
    mineig = minimum(abs.(lamb)) 
    Pmin = 40
    samplerate1 = maxeig*Pmin/(2*pi)
    nsteps = ceil(Int64, T*samplerate1)

    # The above estimate does not account for quickly varying signals or a large number of splines. 
    # Double check at least 2-3 points per spline to resolve control function.
    nsteps_pps = 3*(D1-2)
    nsteps = max(nsteps_pps,nsteps)

    return maxeig,nsteps
end

# Preallocate K and S matrices
function KS_alloc(params)
    Ntot = prod(params.Nt)
    # establish the non-zero pattern for sparse storage
    if typeof(params.H0) == SparseMatrixCSC{Float64, Int64}
        K0 = copy(params.H0)
        S0 = spzeros(size(params.H0,1),size(params.H0,2))
        for q=1:params.Ncoupled
            K0 += params.Hsym_ops[q]
            S0 += params.Hanti_ops[q]
        end
        for q=1:params.Nunc
            if(params.isSymm[q])
                K0 = K0 + params.Hunc_ops[q]
            else
                S0 = S0 + params.Hunc_ops[q]
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
    vtargetr = real(params.utarget)
    vtargeti = imag(params.utarget)
    return K0,S0,K05,S05,K1,S1,vtargetr,vtargeti
end

# Working arrays for timestepping
function time_step_alloc(Ntot::Int64,N::Int64)
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
    hr1       = zeros(Float64,Ntot,N)
    vr        = zeros(Float64,Ntot,N)
    vi        = zeros(Float64,Ntot,N)
    vi05      = zeros(Float64,Ntot,N)
    vr0       = zeros(Float64,Ntot,N)
    vfinalr   = zeros(Float64,Ntot,N) 
    vfinali   = zeros(Float64,Ntot,N)
    return lambdar,lambdar0,lambdai,lambdai0,lambdar05,κ₁,κ₂,ℓ₁,
           ℓ₂,rhs,gr0,gi0,gr1,gi1,hr0,hi0,hr1,vr,vi,vi05,vr0,vfinalr,vfinali
end

function grad_alloc(Nparams::Int64)
    gr = zeros(Nparams)
    gi = zeros(Nparams)
    gradobjfadj = zeros(Nparams)
    tr_adj = zeros(Nparams)
    return gr, gi, gradobjfadj, tr_adj
end


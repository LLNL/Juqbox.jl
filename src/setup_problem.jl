### Set up the Hamiltonians using the standard model 

using LinearAlgebra

function hamiltonians_one_sys(;Ness::Vector{Int64}, Nguard::Vector{Int64}, freq01::Float64, anharm::Float64, rotfreq::Float64, verbose::Bool = true)
    @assert(length(Ness)==1)
    @assert(length(Nguard)==1)
    @assert(minimum(Ness) >= 2)
    @assert(minimum(Nguard) >= 0)

    Ntot = Ness[1] + Nguard[1] # Total number of energy levels

    # frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies 
    # in the Hamiltonian matrix)
    fa = freq01
    xa = anharm

    # rotating frame transformation
    rot_freq = [rotfreq] 

    # setup drift Hamiltonian
    number = Diagonal(collect(0:Ntot-1))
    # Note: xa is negative
    H0  = 2*pi * ( (fa - rot_freq[1])*number + 0.5*xa* (number*number - number) ) 
    H0 = Array(H0)

    # Set up the control drive operators
    amat = Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U) # standard lowering matrix
    adag = Array(transpose(amat));
    Hsym_ops=[Array(amat + adag)]  # operator for p(t)
    Hanti_ops=[Array(amat - adag)] # operator for q(t)

    if verbose
        println("*** Single quantum system setup ***")
        println("System Hamiltonian coefficients [GHz]: f01 = ", fa, " anharmonicity = ", xa)
        println("Number of essential states = ", Ness, " Number of guard states = ", Nguard)
        println("Hamiltonians are of size ", Ntot, " by ", Ntot)
    end
    return H0, Hsym_ops, Hanti_ops, rot_freq
end


function hamiltonians_two_sys(;Ness::Vector{Int64}, Nguard::Vector{Int64}, freq01::Vector{Float64}, anharm::Vector{Float64}, f_rot::Float64,couple_coeff::Float64, couple_type::Int64, msb_order::Bool = true, verbose::Bool = true)
    @assert(length(Ness) == 2)
    @assert(length(Nguard) == 2)
    @assert(couple_type == 1 || couple_type == 2)

    Nt = Ness + Nguard
    N = prod(Ness); # Total number of nonpenalized energy levels
    Ntot = prod(Nt)
    #Nguard = Ntot - N # total number of guard states

    Nt1 = Ness[1] + Nguard[1]
    Nt2 = Ness[2] + Nguard[2]

    # frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies 
    # in the Hamiltonian matrix)
    fa = freq01[1]
    fb = freq01[2]
    x1 = anharm[1]
    x2 = anharm[2]

    # rotational frequencies
    favg = f_rot
    rot_freq = [favg, favg] 

    # detuning
    da = fa - favg
    db = fb - favg

    a1 = Array(Bidiagonal(zeros(Nt1),sqrt.(collect(1:Nt1-1)),:U))
    a2 = Array(Bidiagonal(zeros(Nt2),sqrt.(collect(1:Nt2-1)),:U))

    I1 = Array{Float64, 2}(I, Nt1, Nt1)
    I2 = Array{Float64, 2}(I, Nt2, Nt2)

    # number ops
    num1 = Diagonal(collect(0:Nt1-1))
    num2 = Diagonal(collect(0:Nt2-1))

    if msb_order
        # MSB ordering: Let the elements in the state vector 
        # psi = a_{ji} (e_j kron e_i), for j in [1,Nt2] and i in [1,Nt1]
        # We order the elements in the vector psi such that i varies the fastest 
        # The matrix (I kron a1) acts on alpha in psi = (beta kron alpha)
        # The matrix (a2 kron I) acts on beta in psi = (beta kron alpha)

        # create the a, a^\dag, b and b^\dag vectors
        amat = Array(kron(I2, a1))
        bmat = Array(kron(a2, I1))

        adag = Array(transpose(amat))
        bdag = Array(transpose(bmat))


        # number operators
        N1 = Diagonal(kron(I2, num1) )
        N2 = Diagonal(kron(num2, I1) )

    else
        # LSB ordering: qubit 1 varies the slowest and qubit 2 varies the fastest in the state vector
        amat = Array(kron(a1, I2))
        bmat = Array(kron(I1, a2))

        adag = Array(transpose(amat))
        bdag = Array(transpose(bmat))

        # number operators
        N1 = Diagonal(kron(num1, I2) )
        N2 = Diagonal(kron(I1, num2) )
    end

    # Coupling Hamiltonian: couple_type = 2 # 1: cross-Kerr, 2: Jaynes-Cummings
    if couple_type == 1
        Hcouple = couple_coeff*(N1*N2)
    elseif couple_type == 2
        Hcouple = couple_coeff*(bdag * amat + bmat * adag)
    end
            # System Hamiltonian
    H0 = 2*pi*(  da*N1 + 0.5*x1*(N1*N1-N1) + db*N2 + 0.5*x2*(N2*N2-N2) + Hcouple )
    H0 = Array(H0)

    # set up control hamiltonians
    Hsym_ops =[ amat+adag, bmat+bdag ]
    Hanti_ops=[ amat-adag, bmat-bdag ]

    if verbose
        println("*** Two coupled quantum systems setup ***")
        println("System Hamiltonian frequencies [GHz]: f01 = ", freq01, " rot. freq = ", f_rot)
        println("Anharmonicity = ", anharm, " coupling coeff = ", couple_coeff, " coupling type = ", (couple_type==1) ? "X-Kerr" : "J-C" )
        println("Number of essential states = ", Ness, " Number of guard states = ", Nguard)
        println("Hamiltonians are of size ", Ntot, " by ", Ntot)
    end
    return H0, Hsym_ops, Hanti_ops, rot_freq
end

function hamiltonians_three_sys(;Ness::Vector{Int64}, Nguard::Vector{Int64}, freq01::Vector{Float64}, anharm::Vector{Float64}, f_rot::Float64, couple_coeff::Vector{Float64}, couple_type::Int64, msb_order::Bool = true, verbose::Bool = true)
    @assert(length(Ness) == 3)
    @assert(length(Nguard) == 3)
    @assert(couple_type == 1 || couple_type == 2)

    Nt = Ness + Nguard
    N = prod(Ness); # Total number of nonpenalized energy levels
    Ntot = prod(Nt)
    #Nguard = Ntot - N # total number of guard states

    Nt1 = Nt[1]
    Nt2 = Nt[2]
    Nt3 = Nt[3]

    # frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies 
    # in the Hamiltonian matrix)
    fa = freq01[1]
    fb = freq01[2]
    fc = freq01[3]

    xa = anharm[1]
    xb = anharm[2]
    xc = anharm[3]

    xab = couple_coeff[1]
    xac = couple_coeff[2]
    xbc = couple_coeff[3]

    # rotational frequencies
    favg = f_rot
    rot_freq = [favg, favg, favg] 

    # detuning
    da = fa - favg
    db = fb - favg
    dc = fc - favg

    # single system lowering ops
    a1 = Array(Bidiagonal(zeros(Nt1),sqrt.(collect(1:Nt[1]-1)),:U))
    a2 = Array(Bidiagonal(zeros(Nt2),sqrt.(collect(1:Nt[2]-1)),:U))
    a3 = Array(Bidiagonal(zeros(Nt2),sqrt.(collect(1:Nt[3]-1)),:U))

    I1 = Array{Float64, 2}(I, Nt[1], Nt[1])
    I2 = Array{Float64, 2}(I, Nt[2], Nt[2])
    I3 = Array{Float64, 2}(I, Nt[3], Nt[3])

    # single system number ops
    num1 = Diagonal(collect(0:Nt[1]-1))
    num2 = Diagonal(collect(0:Nt[2]-1))
    num3 = Diagonal(collect(0:Nt[3]-1))

    if msb_order
        # MSB ordering: Let the elements in the state vector be
        # |psi> = sum a_{kji} (|k> kron |j> kron |i>, 
        # for i in [1,Nt1], j in [1,Nt2], , k in [1,Nt3]
        # We order the elements in the vector psi such that i varies the fastest 
        # The matrix amat = I kron I kron a1 acts on alpha in psi = gamma kron beta kron alpha
        # The matrix bmat = I kron a2 kron I acts on beta in psi = gamma kron beta kron alpha
        # The matrix cmat = a3 kron I2 kron I1 acts on gamma in psi = gamma kron beta kron alpha

        # create the combined lowering and raising ops
        amat = kron(I3, kron(I2, a1))
        bmat = kron(I3, kron(a2, I1))
        cmat = kron(a3, kron(I2, I1))

        adag = Array(transpose(amat))
        bdag = Array(transpose(bmat))
        cdag = Array(transpose(cmat))

        # number operators
        Na = Diagonal(kron(I3, kron(I2, num1)) )
        Nb = Diagonal(kron(I3, kron(num2, I1)) )
        Nc = Diagonal(kron(num3, kron(I2, I1)) )
    else
        # LSB ordering: Let the elements in the state vector be
        # |psi> = sum a_{ijk} (|i> kron |j> kron |k>, 
        # for i in [1,Nt1], j in [1,Nt2], , k in [1,Nt3]
        # In the vector representation of the state, qubit 1 varies the slowest and qubit 3 varies the fastest in the state vector
        # create the combined lowering and raising ops
        amat = kron(a1, kron(I2, I3))
        bmat = kron(I1, kron(a2, I3))
        cmat = kron(I1, kron(I2, a3))

        adag = Array(transpose(amat))
        bdag = Array(transpose(bmat))
        cdag = Array(transpose(cmat))

        # number operators
        Na = Diagonal(kron(num1, kron(I2, I3)) )
        Nb = Diagonal(kron(I1, kron(num2, I3)) )
        Nc = Diagonal(kron(I1, kron(I2, num3)) )
    end

    # Coupling Hamiltonian: couple_type = 2 # 1: cross-Kerr, 2: Jaynes-Cummings
    if couple_type == 1
        Hcouple = xab*(Na*Nb) + xac*(Na*Nc) + xbc*(Nb*Nc)
    elseif couple_type == 2
        Hcouple = xab*(amat * bdag + adag * bmat) + xac*(amat * cdag + adag * cmat) + xbc*(bmat * cdag + bdag * cmat)
    end

    # System Hamiltonian
    H0 = 2*pi*(da*Na + 0.5*xa*(Na*Na-Na) + db*Nb + 0.5*xb*(Nb*Nb-Nb) + dc*Nc + xc/2*(Nc*Nc-Nc) + Hcouple )

    H0 = Array(H0)

    # set up control hamiltonians
    Hsym_ops =[ amat+adag, bmat+bdag, cmat+cdag ]
    Hanti_ops=[ amat-adag, bmat-bdag, cmat-cdag ]

    if verbose
        println("*** Three coupled quantum systems setup ***")
        println("System Hamiltonian frequencies [GHz]: f01 = ", freq01, " rot. freq = ", f_rot)
        println("Anharmonicity = ", anharm)
        println("Coupling type = ", (couple_type==1) ? "X-Kerr" : "J-C", ". Coupling coeff = ", couple_coeff )
        println("Number of essential states = ", Ness, " Number of guard states = ", Nguard)
        println("Hamiltonians are of size ", Ntot, " by ", Ntot)
    end
    return H0, Hsym_ops, Hanti_ops, rot_freq
end
# initial parameter guess
function init_control(params::objparams; maxrand::Float64, nCoeff::Int64, startFile::String = "", seed::Int64 = -1)
    Nctrl = size(params.Cfreq,1)
    Nfreq = size(params.Cfreq,2)

    D1 = div(nCoeff, 2*Nctrl*Nfreq)

    if seed >= 0
        Random.seed!(seed)
    end

    nCoeff = 2*Nctrl*Nfreq*D1 # factor '2' is for sin/cos

    # initial parameter guess: from file or random guess?
    if length(startFile) > 0
        # use if you want to read the initial coefficients from file
        pcof0 = vec(readdlm(startFile)) # change to jld2?
        println("*** Starting from B-spline coefficients in file: ", startFile)
        @assert(nCoeff == length(pcof0))
    else
        pcof0 = maxrand * 2 .* (rand(nCoeff) .- 0.5)
        println("*** Starting from random pcof with amplitude ", maxrand)
    end

    return pcof0
end

function control_bounds(params::objparams, maxamp::Vector{Float64}, nCoeff::Int64, zeroCtrlBC::Bool)
    Nctrl  = size(params.Cfreq,1)
    Nfreq  = size(params.Cfreq,2)

    D1 = div(nCoeff, 2*Nctrl*Nfreq)

    if zeroCtrlBC
        @assert(D1 >= 5) # D1 smaller than 5 does not work with zero start & end conditions
    else
        @assert(D1 >=3)
    end

    # min and max coefficient values
    minCoeff, maxCoeff = Juqbox.assign_thresholds_freq(maxamp, Nctrl, Nfreq, D1)
    
    if zeroCtrlBC
        zero_start_end!(Nctrl, Nfreq, D1, minCoeff, maxCoeff) # maxCoeff stores the bounds for the controls amplitudes (zero at the boundary)
    end

    println("control_bounds: Number of coefficients per spline = ", D1, " Total number of parameters = ", nCoeff)

    return minCoeff, maxCoeff
end


function eigen_and_reorder(H0::Union{Matrix{ComplexF64},Matrix{Float64}})
    H0_eigen = eigen(H0)
    Ntot = size(H0_eigen.vectors,1)
    @assert(size(H0_eigen.vectors,2) == Ntot) #only square matrices

    # look for the largest element in each column
    maxrow = zeros(Int64, Ntot)
    for j in 1:Ntot
        maxrow[j] = argmax(abs.(H0_eigen.vectors[:,j]));
    end
    #println("maxrow: ", maxrow)

    # find inverse of maxrow
    perm = zeros(Int64, Ntot) 
    for j in 1:Ntot
        perm[j] = argmin(abs.(maxrow .- j))
    end
    #println("perm: ", perm)

    #println("jc = ", jc, " [GHz]")

    Utrans = H0_eigen.vectors[:,perm[:]]
    # make sure all diagonal elements in Utrans are positive
    for j in 1:Ntot
        if Utrans[j,j]<0.0
            Utrans[:,j] = - Utrans[:,j]
        end
    end 

    # println("Utrans = H0_eigen.vectors (re-ordered):")
    # for i in 1:nrows
    #     println(Utrans[i,:])
    # end
    # Check orthonormality
    Id_tot = Array{Float64, 2}(I, Ntot, Ntot)
    println("Orthonormality check: norm( Utrans' Utrans - I) :", norm(Utrans' * Utrans - Id_tot) )

    return H0_eigen.values[perm[:]], Utrans
end

function exists(x::Float64, invec::Vector{Float64}, prox_thres::Float64 = 5e-3)
    id = -1
    for i in 1:length(invec)
        if abs(invec[i] - x) < prox_thres
        id = i
        break
        end
    end
    return id
end
  
function get_resonances(;Ness::Vector{Int64}, Nguard::Vector{Int64}, Hsys::Matrix{Float64}, Hsym_ops::Vector{Matrix{Float64}}, thres::Float64 = 0.01)
    Nctrl = length(Hsym_ops)

    nrows = size(Hsys,1)
    ncols = size(Hsys,2)

    Nt = Ness + Nguard
    Ntot = prod(Nt)

    # setup mapping between 1-d and 2-d indexing (assumes classical Juqbox ordering)
    it2i1 = zeros(Int64, Ntot)
    it2i2 = zeros(Int64, Ntot)
    it2i3 = zeros(Int64, Ntot)
    is_ess = Array{Bool, 1}(undef, Ntot)
    is_ess .= false # initialize
    if Nctrl == 1
        itot = 0
        for i1=1:Nt[1]
            itot += 1
            it2i1[itot] = i1-1
            if i1 <= Ness[1] 
                is_ess[itot] = true
            end
        end
    elseif Nctrl == 2
        itot = 0
        for i2=1:Nt[2]
            for i1=1:Nt[1]
                itot += 1
                it2i1[itot] = i1-1
                it2i2[itot] = i2-1
                if i1 <= Ness[1] && i2 <= Ness[2]
                is_ess[itot] = true
                end
            end
        end
    elseif Nctrl == 3
        itot = 0
        for i3=1:Nt[3]
            for i2=1:Nt[2]
                for i1=1:Nt[1]
                    itot += 1
                    it2i1[itot] = i1-1
                    it2i2[itot] = i2-1
                    it2i3[itot] = i3-1
                    if i1 <= Ness[1] && i2 <= Ness[2] && i3 <= Ness[3]
                        is_ess[itot] = true
                    end
                end
            end
        end
    end
    # println("is_ess = ", is_ess)
    # println("it2i1= ", it2i1)
    # println("it2i2= ", it2i2)
    # println("it2i3= ", it2i3)

    # Note: if Hsys is diagonal, then Hsys_evals = diag(Hsys) and Utrans = IdentityMatrix
    Hsys_evals, Utrans = eigen_and_reorder(Hsys)

    # scale the eigen-values
    ka_delta = Hsys_evals./(2*pi) 

    #println("Re-computed Hdelta-eigenvals/(2*pi):")
    #println(ka_delta)

    ## Identify resonances for all controls ## 
    # Note: Only resonances between *essential* levels are considered

    resonances = []
    for i in 1:Nctrl
        # Transformation of control Hamiltonian (a+adag)
        Hctrl_a = Hsym_ops[i]
        Hctrl_a_trans = Utrans' * Hctrl_a * Utrans

        #initialize
        resonances_a =zeros(0)

        # identify resonant couplings in 'a+adag'
        println("\nResonances in ctrl ", i, ":")
        for i in 1:nrows
            for j in 1:i # Only consider transitions from lower to higher levels
                if abs(Hctrl_a_trans[i,j]) > thres
                    # Use only essential level transitions
                    if is_ess[i] && is_ess[j]
                        println(" resonance from (i1 i2 i3) = (", it2i1[j], it2i2[j], it2i3[j], ") to (i1 i2 i3) = (", it2i1[i], it2i2[i], it2i3[i], "), Freq = ", ka_delta[i] - ka_delta[j], " = l_", i, " - l_", j, ", coeff=", Hctrl_a_trans[i,j])
                        resi = ka_delta[i] - ka_delta[j]
                        if abs(resi) < 1e-10 
                            resi = 0.0
                        end
                        if exists(resi, resonances_a) < 0
                            append!(resonances_a, resi)
                        else
                            println("Info, skipping frequency: ", resi, " Too close to previous frequencies")
                        end
                    end
                end
            end
        end 
    
        # Store the result
        push!(resonances, resonances_a)
    end

    # Convert to radians
    resonances = resonances*2*pi

    # Return as a matrix, one row per control
    return Matrix(reduce(hcat, resonances)'), Utrans
end

function transformHamiltonians!(H0::Matrix{Float64}, Hsym_ops::Vector{Matrix{Float64}}, Hanti_ops::Vector{Matrix{Float64}}, Utrans::Matrix{Float64})
    H0 = Utrans'*H0*Utrans # use the transformed (diagonal) system Hamiltonian

    # transform the control Hamiltonians
    for q = 1:length(Hsym_ops)
        Hsym_ops[q] = Utrans'*(Hsym_ops[q])*Utrans
        Hanti_ops[q] = Utrans'*(Hanti_ops[q])*Utrans
    end
end
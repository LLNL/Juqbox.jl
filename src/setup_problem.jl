### Set up the Hamiltonians using the standard model 

using LinearAlgebra

function hamiltonians_one_sys(;Ness::Vector{Int64}, Nguard::Vector{Int64}, freq01::Vector{Float64}, anharm::Vector{Float64}, rot_freq::Vector{Float64}, verbose::Bool = true)
    @assert(length(Ness)==1)
    @assert(length(Nguard)==1)
    @assert(minimum(Ness) >= 2)
    @assert(minimum(Nguard) >= 0)

    Ntot = Ness[1] + Nguard[1] # Total number of energy levels

    # NOTE: input frequencies are in GHz, will be multiplied by 2*pi to get angular frequencies 

    # setup drift Hamiltonian
    number = Diagonal(collect(0:Ntot-1))
    # Note: xa is negative
    H0  = 2*pi * ( (freq01[1] - rot_freq[1])*number + 0.5*anharm[1]* (number*number - number) ) 
    H0 = Array(H0)

    # Set up the control drive operators
    amat = Bidiagonal(zeros(Ntot),sqrt.(collect(1:Ntot-1)),:U) # standard lowering matrix
    adag = Array(transpose(amat));
    Hsym_ops=[Array(amat + adag)]  # operator for p(t)
    Hanti_ops=[Array(amat - adag)] # operator for q(t)

    if verbose
        println("*** Single quantum system setup ***")
        println("System Hamiltonian coefficients [GHz]: f01 = ", freq01, " anharmonicity = ", anharm, " rot_freq = ", rot_freq)
        println("Number of essential states = ", Ness, " Number of guard states = ", Nguard)
        println("Hamiltonians are of size ", Ntot, " by ", Ntot)
    end
    return H0, Hsym_ops, Hanti_ops
end


function hamiltonians_two_sys(;Ness::Vector{Int64}, Nguard::Vector{Int64}, freq01::Vector{Float64}, anharm::Vector{Float64}, rot_freq::Vector{Float64}, couple_coeff::Vector{Float64}, couple_type::Int64, msb_order::Bool = true, verbose::Bool = true)
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
    x1 = anharm[1]
    x2 = anharm[2]

    # detuning
    da = freq01[1] - rot_freq[1]
    db = freq01[2] - rot_freq[2]

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

        # number operators
        N1 = Diagonal(kron(I2, num1) )
        N2 = Diagonal(kron(num2, I1) )

    else
        # LSB ordering: qubit 1 varies the slowest and qubit 2 varies the fastest in the state vector
        amat = Array(kron(a1, I2))
        bmat = Array(kron(I1, a2))

        # number operators
        N1 = Diagonal(kron(num1, I2) )
        N2 = Diagonal(kron(I1, num2) )
    end

    adag = Array(transpose(amat))
    bdag = Array(transpose(bmat))

    # Coupling Hamiltonian: couple_type = 2 # 1: cross-Kerr, 2: Jaynes-Cummings
    if couple_type == 1
        Hcouple = couple_coeff[1]*(N1*N2)
    elseif couple_type == 2
        Hcouple = couple_coeff[1]*(bdag * amat + bmat * adag)
    end
            # System Hamiltonian
    H0 = 2*pi*(  da*N1 + 0.5*x1*(N1*N1-N1) + db*N2 + 0.5*x2*(N2*N2-N2) + Hcouple )
    H0 = Array(H0)

    # set up control hamiltonians
    Hsym_ops =[ amat+adag, bmat+bdag ]
    Hanti_ops=[ amat-adag, bmat-bdag ]

    if verbose
        println("*** Two coupled quantum systems setup ***")
        println("System Hamiltonian frequencies [GHz]: f01 = ", freq01, " rot. freq = ", rot_freq)
        println("Anharmonicity = ", anharm, " coupling coeff = ", couple_coeff, " coupling type = ", (couple_type==1) ? "X-Kerr" : "J-C" )
        println("Number of essential states = ", Ness, " Number of guard states = ", Nguard)
        println("Hamiltonians are of size ", Ntot, " by ", Ntot)
    end
    return H0, Hsym_ops, Hanti_ops, rot_freq
end

function hamiltonians_three_sys(;Ness::Vector{Int64}, Nguard::Vector{Int64}, freq01::Vector{Float64}, anharm::Vector{Float64}, rot_freq::Vector{Float64}, couple_coeff::Vector{Float64}, couple_type::Int64, msb_order::Bool = true, verbose::Bool = true)
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
    xa = anharm[1]
    xb = anharm[2]
    xc = anharm[3]

    xab = couple_coeff[1]
    xac = couple_coeff[2]
    xbc = couple_coeff[3]

    # detuning frequencies
    da = freq01[1] - rot_freq[1]
    db = freq01[2] - rot_freq[2]
    dc = freq01[3] - rot_freq[3]

    # single system lowering ops
    a1 = Array(Bidiagonal(zeros(Nt[1]),sqrt.(collect(1:Nt[1]-1)),:U))
    a2 = Array(Bidiagonal(zeros(Nt[2]),sqrt.(collect(1:Nt[2]-1)),:U))
    a3 = Array(Bidiagonal(zeros(Nt[3]),sqrt.(collect(1:Nt[3]-1)),:U))

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

        # number operators
        Na = Diagonal(kron(num1, kron(I2, I3)) )
        Nb = Diagonal(kron(I1, kron(num2, I3)) )
        Nc = Diagonal(kron(I1, kron(I2, num3)) )
    end

    adag = Array(transpose(amat))
    bdag = Array(transpose(bmat))
    cdag = Array(transpose(cmat))

    # Coupling Hamiltonian: couple_type = 2 # 1: cross-Kerr, 2: Jaynes-Cummings
    if couple_type == 1
        Hcouple = xab*(Na*Nb) + xac*(Na*Nc) + xbc*(Nb*Nc)
    elseif couple_type == 2
        Hcouple = xab*(amat * bdag + adag * bmat) + xac*(amat * cdag + adag * cmat) + xbc*(bmat * cdag + bdag * cmat)
    end

    # System Hamiltonian
    H0 = 2*pi*(da*Na + 0.5*xa*(Na*Na-Na) + db*Nb + 0.5*xb*(Nb*Nb-Nb) + dc*Nc + 0.5*xc*(Nc*Nc-Nc) + Hcouple )

    H0 = Array(H0)

    # set up control hamiltonians
    Hsym_ops =[ amat+adag, bmat+bdag, cmat+cdag ]
    Hanti_ops=[ amat-adag, bmat-bdag, cmat-cdag ]

    if verbose
        println("*** Three coupled quantum systems setup ***")
        println("System Hamiltonian frequencies [GHz]: f01 = ", freq01, " rot. freq = ", rot_freq)
        println("Anharmonicity = ", anharm)
        println("Coupling type = ", (couple_type==1) ? "X-Kerr" : "J-C", ". Coupling coeff = ", couple_coeff )
        println("Number of essential states = ", Ness, " Number of guard states = ", Nguard)
        println("Hamiltonians are of size ", Ntot, " by ", Ntot)
    end
    return H0, Hsym_ops, Hanti_ops
end

function hamiltonians_four_sys(;Ness::Vector{Int64}, Nguard::Vector{Int64}, freq01::Vector{Float64}, anharm::Vector{Float64}, rot_freq::Vector{Float64}, couple_coeff::Vector{Float64}, couple_type::Int64, msb_order::Bool = true, verbose::Bool = true)
    @assert(length(Ness) == 4)
    @assert(length(Nguard) == 4)
    @assert(length(anharm) == 4)
    @assert(length(freq01) == 4)
    @assert(length(rot_freq) == 4)
    @assert(length(couple_coeff) == 6)
    @assert(couple_type == 1 || couple_type == 2)

    Nt = Ness + Nguard
    N = prod(Ness); # Total number of nonpenalized energy levels
    Ntot = prod(Nt)
    #Nguard = Ntot - N # total number of guard states


    # frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies 
    # in the Hamiltonian matrix)
    xa = anharm[1]
    xb = anharm[2]
    xc = anharm[3]
    xd = anharm[4]

    xab = couple_coeff[1]
    xac = couple_coeff[2]
    xad = couple_coeff[3]
    xbc = couple_coeff[4]
    xbd = couple_coeff[5]
    xcd = couple_coeff[6]

    # detuning frequencies
    da = freq01[1] - rot_freq[1]
    db = freq01[2] - rot_freq[2]
    dc = freq01[3] - rot_freq[3]
    dd = freq01[4] - rot_freq[4]

    # single system lowering ops
    a1 = Array(Bidiagonal(zeros(Nt[1]),sqrt.(collect(1:Nt[1]-1)),:U))
    a2 = Array(Bidiagonal(zeros(Nt[2]),sqrt.(collect(1:Nt[2]-1)),:U))
    a3 = Array(Bidiagonal(zeros(Nt[3]),sqrt.(collect(1:Nt[3]-1)),:U))
    a4 = Array(Bidiagonal(zeros(Nt[4]),sqrt.(collect(1:Nt[4]-1)),:U))

    I1 = Array{Float64, 2}(I, Nt[1], Nt[1])
    I2 = Array{Float64, 2}(I, Nt[2], Nt[2])
    I3 = Array{Float64, 2}(I, Nt[3], Nt[3])
    I4 = Array{Float64, 2}(I, Nt[4], Nt[4])

    # single system number ops
    num1 = Diagonal(collect(0:Nt[1]-1))
    num2 = Diagonal(collect(0:Nt[2]-1))
    num3 = Diagonal(collect(0:Nt[3]-1))
    num4 = Diagonal(collect(0:Nt[4]-1))

    if msb_order
        # MSB ordering: Let the elements in the state vector be
        # |psi> = sum a_{kji} (|k> kron |j> kron |i>, 
        # for i in [1,Nt1], j in [1,Nt2], , k in [1,Nt3]
        # We order the elements in the vector psi such that i varies the fastest 
        # The matrix amat = I kron I kron a1 acts on alpha in psi = gamma kron beta kron alpha
        # The matrix bmat = I kron a2 kron I acts on beta in psi = gamma kron beta kron alpha
        # The matrix cmat = a3 kron I2 kron I1 acts on gamma in psi = gamma kron beta kron alpha

        # create the combined lowering and raising ops
        amat = kron(I4, kron(I3, kron(I2, a1)))
        bmat = kron(I4, kron(I3, kron(a2, I1)))
        cmat = kron(I4, kron(a3, kron(I2, I1)))
        dmat = kron(a4, kron(I3, kron(I2, I1)))

        adag = Array(transpose(amat))
        bdag = Array(transpose(bmat))
        cdag = Array(transpose(cmat))
        ddag = Array(transpose(dmat))

        # number operators
        Na = Diagonal(kron(I4, kron(I3, kron(I2, num1))) )
        Nb = Diagonal(kron(I4, kron(I3, kron(num2, I1))) )
        Nc = Diagonal(kron(I4, kron(num3, kron(I2, I1))) )
        Nd = Diagonal(kron(num4, kron(I3, kron(I2, I1))) )
    else
        # LSB ordering: Let the elements in the state vector be
        # |psi> = sum a_{ijk} (|i> kron |j> kron |k>, 
        # for i in [1,Nt1], j in [1,Nt2], , k in [1,Nt3]
        # In the vector representation of the state, qubit 1 varies the slowest and qubit 3 varies the fastest in the state vector
        # create the combined lowering and raising ops
        amat = kron(a1, kron(I2, kron(I3, I4)))
        bmat = kron(I1, kron(a2, kron(I3, I4)))
        cmat = kron(I1, kron(I2, kron(a3, I4)))
        dmat = kron(I1, kron(I2, kron(I3, a4)))
        
        adag = Array(transpose(amat))
        bdag = Array(transpose(bmat))
        cdag = Array(transpose(cmat))
        ddag = Array(transpose(dmat))

        # number operators
        Na = Diagonal(kron(num1, kron(I2, kron(I3, I4))) )
        Nb = Diagonal(kron(I1, kron(num2, kron(I3, I4))) )
        Nc = Diagonal(kron(I1, kron(I2, kron(num3, I4))) )
        Nd = Diagonal(kron(I1, kron(I2, kron(I3, num4))) )
    end

    # Coupling Hamiltonian: couple_type = 2 # 1: cross-Kerr, 2: Jaynes-Cummings
    if couple_type == 1
        Hcouple = xab*(Na*Nb) + xac*(Na*Nc) + xad*(Na*Nd) + xbc*(Nb*Nc) + xbd*(Nb*Nd) + xcd*(Nc*Nd)
    elseif couple_type == 2
        Hcouple = xab*(amat * bdag + adag * bmat) + xac*(amat * cdag + adag * cmat) + xad*(amat * ddag + adag * dmat) + xbc*(bmat * cdag + bdag * cmat) + xbd*(bmat * ddag + bdag * dmat) + xcd*(cmat * ddag + cdag * dmat)
    end

    # System Hamiltonian
    H0 = 2*pi*(da*Na + 0.5*xa*(Na*Na-Na) + db*Nb + 0.5*xb*(Nb*Nb-Nb) + dc*Nc + 0.5*xc*(Nc*Nc-Nc) + dd*Nd + 0.5*xd*(Nd*Nd-Nd) + Hcouple )

    H0 = Array(H0)

    # set up control hamiltonians
    Hsym_ops =[ amat+adag, bmat+bdag, cmat+cdag, dmat+ddag ]
    Hanti_ops=[ amat-adag, bmat-bdag, cmat-cdag, dmat-ddag ]

    if verbose
        println("*** Four coupled quantum systems setup ***")
        println("System Hamiltonian frequencies [GHz]: f01 = ", freq01, " rot. freq = ", rot_freq)
        println("Anharmonicity = ", anharm)
        println("Coupling type = ", (couple_type==1) ? "X-Kerr" : "J-C", ". Coupling coeff = ", couple_coeff )
        println("Number of essential states = ", Ness, " Number of guard states = ", Nguard)
        println("Hamiltonians are of size ", Ntot, " by ", Ntot)
    end
    return H0, Hsym_ops, Hanti_ops
end

function hamiltonians(;Nsys::Int64, Ness::Vector{Int64}, Nguard::Vector{Int64}, freq01::Vector{Float64}, anharm::Vector{Float64}, rot_freq::Vector{Float64}, couple_coeff::Vector{Float64}, couple_type::Int64, msb_order::Bool = false, verbose::Bool = true)
    @assert(Nsys>=1)
    @assert(length(Ness) == Nsys)
    @assert(length(Nguard) == Nsys)
    @assert(length(anharm) == Nsys)
    @assert(length(freq01) == Nsys)
    @assert(length(rot_freq) == Nsys)
    @assert(length(couple_coeff) == div(Nsys^2 - Nsys,2)) # general number of coupling coeff's?
    @assert(couple_type == 1 || couple_type == 2)

    # inline functions for making identity, number, and lowering matrices for the sub-systems
    ident(n::Int64) = diagm(ones(n))
    lowering(n::Int64) = Bidiagonal(zeros(n),sqrt.(collect(1:n-1)),:U)
    number(n::Int64) = Diagonal(collect(0:n-1))

    Nt = Ness + Nguard
    N = prod(Ness); # Total number of nonpenalized energy levels
    Ntot = prod(Nt) # Total number of energy levels

    # Number and lowering ops 
    Num = Vector{Matrix{Float64}}(undef,Nsys)
    Amat = Vector{Matrix{Float64}}(undef,Nsys)
    for q=1:Nsys
        preIdent = ident.(Nt[1:q-1])
        numq = foldr(kron,preIdent,init=number(Nt[q])) # Number op for system 'q'
        lowq = foldr(kron,preIdent,init=lowering(Nt[q]))
        if q<Nsys
            postIdent = ident.(Nt[q+1:Nsys])
            Num[q]  = foldl(kron,postIdent,init=numq) # Full size number operator
            Amat[q] = foldl(kron,postIdent,init=lowq)
        else
            Num[q]  = numq # Full size number operator
            Amat[q] = lowq
        end
    end


    # System Hamiltonian (Duffing oscillator)
    H0 = zeros(Ntot,Ntot)
    for q=1:Nsys
        H0 += (freq01[q] - rot_freq[q]) * Num[q] + 0.5*anharm[q] * (Num[q]*Num[q] - Num[q])
    end
    # H0 = 2*pi*(da*Na + 0.5*xa*(Na*Na-Na) + db*Nb + 0.5*xb*(Nb*Nb-Nb) + dc*Nc + 0.5*xc*(Nc*Nc-Nc) + dd*Nd + 0.5*xd*(Nd*Nd-Nd) + Hcouple )

    # Coupling terms
    if couple_type == 1
        k=0
        for q=1:Nsys
            for p=q+1:Nsys
                k += 1
                H0 += couple_coeff[k]*Num[q]*Num[p]
            end
        end
        #Hcouple = xab*(Na*Nb) + xac*(Na*Nc) + xad*(Na*Nd) + xbc*(Nb*Nc) + xbd*(Nb*Nd) + xcd*(Nc*Nd)
    elseif couple_type == 2
        k=0
        for q=1:Nsys
            for p=q+1:Nsys
                k += 1
                H0 += couple_coeff[k] * (Amat[q]*(Amat[p])' + (Amat[q])'*Amat[p])
            end
        end
        #Hcouple = xab*(amat * bdag + adag * bmat) + xac*(amat * cdag + adag * cmat) + xad*(amat * ddag + adag * dmat) + xbc*(bmat * cdag + bdag * cmat) + xbd*(bmat * ddag + bdag * dmat) + xcd*(cmat * ddag + cdag * dmat)
    end
    
    H0 = 2*pi*Array(H0) # Mult by 2*pi to get to rad/ns

    # set up control hamiltonians
    Hsym_ops  = Vector{Matrix{Float64}}(undef,Nsys)
    Hanti_ops = Vector{Matrix{Float64}}(undef,Nsys)
    for q=1:Nsys
        Hsym_ops[q]  = Amat[q] + (Amat[q])'
        Hanti_ops[q] = Amat[q] - (Amat[q])'
    end
    # Hsym_ops =[ amat+adag, bmat+bdag, cmat+cdag, dmat+ddag ]
    # Hanti_ops=[ amat-adag, bmat-bdag, cmat-cdag, dmat-ddag ]

    if verbose
        println("*** ", Nsys, " coupled quantum systems setup ***")
        println("System Hamiltonian frequencies [GHz]: f01 = ", freq01, " rot. freq = ", rot_freq)
        println("Anharmonicity = ", anharm)
        println("Coupling type = ", (couple_type==1) ? "X-Kerr" : "J-C", ". Coupling coeff = ", couple_coeff )
        println("Number of essential states = ", Ness, " Number of guard states = ", Nguard)
        println("Hamiltonians are of size ", Ntot, " by ", Ntot)
    end
    return H0, Hsym_ops, Hanti_ops
end

# initial parameter guess
function init_control(;rand_frac::Float64, maxAmp::Vector{Float64}, D1::Int64, Nfreq::Vector{Int64}, startFile::String = "", seed::Int64 = -1)
    Nosc = length(Nfreq)
    nCoeff = 2*D1*sum(Nfreq)

    if seed >= 0
        Random.seed!(seed)
    end

    # initial parameter guess: from file or random guess?
    if length(startFile) > 0
        # use if you want to read the initial coefficients from file
        pcof0 = vec(readdlm(startFile)) # change to jld2?
        println("*** Starting from B-spline coefficients in file: ", startFile)
        @assert(nCoeff == length(pcof0))
    else
        pcof0 = zeros(nCoeff)
        offset = 0
        for q=1:Nosc
            if Nfreq[q] > 0
                maxrand = rand_frac*maxAmp[q]/sqrt(2)/Nfreq[q]
                Nq = 2*D1*Nfreq[q]
                pcof0[offset+1:offset+Nq] = maxrand * 2 * (rand(Nq) .- 0.5)
                offset += Nq
            end
        end
        #pcof0 = maxrand * 2 .* (rand(nCoeff) .- 0.5)
        println("*** Starting from RANDOM control vector with rand_frac = ", rand_frac)
    end

    return pcof0
end

function control_bounds(params::objparams, maxAmp::Vector{Float64})
    Nctrl  = length(params.Cfreq)
    Nfreq  = params.Nfreq
    NfreqTot  = params.NfreqTot
    nCoeff = params.nCoeff

    D1 = div(nCoeff, 2*NfreqTot)

    if params.zeroCtrlBC
        @assert D1 >= 5 "D1 smaller than 5 does not work with zero start & end conditions"
    else
        @assert D1 >=3 "D1 can not be less than 3"
    end

    # min and max coefficient values, maxamp[Nctrl]
    minCoeff, maxCoeff = Juqbox.assign_thresholds(params, D1, maxAmp)
    
    if params.zeroCtrlBC
        zero_start_end!(Nctrl, Nfreq, D1, minCoeff, maxCoeff) # maxCoeff stores the bounds for the controls amplitudes (zero at the boundary)
    end

    #println("Number of coefficients per spline = ", D1, " Total number of parameters = ", nCoeff)

    return minCoeff, maxCoeff
end


function eigen_and_reorder(H0::Union{Matrix{ComplexF64},Matrix{Float64}}, is_ess::Vector{Bool})
    H0_eigen = eigen(H0)
    Ntot = size(H0_eigen.vectors,1)
    @assert(size(H0_eigen.vectors,2) == Ntot) #only square matrices

    # test
    # println("H0 eigenvalues:", H0_eigen.values)

    # look for the largest element in each column
    # What if 2 elements have the same magnitude?
    maxrow = zeros(Int64, Ntot)
    for j in 1:Ntot
        maxrow[j] = argmax(abs.(H0_eigen.vectors[:,j]));
    end
    #println("maxrow: ", maxrow)
    #println()
    # pl1 = histogram(maxrow,bins=1:Ntot+1,bar_width=0.25, leg=:none)

    # loop over all columns and check maxrow for duplicates
    println()
    Ndup = 0 
    for j = 1:Ntot-1 
        for k = j+1:Ntot
            if maxrow[j] == maxrow[k]
                Ndup += 1
                #println("Warning: detected identical maxrow = ", maxrow[j], " in columns j = ", j, " and k = ", k, " is_ess = (", is_ess[j], is_ess[k], ")")
            end
        end
    end

    if Ndup > 0
        println()
        println("eigen_and_reorder: Found ", Ndup, " duplicate maxrow entries, attempting a correction of maxrow")
        println()
        col_pair = zeros(Int64, Ndup, 2)
        bad_maxrow = zeros(Int64, Ndup)
        ip = 0
        for j = 1:Ntot-1 # loop over all columns and record duplicate columns in maxrow
            for k = j+1:Ntot
                if maxrow[j] == maxrow[k]
                    ip += 1
                    bad_maxrow[ip] = maxrow[j]
                    col_pair[ip,1] = j
                    col_pair[ip,2] = k
                end
            end
        end

        println("Offending column pairs:")
        for q=1:Ndup
            println("maxrow = ", bad_maxrow[q], " column_pair: ", col_pair[q,:])
        end
        println()

        # count missing entries in maxrow
        Nzero = 0
        missing_maxrow = Int64[]
        for j = 1:Ntot
            nEach = 0
            for k =1:Ntot
                if maxrow[k] == j
                    nEach += 1
                end
            end
            if nEach == 0
                Nzero += 1
                println("Zero columns have maxrow index = ", j)
                push!(missing_maxrow, j)
            end
        end
        println()

        # helper function
        function find_match_rc(row_order::Vector{Int64}, missing_maxrow::Vector{Int64})
            println("row_order: ", row_order)
            println("missing_maxrow:", missing_maxrow)
            row1 = 0
            for r in row_order
                if r in missing_maxrow
                    # which position in missing_maxrow ?
                    imr = argmin(abs.(missing_maxrow[:] .- r))
                    row1 = r
                    println("found match for row1 = ", row1)
                    missing_maxrow[imr] = 0; # clear this item from the vector
                    return row1
                end
            end
            return row1
        end

        maxRow = 4 # consider this many of the largest elements in each row
        for q=1:Ndup
            # find the 2 largest entries in row = maxrow[q]
            col_order = sortperm(abs.(H0_eigen.vectors[bad_maxrow[q],:]),rev=true)
            println("q = ", q, " bad_maxrow[q] = ", bad_maxrow[q], " emat[col_order[1:2]] = ", abs.(H0_eigen.vectors[bad_maxrow[q], col_order[1:2]]), " col_order[1:2] = ", col_order[1:2])
            # find a matching column
            row1 = 0
            col1 = 0
            for c in col_order[1:2]
                row_order = sortperm(abs.(H0_eigen.vectors[:, c]),rev=true)
                println("col = ", c, " emat[row_order[1:maxRow]] = ", abs.(H0_eigen.vectors[row_order[1:maxRow], c]), " row_order[1:maxRow] = ", row_order[1:maxRow])
                row1 = find_match_rc(row_order[1:maxRow], missing_maxrow);
                col1 = c
                if row1 > 0 
                    break;
                end
            end
            # update maxrow for column = col1 to be row1
            if col1 > 0 && row1 > 0
                println("Assigning maxrow[", col1, "] = ", row1)
                println()
                maxrow[col1] = row1
            else
                println("Warning: q = ", q, " col = ", col1, " row = ", row1, " Unable to find matching entry in missing_row")
            end
        end

        # count entries in maxrow
        Nzero = 0
        Nmulti = 0
        for j = 1:Ntot
            nEach = 0
            for k =1:Ntot
                if maxrow[k] == j
                    nEach += 1
                end
            end
            if nEach == 0
                Nzero += 1
                println("Zero columns have maxrow index = ", j)
            elseif nEach >=2
                Nmulti += 1
                println(nEach, " columns have maxrow index = ", j)
            end
        end

        if Nzero > 0 && Nmulti > 0
            throw("eigen_and_reorder: Unsuccessful correction of maxrow")
        else
            println()
            println("eigen_and_reorder: Successful correction of maxrow")
            println()
        end

        # # look at row j=24
        # j=36
        # pl1 = scatter(abs.(H0_eigen.vectors[j,21:40]), lab="Row=36, col=(21:40)")
        # # columns 25 and 26
        # k=25
        # scatter!(abs.(H0_eigen.vectors[21:40,k]), lab="Row=(21:40), col=25")
        # k=26
        # scatter!(abs.(H0_eigen.vectors[21:40,k]), lab="Row=(21:40), col=26")

        # pl1 = scatter(sort(maxrow),lab="sort(maxrow)",leg=:top)
        # pl1 = histogram(maxrow, bins=1:Ntot+1, bar_width=0.25, leg=:none)
        # display(pl1)

    end # Ndup > 0

    # get the permutation vector
    s_perm = sortperm(maxrow)

    Utrans = H0_eigen.vectors[:,s_perm[:]]
    # make sure all diagonal elements in Utrans are positive
    for j in 1:Ntot
        if Utrans[j,j]<0.0
            Utrans[:,j] = - Utrans[:,j]
        end
    end 

    # H0diag = Utrans' * H0 * Utrans
    # println("H0diag = Utrans' * H0 * Utrans:")
    # for i in 1:Ntot
    #     for j in 1:Ntot
    #         @printf("%+6.2e, ", H0diag[i,j])
    #     end
    #     @printf("\n")
    # end
    # println("Utrans = H0_eigen.vectors (re-ordered):")
    # for i in 1:Ntot
    #     println(Utrans[i,:])
    # end
    # Check orthonormality
    Id_tot = Array{Float64, 2}(I, Ntot, Ntot)
    uniTest = norm(Utrans' * Utrans - Id_tot)
    println("Orthonormality check: norm( Utrans' Utrans - I) :", uniTest )
    if uniTest > 1e-9 
        throw(error("Something went wrong with the diagonalization"))
    end

    return H0_eigen.values[s_perm[:]], Utrans
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

function identify_essential_levels(Ness::Vector{Int64}, Nt::Vector{Int64}, msb_order::Bool)
    # setup mapping between 1-d and 2-d indexing
    @assert(length(Ness) == length(Nt))
    Nosc = length(Nt)

    Ntot = prod(Nt)
    it2in = zeros(Int64, Ntot, Nosc) # Translation between the 1-D index 1:Ntot and zero-based multi-dimensional indices, where dim = Nosc

    is_ess = Vector{Bool}(undef, Ntot)
    is_ess .= false # initialize

    # See bookmark: Julia: Multidimensional indexing
    if msb_order
        t = Tuple(1:x for x in Nt) # Make a tuple t = (1:Nt[1], 1:Nt[2], ..., 1:Nt[Nosc])
    else
        t = Tuple(1:x for x in reverse(Nt))
    end
    R = CartesianIndices(t) # Setup the CartesianIndices for an Nosc-dimensional array with the above size
    itot = 0
    if msb_order
        for ind in R
            itot += 1
            for j = 1:Nosc
                it2in[itot,j] = ind[j] - 1 # MSB ordering is the default for CartesianIndices
            end
        end
    else
        for ind in R
            itot += 1
            ess = true
            for j = 1:Nosc
                it2in[itot,Nosc-j+1] = ind[j] - 1 # LSB ordering needs to be reversal
                ess = ess && ind[j] <= Ness[Nosc-j+1]
            end
            is_ess[itot] = ess
        end
    end

    return is_ess, it2in
end

function get_resonances(is_ess::Vector{Bool}, it2in::Matrix{Int64};Ness::Vector{Int64}, Nguard::Vector{Int64}, Hsys::Matrix{Float64}, Hsym_ops::Vector{Matrix{Float64}}, Hanti_ops::Vector{Matrix{Float64}}, msb_order::Bool = true, cw_amp_thres::Float64, cw_prox_thres::Float64, rot_freq::Vector{Float64})
    Nosc = length(Hsym_ops)

    nrows = size(Hsys,1)
    ncols = size(Hsys,2)

    Nt = Ness + Nguard
    Ntot = prod(Nt)

    # is_ess, it2in = identify_essential_levels_old(Ness, Nt, msb_order)
    # for j = 1:length(is_ess)
    #     println("is_ess-orig: ", is_ess2[j], " new: ", is_ess[j])
    # end
    #throw("Temporary breakpoint")

    # Note: if Hsys is diagonal, then Hsys_evals = diag(Hsys) and Utrans = IdentityMatrix
    Hsys_evals, Utrans = eigen_and_reorder(Hsys, is_ess)

    # H0diag = Utrans' * Hsys * Utrans
    # println("get_resonances: H0diag = Utrans' * Hsys * Utrans:")
    # for i in 1:Ntot
    #     for j in 1:Ntot
    #         @printf("%+6.2e, ", H0diag[i,j])
    #     end
    #     @printf("\n")
    # end

    # scale the eigen-values
    ka_delta = Hsys_evals./(2*pi) 

    #println("Re-computed Hdelta-eigenvals/(2*pi):")
    #println(ka_delta)

    ## Identify resonances for all controls ## 
    # Note: Only resonances between *essential* levels are considered

    println("\nget_resonances: Ignoring couplings slower than (ad_coeff): ", cw_amp_thres, " and frequencies closer than: ", cw_prox_thres, " [GHz]")
        
    resonances = []
    speed = []
    for q in 1:Nosc
        # Transformation of control Hamiltonian (a+adag) - (a-adag) = 2*adag
        Hctrl_ad = Hsym_ops[q] - Hanti_ops[q] # raising op
        Hctrl_ad_trans = Utrans' * Hctrl_ad * Utrans

        #initialize
        resonances_a =zeros(0)
        speed_a = zeros(0)

        # identify resonant couplings in 'a+adag'
        println("\nResonances in oscillator # ", q, " Ignoring transitions with ad_coeff <: ", cw_amp_thres)
        for i in 1:nrows # Hsys is of size nrows x nrows
            for j in 1:i # Only consider transitions from lower to higher levels
                if abs(Hctrl_ad_trans[i,j]) >= cw_amp_thres
                    # Use only essential level transitions
                    if is_ess[i] && is_ess[j]
                        delta_f = ka_delta[i] - ka_delta[j]
                        if abs(delta_f) < 1e-10 
                            delta_f = 0.0
                        end
                        # Use all sufficiently separated resonance frequencies
                        if exists(delta_f, resonances_a, cw_prox_thres) < 0
                            append!(resonances_a, delta_f)
                            append!(speed_a, abs(Hctrl_ad_trans[i,j]))
                            #println(" resonance from (i1 i2 i3 i4) = (", it2i1[j], it2i2[j], it2i3[j], it2i4[j], ") to (i1 i2 i3 i4) = (", it2i1[i], it2i2[i], it2i3[i], it2i4[i], "), Freq = ", delta_f, " = l_", i, " - l_", j, ", ad_coeff=", abs(Hctrl_ad_trans[i,j]))
                            println(" resonance from (j-idx) = (", it2in[j,:], ") to (i-idx) = (", it2in[i,:], "), lab-freq = ", rot_freq[q] + delta_f, " = l_", i, " - l_", j, ", ad_coeff=", abs(Hctrl_ad_trans[i,j]))
                        # else
                        #     println("Info, skipping frequency: ", delta_f, " Too close to previous frequencies")
                        end
                    end
                end
            end
        end 
    
        # Store the result
        push!(resonances, resonances_a)
        push!(speed, speed_a)
    end

    # Return the number of frequencies for each control 
    Nfreq = zeros(Int64, Nosc)
    
    # Allocate Vector of pointers to the carrier frequencies
    om = Vector{Vector{Float64}}(undef, Nosc)
    rate = Vector{Vector{Float64}}(undef, Nosc)
    # copy over resonances[]
    for q in 1:Nosc
        Nfreq[q] = length(resonances[q])
        om[q] = zeros(Nfreq[q])
        om[q] .= resonances[q]
        rate[q] = zeros(Nfreq[q])
        rate[q] .= speed[q]
    end

    # Convert carrier frequencies to radians
    om = 2*pi * om

    # H0diag = Utrans' * Hsys * Utrans
    # println("get_resonances_2: H0diag = Utrans' * Hsys * Utrans:")
    # for i in 1:Ntot
    #     for j in 1:Ntot
    #         @printf("%+6.2e, ", H0diag[i,j])
    #     end
    #     @printf("\n")
    # end
    # println("typeof(Utrans): ", typeof(Utrans))
    println()
    return om, rate, Utrans
end

function transformHamiltonians!(H0::Matrix{Float64}, Hsym_ops::Vector{Matrix{Float64}}, Hanti_ops::Vector{Matrix{Float64}}, Utrans::Matrix{Float64})
    # transform (diagonalize) the system Hamiltonian
    H0 .= Utrans'*H0*Utrans # the . is essential, otherwise H0 is not changed in the calling fcn

    # println("transformHamiltonians!: H0 := Utrans' * H0 * Utrans:")
    # Ntot = size(H0,1)
    # for i in 1:Ntot
    #     for j in 1:Ntot
    #         @printf("%+6.2e, ", H0[i,j])
    #     end
    #     @printf("\n")
    # end

    # transform the control Hamiltonians
    for q = 1:length(Hsym_ops)
        Hsym_ops[q] .= Utrans'*(Hsym_ops[q])*Utrans  # Here the . is not required, but added for consistency
        Hanti_ops[q] .= Utrans'*(Hanti_ops[q])*Utrans
    end
end

"""
    wmat = wmatsetup_old(Ne, Ng[, msb_order])

Build the default positive semi-definite weighting matrix W to calculate the 
leakage into higher energy forbidden states
 
# Arguments
- `Ne::Array{Int64,1}`: Number of essential energy levels for each subsystem
- `Ng::Array{Int64,1}`: Number of guard energy levels for each subsystem
- `msb_order::Bool`: Ordering of the subsystems within the state vector (default is true)
"""
function wmatsetup_old(Ne::Array{Int64,1}, Ng::Array{Int64,1}, msb_order::Bool = true)

Nt = Ne + Ng
Ndim = length(Ne)
@assert(Ndim == 1 || Ndim == 2 || Ndim ==3 || Ndim ==4)

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
            # end Ndim == 3
        elseif Ndim == 4
            fact = 1e-3 #  0.1 # for more emphasis on the "forbidden" states. Old value: 0.1
            nForb = 0 # number of states with the highest index in at least one dimension
            q = 0
            for i4 = 1:Nt[4]
                for i3 = 1:Nt[3]
                    for i2 = 1:Nt[2]
                        for i1 = 1:Nt[1]
                            q += 1
                            # initialize temp variables
                            temp1 = 0.0
                            temp2 = 0.0
                            temp3 = 0.0
                            temp4 = 0.0
                            if i1 <= Ne[1] && i2 <= Ne[2] && i3 <= Ne[3] && i4 <= Ne[4]
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
                                if i4 > Ne[4]   #only included if at a guard level
                                    temp4 = fact^(Nt[4]-i4)
                                end

                                forbFact=1.0
                                w[q] = forbFact*max(temp1, temp2, temp3, temp4)

                                if i1 == Nt[1] || i2 == Nt[2] || i3 == Nt[3] || i4 == Nt[4]
                                    nForb += 1
                                end

                            end # if
                        end # for i1
                    end # for i2
                end # for i3
            end # for i4

            # normalize by the number of entries with w=1
            coeff = 10.0/nForb # was 1/nForb
        end # if ndim == 4
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
            # end Ndim == 2
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
            # end Ndim == 3
        elseif Ndim == 4
            fact = 1e-3 #  0.1 # for more emphasis on the "forbidden" states. Old value: 0.1
            nForb = 0 # number of states with the highest index in at least one dimension
            q = 0
            for i1 = 1:Nt[1]
                for i2 = 1:Nt[2]
                    for i3 = 1:Nt[3]
                        for i4 = 1:Nt[4]            
                            q += 1
                            # initialize temp variables
                            temp1 = 0.0
                            temp2 = 0.0
                            temp3 = 0.0
                            temp4 = 0.0
                            if i1 <= Ne[1] && i2 <= Ne[2] && i3 <= Ne[3] && i4 <= Ne[4]
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
                                if i4 > Ne[4]   #only included if at a guard level
                                    temp4 = fact^(Nt[4]-i4)
                                end

                                forbFact=1.0
                                w[q] = forbFact*max(temp1, temp2, temp3, temp4)

                                if i1 == Nt[1] || i2 == Nt[2] || i3 == Nt[3] || i4 == Nt[4]
                                    nForb += 1
                                end

                            end # if
                        end # for i1
                    end # for i2
                end # for i3
            end # for i4

            # normalize by the number of entries with w=1
            coeff = 10.0/nForb # was 1/nForb
        end # if ndim == 4
    end # lsb ordering

    # println("wmatsetup: Number of forbidden states = ", nForb, " scaling coeff = ", coeff)
end # if sum(Ng) > 0

wmat = coeff * Diagonal(w) # turn vector into diagonal matrix
return wmat
end

function wmatsetup(is_ess::Vector{Bool}, it2in::Matrix{Int64}, Ne::Vector{Int64}, Ng::Vector{Int64})

    @assert(length(Ne) == size(it2in,2))
    Ndim = length(Ne)
    Ntot = length(is_ess)
    w = zeros(Ntot)
    Nt = Ne + Ng # total # of levels for each dimension

    coeff = 1.0

    if sum(Ng) > 0
        nForb = 0 # number of states with the highest index in at least one dimension
        temp = zeros(Ndim) # temp variable

        # Rather ad-hoc to conform with previous implementation
        if Ndim == 1
            bfact = 0.1 
        else
            bfact = 1e-3 
        end
        forbFact = 1.0

        # Re-use the indexing info from is_ess and it2in
        for q = 1:Ntot
            if !is_ess[q]
                temp[:] .= 0.0
                is_forb = false
                for k=1:Ndim
                    ik = it2in[q,k] + 1 # it2in is in [0, Nt[k]-1]
                    if ik > Ne[k]
                        temp[k] = bfact^(Nt[k] - ik)
                    end
                    is_forb = is_forb || (ik == Nt[k]) 
                end
                if is_forb
                    nForb += 1
                end
                w[q] = forbFact*maximum(temp)
            end
        end
        
        if Ndim <= 2
            coeff = 1.0/nForb # Normalize with respect to the totl number of forbidden levels
        else
            coeff = 10.0/nForb
        end
                
        # println("wmatsetup: Number of forbidden states = ", nForb, " scaling coeff = ", coeff)
    end # if sum(Ng) > 0

    wmat = coeff * Diagonal(w) # turn vector into diagonal matrix
    return wmat
end

function get_H4_gate()
	# 4-dimensional Discrete Fourier Transform unitary gate
	gate_H4 =  zeros(ComplexF64, 4, 4)
	gate_H4[1,1] = 1
	gate_H4[1,2] = 1
	gate_H4[1,3] = 1
	gate_H4[1,4] = 1
	gate_H4[2,1] = 1
	gate_H4[2,2] = im
	gate_H4[2,3] = -1
	gate_H4[2,4] = -im
	gate_H4[3,1] = 1
	gate_H4[3,2] = -1
	gate_H4[3,3] = 1
	gate_H4[3,4] = -1
	gate_H4[4,1] = 1
	gate_H4[4,2] = -im
	gate_H4[4,3] = -1
	gate_H4[4,4] = im
	gate_H4 .*= 0.5 # Normalize

	return gate_H4
end

function get_Hd_gate(d::Int64)
	# Set the target: d-dimensional Discrete Fourier Transform unitary gate 
    @assert(d>= 2)
	gate_Hd =  zeros(ComplexF64, d, d)
    om_d = exp(im*2*pi/d)

    for j=1:d
        for k = 1:d
            gate_Hd[j,k] = om_d^((j-1)*(k-1))
        end
    end
	gate_Hd ./= sqrt(d) # Normalize

	return gate_Hd
end

function get_CNOT()
    gate_H4 =  zeros(ComplexF64, 4, 4)
    gate_H4[1,1] = 1.0
    gate_H4[2,2] = 1.0
    gate_H4[3,4] = 1.0
    gate_H4[4,3] = 1.0
    return gate_H4
end

function get_CpNOT(p::Int64 = 1)
    # Number of qubits = p+1
    # First p qubits control qubit p+1
    # Unitary has size d = N x N, N = 2^(p+1)
    N = 2^(p+1)

    # Initialize as identity matrix
    gate = Matrix{ComplexF64}(I,N,N)

    # correct the final 2x2 block
    gate[N-1,N-1] = 0.0
    gate[N-1,N] = 1.0
    gate[N,N-1] = 1.0
    gate[N,N] = 0.0

    return gate
end

# This returns the threewave Hamiltonian matrix 3x3 or 4x4
function get_Threewave_Hamiltonian(theta,s, dim=3)
    v0 = exp.(im*theta)
    v0p = exp.(-im*theta)
    v1 = sqrt.(2*(s-1))
    v2 = sqrt.(2*s)
    if dim == 3
        return [0 v0*v1 0;v0p*v1 0 v0*v2;0 v0p*v2 0]
    elseif dim == 4
        return [0 v0*v1 0 0;v0p*v1 0 v0*v2 0;0 v0p*v2 0 0; 0 0 0 1]
    else
        println("Error in threewave gate setup: dim=", dim, ", but should be 3 or 4. Exiting.\n")
        stop
    end
end

# This returns the threewave gate for stepsize dt 3x3
function get_Threewave_gate(dt, dim=3)
    theta = pi/2
    s = 2
    threewave_Ham = get_Threewave_Hamiltonian(theta,s, dim)
    threewave_gate = exp(-im*dt*threewave_Ham)
    return threewave_gate
end

function get_iSWAP()
    gate =zeros(ComplexF64, 4,4)
    gate[1,1] = 1.0
    gate[2,3] = im*1.0
    gate[3,2] = im*1.0
    gate[4,4] = 1.0
end

function get_swap_1d_gate(d::Int64 = 2)
    if d == 2
        swap_gate = zeros(ComplexF64, 4, 4)
        swap_gate[1,1] = 1.0
        swap_gate[2,3] = 1.0
        swap_gate[3,2] = 1.0
        swap_gate[4,4] = 1.0
    elseif d == 3
        swap_gate =  zeros(ComplexF64, 8, 8)
        swap_gate[1,1] = 1.0
        swap_gate[2,5] = 1.0
        swap_gate[3,3] = 1.0
        swap_gate[4,7] = 1.0
        swap_gate[5,2] = 1.0
        swap_gate[6,6] = 1.0
        swap_gate[7,4] = 1.0
        swap_gate[8,8] = 1.0
    elseif d == 4
        swap_gate =  zeros(ComplexF64, 16, 16)
        swap_gate[1,1] = 1.0
        swap_gate[2,9] = 1.0
        swap_gate[3,3] = 1.0
        swap_gate[4,11] = 1.0
        swap_gate[5,5] = 1.0
        swap_gate[6,13] = 1.0
        swap_gate[7,7] = 1.0
        swap_gate[8,15] = 1.0
        swap_gate[9,2] = 1.0
        swap_gate[10,10] = 1.0
        swap_gate[11,4] = 1.0
        swap_gate[12,12] = 1.0
        swap_gate[13,6] = 1.0
        swap_gate[14,14] = 1.0
        swap_gate[15,8] = 1.0
        swap_gate[16,16] = 1.0
    else
        @assert false "Only implemented swap1d gates for d={2,3,4}"
    end
    return swap_gate
end

function get_ident_kron_swap23()
    # I kron Swap: Do a swap between osc 2 & 3, and the identity on osc 1
    # (0,0): 1, 0, 0, 0, 0, 0, 0, 0, 
    # (1,0): 0, 0, 1, 0, 0, 0, 0, 0, 
    # (2,0): 0, 1, 0, 0, 0, 0, 0, 0, 
    # (3,0): 0, 0, 0, 1, 0, 0, 0, 0, 
    # (4,0): 0, 0, 0, 0, 1, 0, 0, 0, 
    # (5,0): 0, 0, 0, 0, 0, 0, 1, 0, 
    # (6,0): 0, 0, 0, 0, 0, 1, 0, 0, 
    # (7,0): 0, 0, 0, 0, 0, 0, 0, 1, 
    swap_gate =  zeros(ComplexF64, 8, 8)
    swap_gate[1,1] = 1.0
    swap_gate[2,3] = 1.0
    swap_gate[3,2] = 1.0
    swap_gate[4,4] = 1.0
    swap_gate[5,5] = 1.0
    swap_gate[6,7] = 1.0
    swap_gate[7,6] = 1.0
    swap_gate[8,8] = 1.0
    return swap_gate
end

function get_swap12_kron_ident()
    # I kron Swap: Do a swap between osc 2 & 3, and the identity on osc 1
    # (0,0): 1, 0, 0, 0, 0, 0, 0, 0, 
    # (1,0): 0, 1, 0, 0, 0, 0, 0, 0, 
    # (2,0): 0, 0, 0, 0, 1, 0, 0, 0, 
    # (3,0): 0, 0, 0, 0, 0, 1, 0, 0, 
    # (4,0): 0, 0, 1, 0, 0, 0, 0, 0, 
    # (5,0): 0, 0, 0, 1, 0, 0, 0, 0, 
    # (6,0): 0, 0, 0, 0, 0, 0, 1, 0, 
    # (7,0): 0, 0, 0, 0, 0, 0, 0, 1, 
    swap_gate =  zeros(ComplexF64, 8, 8)
    swap_gate[1,1] = 1.0
    swap_gate[2,2] = 1.0
    swap_gate[3,5] = 1.0
    swap_gate[4,6] = 1.0
    swap_gate[5,3] = 1.0
    swap_gate[6,4] = 1.0
    swap_gate[7,7] = 1.0
    swap_gate[8,8] = 1.0
    return swap_gate
end

function setup_std_model(Ne::Vector{Int64}, Ng::Vector{Int64}, f01::Vector{Float64}, xi::Vector{Float64}, couple_coeff::Vector{Float64}, couple_type::Int64, rot_freq::Vector{Float64}, T::Float64, D1::Int64, gate_final::Matrix{ComplexF64}; maxctrl_MHz::Float64=10.0, msb_order::Bool = true, Pmin::Int64 = 40, rand_frac::Float64=0.0, rand_seed::Int64=2345, pcofFileName::String="", zeroCtrlBC::Bool = true, use_eigenbasis::Bool = false, cw_amp_thres::Float64=5e-2, cw_prox_thres::Float64=2e-3)

    # enforce inequality constraint on the leakage?
    useLeakIneq = false # true
    leakThreshold = 1e-3
  
    # convert maxctrl_MHz to rad/ns per frequency
    # This is (approximately) the max amplitude of each control function (p & q)
    maxctrl_radns = 2*pi * maxctrl_MHz * 1e-3 
  
    pdim = length(Ne)
    # @assert pdim <= 4 "Hamiltonian setup only implemented for up to 4 sub-systems"
    # if pdim==1
    #   Hsys, Hsym_ops, Hanti_ops = hamiltonians_one_sys(Ness=Ne, Nguard=Ng, freq01=f01, anharm=xi, rot_freq=rot_freq)
    #   #    Hsys, Hsym_ops, Hanti_ops, om, rot_freq = setup_free(Ne[1], Ng[1], f[1], xi[1], rfreq)
    # elseif pdim == 2
    #   Hsys, Hsym_ops, Hanti_ops = hamiltonians_two_sys(Ness=Ne, Nguard=Ng, freq01=f01, anharm=xi, rot_freq=rot_freq, couple_coeff=couple_coeff, couple_type=couple_type, msb_order=msb_order)
    #   # Hsys, Hsym_ops, Hanti_ops, om, rot_freq = setup_twoqubit_free(Ne, Ng, f, xi, couple_coeff, couple_type)
    # elseif pdim == 3
    #   Hsys, Hsym_ops, Hanti_ops = hamiltonians_three_sys(Ness=Ne, Nguard=Ng, freq01=f01, anharm=xi, rot_freq=rot_freq, couple_coeff=couple_coeff, couple_type=couple_type, msb_order = msb_order)
    # elseif pdim == 4
    #     Hsys, Hsym_ops, Hanti_ops = hamiltonians_four_sys(Ness=Ne, Nguard=Ng, freq01=f01, anharm=xi, rot_freq=rot_freq, couple_coeff=couple_coeff, couple_type=couple_type, msb_order = msb_order)  
    # end

    # General case
    Hsys, Hsym_ops, Hanti_ops = hamiltonians(Nsys=pdim, Ness=Ne, Nguard=Ng, freq01=f01, anharm=xi, rot_freq=rot_freq, couple_coeff=couple_coeff, couple_type=couple_type, msb_order = msb_order)  

    is_ess, it2in = identify_essential_levels(Ne, Ne+Ng, msb_order)

    # println("Results from identify_essential_levels:")
    # for j = 1:length(is_ess)
    #     println("j: ", j, " it2in[j,:]: ", it2in[j,:], " is_ess[j]: ", is_ess[j])
    # end


    om, rate, Utrans = get_resonances(is_ess, it2in, Ness=Ne, Nguard=Ng, Hsys=Hsys, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, msb_order=msb_order, cw_amp_thres=cw_amp_thres, cw_prox_thres=cw_prox_thres, rot_freq=rot_freq)
    
    Ness = prod(Ne)
    Nosc = length(om) 
    @assert(Nosc == pdim) # Nosc must equal pdim
    Nfreq = zeros(Int64,Nosc) # Number of frequencies per control Hamiltonian
    for q=1:Nosc
        Nfreq[q] = length(om[q])
    end

    println("D1 = ", D1, " Ness = ", Ness, " Nosc = ", Nosc, " Nfreq = ", Nfreq)
  
    # Amplitude bounds to be imposed during optimization
    maxAmp = maxctrl_radns * ones(Nosc) # internally scaled by 1/(sqrt(2)*Nfreq[q]) in setup_ipopt() and Quandary

    # allocate and sort the vectors (ascending order)
    om_p = Vector{Vector{Float64}}(undef, Nosc)
    rate_p = Vector{Vector{Float64}}(undef, Nosc)
    use_p = Vector{Vector{Int64}}(undef, Nosc)
    for q = 1:Nosc
        om_p[q] = zeros(Nfreq[q])
        rate_p[q] = zeros(Nfreq[q])
        use_p[q] = zeros(Int64,Nfreq[q]) # By default, don't use any freq's
        p = sortperm(om[q]) # sortperm(rate[q],rev=true)
        om_p[q] .= om[q][p]
        rate_p[q] .= rate[q][p]
    end

    println("Sorted CW freq's:")
    for q = 1:Nosc
      println("Ctrl Hamiltonian # ", q, ", lab frame carrier frequencies: ", rot_freq[q] .+ om_p[q]./(2*pi), " [GHz]")
      println("Ctrl Hamiltonian # ", q, ",                   growth rate: ", rate_p[q], " [1/ns]")
    end

    # Try to identify groups of almost equal frequencies
    for q = 1:Nosc
        seg = 0
        rge_q = maximum(om_p[q]) - minimum(om_p[q]) # this is the range of frequencies
        k0 = 1
        for k = 2:Nfreq[q]
            delta_k = om_p[q][k] - om_p[q][k0]
            if delta_k > 0.1*rge_q
                seg += 1
                # find the highest rate within the range [k0,k-1]
                rge = k0:k-1
                om_avg = sum(om_p[q][rge])/length(rge)
                println("Osc # ", q, " segment # ", seg, " Freq-range: ", (maximum(om_p[q][rge]) - minimum(om_p[q][rge]))/(2*pi), " Freq-avg: ", om_avg/(2*pi) + rot_freq[q])
                # kmax = argmax(rate_p[q][rge])
                use_p[q][k0] = 1
                # average the cw frequency over the segment
                om_p[q][k0] = om_avg 
                k0 = k # start a new group
            end
        end
        # find the highest rate within the last range [k0,Nfreq[q]]
        seg += 1
        rge = k0:Nfreq[q]
        om_avg = sum(om_p[q][rge])/length(rge)
        println("Osc # ", q, " segment # ", seg, " Freq-range: ", (maximum(om_p[q][rge]) - minimum(om_p[q][rge]))/(2*pi), " Freq-avg: ", om_avg/(2*pi) + rot_freq[q])
        # kmax = argmax(rate_p[q][rge])
        use_p[q][k0] = 1
        om_p[q][k0] = om_avg 

        # cull out unused frequencies
        om[q] = zeros(sum(use_p[q]))
        j = 0
        for k=1:Nfreq[q]
            if use_p[q][k] == 1
                j += 1
                om[q][j] = om_p[q][k]
            end
        end
        Nfreq[q] = j
    end

    println("\nSorted and culled CW freq's:")
    for q = 1:Nosc
      println("Ctrl Hamiltonian # ", q, ", lab frame carrier frequencies: ", rot_freq[q] .+ om[q]./(2*pi), " [GHz]")
    end
  
    # Set the initial condition: Basis with guard levels
    # Ubasis2 = initial_cond_old(Ne, Ng, msb_order) # Ubasis holds the basis that will be used as initial cond.
    Ubasis = initial_cond_general(is_ess, Ne, Ng)

    #println("Testing initial conditions, norm(U - Unew): ", norm(Ubasis-Ubasis2))
    #throw("Intentionally stopping here")

    # NOTE:
    # To impose the target transformation in the eigenbasis, keep the Hamiltonians the same
    # but change the target to be Utrans*Ubasis*gate_final

    if use_eigenbasis
        Utarget = Utrans * Ubasis * gate_final
    else
        Utarget = Ubasis * gate_final
    end

    # use_diagonal_H0 = false  # For comparisson with Quandary: use original Hamiltonian
    # if use_diagonal_H0 # transformation to diagonalize the system Hamiltonian
    #   transformHamiltonians!(Hsys, Hsym_ops, Hanti_ops, Utrans) 
    # end
  
    # Set up the initial control vector
    nCoeff = 2*D1*sum(Nfreq) # factor '2' is for Re/Im parts of ctrl vector
  
    # Set up the initial control parameter  
    pcof0 = init_control(rand_frac = rand_frac, maxAmp=maxAmp, D1=D1, Nfreq=Nfreq, startFile=pcofFileName, seed=rand_seed)
  
    # Estimate time step based on the number of time steps per shortest period
  
    # Note: calculate_timestep expects maxCoupled to have Nosc elements
    nsteps = calculate_timestep(T, Hsys, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, maxCoupled=maxAmp, Pmin=Pmin)
    println("Starting point: nsteps = ", nsteps, " maxAmp = ", maxAmp, " [rad/ns]")
    
    # create a linear solver object
    linear_solver = Juqbox.lsolver_object(solver=Juqbox.JACOBI_SOLVER, max_iter=100, tol=1e-12, nrhs=prod(Ne))
  
    # create diagonal W-matrix with weights for suppressing leakage
    wmatScale = 1.0
    #w_diag_mat = wmatScale * wmatsetup_old(Ne, Ng, msb_order)
    w_diag_mat = wmatsetup(is_ess, it2in, Ne, Ng)

    # println("norm(wmat1 - wmat2): ", norm(w_diag_mat-w_diag_2))
    # println("w_diag_1: ", diag(w_diag_mat))
    # println("w_diag_2: ", diag(w_diag_2))
    # println("differen: ", diag(w_diag_mat-w_diag_2))

    # Set up parameter struct using the free evolution target
    if useLeakIneq
      params = Juqbox.objparams(Ne, Ng, T, nsteps, Uinit=Ubasis, Utarget=Utarget, Cfreq=om, Rfreq=rot_freq, Hconst=Hsys, w_diag_mat=w_diag_mat, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, linear_solver=linear_solver, objFuncType=3, leak_ubound=leakThreshold, nCoeff=nCoeff, msb_order=msb_order)
    else
      params = Juqbox.objparams(Ne, Ng, T, nsteps, Uinit=Ubasis, Utarget=Utarget, Cfreq=om, Rfreq=rot_freq, Hconst=Hsys, w_diag_mat=w_diag_mat, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, linear_solver=linear_solver, nCoeff=nCoeff, freq01=f01, self_kerr=xi, couple_coeff=couple_coeff, couple_type=couple_type, msb_order=msb_order, zeroCtrlBC=zeroCtrlBC)
    end
  
    println("*** Settings ***")
    println("Number of coefficients per spline = ", D1, " Total number of control parameters = ", length(pcof0))
    println()
    println("Returning problem setup as a tuple (params, pcof0, maxAmp)")
    println("params::objparams: object holding the Hamiltonians, carrier freq's, time-stepper, etc")
    println("pcof0:: Vector{Float64}: Initial coefficient vector is stored in 'pcof0' vector")
    println("maxAmp:: Vector{Float64}: Approximate max control amplitude for the p(t) and q(t) control function for each control Hamiltonian")
    println("")
  
    return params, pcof0, maxAmp
  end
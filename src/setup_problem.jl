### Set up the Hamiltonians using the standard model 

using LinearAlgebra
using Documenter

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

# initial control guess
function init_control(; amp_frac::Float64, maxAmp::Vector{Float64}, D1::Int64, Nfreq::Vector{Int64}, startFile::String = "", seed::Int64 = -1, randomize::Bool = true, growth_rate::Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef,0), nTimeIntervals::Int64=1, U0::Matrix{ComplexF64}=Matrix{ComplexF64}(undef,0,0), Utarget::Matrix{ComplexF64}=Matrix{ComplexF64}(undef,0,0))

    Nosc = length(Nfreq)
    nAlpha = 2*D1*sum(Nfreq)
# pcof array now contains alpha-vector and intermediate initial conditions
    if nTimeIntervals > 1
        Ntot = size(U0, 1)
        nCoeff = nAlpha + (nTimeIntervals - 1)*2*Ntot*Ntot
    else
        nCoeff = nAlpha
    end

    
 
    # initial parameter guess: from file?
    if length(startFile) > 0
        # use if you want to read the initial coefficients from file
        pcof0 = vec(readdlm(startFile)) # change to jld2?
        println("*** Starting from design coefficients in file: ", startFile)
        @assert(nCoeff == length(pcof0))
    else
        if seed >= 0
            Random.seed!(seed)
        end

        pcof0 = zeros(nCoeff)
        offc = 0

        # randomize?
        if length(growth_rate) == 0
            randomize = true
        end

        if randomize
            for q=1:Nosc
                if Nfreq[q] > 0
                    maxrand = amp_frac*maxAmp[q]/sqrt(2)/Nfreq[q]
                    Nq = 2*D1*Nfreq[q]
                    pcof0[offc+1:offc+Nq] = maxrand * 2 * (rand(Nq) .- 0.5)
                    offc += Nq
                end
            end
            println("*** Starting from RANDOM control vector with amp_frac = ", amp_frac)
        else # picewise constant with amplitude depending on scaled growth rate
            max_rate = 0.0
            for q = 1:Nosc
                max_rate = max( max_rate, maximum(growth_rate[q]) )
            end
            println("max_rate = ", max_rate)
            for q = 1:Nosc
                for k = 1:Nfreq[q]
                    const_amp = amp_frac * maxAmp[q]/sqrt(2)/Nfreq[q] * max_rate/(growth_rate[q][k])
                    pcof0[offc+1:offc+2*D1] .= const_amp
        
                    # randomizing p/q knocks out any Rabi-style oscillation
                    # pcof0[offc+1:offc+D1] = fact*(rand(D1) .- 0.5)
                    # pcof0[offc+D1+1:offc+2*D1] = fact*(rand(D1) .- 0.5)
                    offc += 2*D1
                end
            end
            println("*** Starting from PIECEWISE CONSTANT control vector with amp_frac = ", amp_frac)
        end

    end

    # Add initial conditions for intermediate time-intervals
    if nTimeIntervals>1
        offc = nAlpha
        ds = 1.0/nTimeIntervals
        Hdelta = im*log(U0'*Utarget)
        nMat = Ntot^2
        for q = 2:nTimeIntervals
            s = (q-1)*ds
            Winit = U0 * exp(-im*s*Hdelta)  # the geodesic from U0 to Utarget
            pcof0[offc+1:offc+nMat] = vec(real(Winit)) # save real part
            offc += nMat
            pcof0[offc+1:offc+nMat] = vec(imag(Winit)) # save imaginary part
            offc += nMat
        end

    end

    return pcof0
end

function control_bounds(params::objparams, maxAmp::Vector{Float64})
    Nctrl  = length(params.Cfreq)
    Nfreq  = params.Nfreq
    NfreqTot  = params.NfreqTot
    nCoeff = params.nCoeff

    D1 = params.D1
    #div(nCoeff, 2*NfreqTot)

    if params.zeroCtrlBC
        @assert D1 >= 5 "D1 smaller than 5 does not work with zero start & end conditions"
    else
        @assert D1 >=3 "D1 can not be less than 3"
    end

    # min and max coefficient values, maxamp[Nctrl]
    minCoeff, maxCoeff = Juqbox.assign_thresholds(params, maxAmp)
    
    if params.zeroCtrlBC
        zero_start_end!(Nctrl, Nfreq, D1, minCoeff, maxCoeff) # maxCoeff stores the bounds for the controls amplitudes (zero at the boundary)
    end

    #println("Number of coefficients per spline = ", D1, " Total number of parameters = ", nCoeff)

    return minCoeff, maxCoeff
end


function eigen_and_reorder(H0::Union{Matrix{ComplexF64},Matrix{Float64}}, is_ess::Vector{Bool}, verbose::Bool = false)

    H0_eigen = eigen(H0)
    Ntot = size(H0_eigen.vectors,1)
    @assert(size(H0_eigen.vectors,2) == Ntot) #only square matrices

    # test
    if verbose
        println("H0 eigenvalues (before sorting):", H0_eigen.values)
    end

    # look for the largest element in each column
    # What if 2 elements have the same magnitude?
    maxrow = zeros(Int64, Ntot)
    for j in 1:Ntot
        maxrow[j] = argmax(abs.(H0_eigen.vectors[:,j]));
    end
    
    if verbose
        println("maxrow: ", maxrow)
        println()
    end
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

    if verbose
        println("Ndup = ", Ndup)
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

    if verbose
        println("s_perm = ", s_perm)
    end

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

function get_resonances(is_ess::Vector{Bool}, it2in::Matrix{Int64};Ness::Vector{Int64}, Nguard::Vector{Int64}, Hsys::Matrix{Float64}, Hsym_ops::Vector{Matrix{Float64}}, Hanti_ops::Vector{Matrix{Float64}}, cw_amp_thres::Float64, cw_prox_thres::Float64, rot_freq::Vector{Float64}, verbose::Bool=false)
    # Enable verbose mode for debug printout

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
    Hsys_evals, Utrans = eigen_and_reorder(Hsys, is_ess, verbose)

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

    if verbose
        println("Hsys eigenvals/(2*pi):")
        println(ka_delta)
    end

    ## Identify resonances for all controls ## 
    # Note: Only resonances between *essential* levels are considered

    println("\nget_resonances: Ignoring couplings slower than (ad_coeff): ", cw_amp_thres, " and frequencies closer than: ", cw_prox_thres, " [GHz]")
        
    resonances = []
    speed = []
    for q in 1:Nosc
        # Transformation of control Hamiltonian (a+adag) - (a-adag) = 2*adag
        Hctrl_ad = Hsym_ops[q] - Hanti_ops[q] # raising op
        Hctrl_ad_trans = Utrans' * Hctrl_ad * Utrans

        # if verbose
        #     println("q = ", q, " Hctrl_ad_trans = ", Hctrl_ad_trans)
        # end
        #initialize
        resonances_a =zeros(0)
        speed_a = zeros(0)

        # identify resonant couplings in 'a+adag'
        println("\nResonances in oscillator # ", q, " Ignoring transitions with ad_coeff <: ", cw_amp_thres)
        for i in 1:nrows # Hsys is of size nrows x nrows
            for j in 1:i # Only consider transitions from lower to higher levels
                if verbose
                    println("i=", i, " j=", j, " is_ess[i]=", is_ess[i], " is_ess[j]", is_ess[j])
                end
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
    growth_rate = Vector{Vector{Float64}}(undef, Nosc)
    # copy over resonances[]
    for q in 1:Nosc
        Nfreq[q] = length(resonances[q])
        om[q] = zeros(Nfreq[q])
        om[q] .= resonances[q]
        growth_rate[q] = zeros(Nfreq[q])
        growth_rate[q] .= speed[q]
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
    return om, growth_rate, Utrans
end

function sort_and_cull_carrier_freqs(Nosc::Int64, Nfreq::Vector{Int64}, om::Vector{Vector{Float64}}, growth_rate::Vector{Vector{Float64}}, rot_freq::Vector{Float64})
    # allocate and sort the vectors (ascending order)
    om_p = Vector{Vector{Float64}}(undef, Nosc)
    growth_rate_p = Vector{Vector{Float64}}(undef, Nosc)
    use_p = Vector{Vector{Int64}}(undef, Nosc)
    for q = 1:Nosc
        om_p[q] = zeros(Nfreq[q])
        growth_rate_p[q] = zeros(Nfreq[q])
        use_p[q] = zeros(Int64,Nfreq[q]) # By default, don't use any freq's
        p = sortperm(om[q]) # sortperm(growth_rate[q],rev=true)
        om_p[q] .= om[q][p]
        growth_rate_p[q] .= growth_rate[q][p]
    end

    println("Sorted CW freq's:")
    for q = 1:Nosc
      println("Ctrl Hamiltonian # ", q, ", lab frame carrier frequencies: ", rot_freq[q] .+ om_p[q]./(2*pi), " [GHz]")
      println("Ctrl Hamiltonian # ", q, ",                   growth rate: ", growth_rate_p[q], " [1/ns]")
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
                # kmax = argmax(growth_rate_p[q][rge])
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
        # kmax = argmax(growth_rate_p[q][rge])
        use_p[q][k0] = 1
        om_p[q][k0] = om_avg 

        # cull out unused frequencies
        om[q] = zeros(sum(use_p[q]))
        growth_rate[q] = zeros(sum(use_p[q]))
        j = 0
        for k=1:Nfreq[q]
            if use_p[q][k] == 1
                j += 1
                om[q][j] = om_p[q][k]
                growth_rate[q][j] = growth_rate_p[q][k]
            end
        end
        Nfreq[q] = j # correct the number of CW frequencies for oscillator 'q'
    end

    println("\nSorted and culled CW freq's:")
    for q = 1:Nosc
      println("Ctrl Hamiltonian # ", q, ", lab frame carrier frequencies: ", rot_freq[q] .+ om[q]./(2*pi), " [GHz]")
      println("Ctrl Hamiltonian # ", q, ",                   growth rate: ", growth_rate[q], " [1/ns]")
    end

    return Nfreq, om, growth_rate
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
    wmat = wmatsetup(is_ess, it2in, Ne, Ng, wmatScale)

Build the default positive semi-definite weighting matrix W to calculate the 
leakage into higher energy forbidden states
 
# Arguments
- `is_ess::Vector{Bool}`: Vector of the size Ntot=prod(Ne+Ng); true if element is essential, otherwise false
- `it2in::Matrix{Int64}`: Matrix of size (Ntot, Ndim) holding the conversion between 1-d and sub-system numbering of the state vector
- `Ne::Vector{Int64}`: Vector of size Ndim holding the number of essential energy levels for each subsystem
- `Ng::Vector{Int64}`: Vector of size Ndim holding the number of guard energy levels for each subsystem
- `wmatScale::Float64`: Scaling coefficient for the W-matrix
"""
function wmatsetup(is_ess::Vector{Bool}, it2in::Matrix{Int64}, Ne::Vector{Int64}, Ng::Vector{Int64}, wmatScale::Float64 = 1.0)

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

    wmat = wmatScale*coeff * Diagonal(w) # turn vector into diagonal matrix
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
    elseif d == 5
        swap_gate =  Matrix{ComplexF64}(I, 32, 32)
        delta = 15
        for j=2:2:16
            swap_gate[j,j] = 0.0
            swap_gate[j,j+delta] = 1.0
        end
        for j=17:2:31
            swap_gate[j,j] = 0.0
            swap_gate[j,j-delta] = 1.0
        end
    else
        @assert false "Only implemented swap1d gates for d={2,3,4,5}"
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

"""
    params, pcof0, maxAmp = setup_std_model(Ne, Ng, f01, xi, couple_coeff, couple_type, rot_freq, T, D1, gate_final;
     maxctrl_MHz = 10.0, 
     msb_order::Bool = false, 
     Pmin::Int64 = 40, 
     init_amp_frac::Float64 = 0.0, 
     randomize_init_ctrl::Bool = true, 
     rand_seed = 2345, 
     pcofFileName="", 
     zeroCtrlBC::Bool = true, 
     use_eigenbasis::Bool = false, 
     cw_amp_thres = 5e-2, 
     cw_prox_thres = 2e-3, 
     splines_real_imag = true, 
     wmatScale::Float64 = 1.0, 
     use_carrier_waves::Bool = true,
     nTimeIntervals::Int64 = 1)

Setup a Hamiltonian model, parameters for numerical time stepping, a target unitary gate, carrier frequencies, boundary conditions for the control functions, amplitude bounds for the controls, and initialize the control vector for optimization.

# Required arguments
- `Ne::Vector{Int64}`: Number of essential energy levels in each subsystem.
- `Ng::Vector{Int64}`: Number of guard levels in each subssystem.
- `f01::Vector{Float64}`: Transistion frequencies in each subsystem (Ground to first excited state) [GHz].
- `xi::Vector{Float64}`: Anharmonicities in each subsystem [GHz].
- `couple_coeff::Vector{Float64}`: Coupling coefficients between subsystems [GHz], ordered as ``[x_{1,2}, ..., x_{1,n}, x_{2,3}, ..., x_{2,n}, ..., x_{n-1,n}]``.
- `couple_type::Int64`: Type of coupling Hamiltonian. Use 1 for cross-Kerr (``\\hat{a}^\\dagger \\hat{a} \\hat{b}^\\dagger \\hat{b}``) and 2 for a dipole-dipole coupling (``\\hat{a}^\\dagger \\hat{b} + \\hat{a} \\hat{b}^\\dagger``).
- `rot_freq::Vector{Float64}`: Frequency of rotation for each sub-system [GHz].
- `T::Float64`: Duration of the simulation/optimization.
- `D1::Int64`: Number of B-spline coefficients per segment of the control function. Here each segment corresponds to one control Hamiltonian, one carrier frequency, and either the real or the imaginary part of the control function.
- `gate_final::Matrix{ComplexF64}`: Target unitary gate of size N x N, where N=prod(Ne).

# Optional key-word arguments
- `maxctrl_MHz::Float64 = 10.0`: Approximate max control amplitude [MHz].
- `msb_order::Bool = false`: Ordering of subsystems in the state vector. Should be 'false' when using the Quandary backend.
- `Pmin::Int64 = 40`
- `init_amp_frac::Float64 = 0.0`
- `randomize_init_ctrl::Bool = true`
- `rand_seed::Int64 = 2345`
- `pcofFileName = ""`: Read initial control vector from a `jld2` formatted file.
- `zeroCtrlBC::Bool = true`
- `use_eigenbasis::Bool = false`: Experimental option. Avoid using. See source code for details.
- `cw_amp_thres::Float64 = 5e-2`: Only consider elements in the transformed control Hamiltonian that are larger than this threshold.
- `cw_prox_thres::Float64 = 2e-3`: Only consider carrier frequencies that are separated by at least this threshold.
- `splines_real_imag::Bool=true`: B-spline parameterization: `true` (default) use both real and imaginary parts; `false` only control the amplitude and a fixed phase.
- `wmatScale::Float64=1.0`: Scale factor for the leakage term in the objective function.
- `use_carrier_waves::Bool=true`: Set to true (default) to use carrier waves, otherwise only use B-splines to parameterize the control pulses.
- `nTimeIntervals::Int64=1`: Split time domain in this number of intervals.

# Return arguments 
- `params::objparams`: Object specifying the quantum system and the optimization problem.
- `pcof0::Vector{Float64}`: Initial control vector.
- `maxAmp::Vector{Float64}`: Max amplitudes for each segement of the control vector. Here a segment corresponds to a control Hamiltonian
"""
  ################################
  function setup_std_model(Ne::Vector{Int64}, Ng::Vector{Int64}, f01::Vector{Float64}, xi::Vector{Float64}, couple_coeff::Vector{Float64}, couple_type::Int64, rot_freq::Vector{Float64}, T::Float64, D1::Int64, gate_final::Matrix{ComplexF64}; maxctrl_MHz::Float64=10.0, msb_order::Bool = false, Pmin::Int64 = 40, init_amp_frac::Float64=0.0, randomize_init_ctrl::Bool = true, rand_seed::Int64=2345, pcofFileName::String="", zeroCtrlBC::Bool = true, use_eigenbasis::Bool = false, cw_amp_thres::Float64=5e-2, cw_prox_thres::Float64=2e-3, splines_real_imag::Bool=true, wmatScale::Float64=1.0, use_carrier_waves::Bool=true, nTimeIntervals::Int64=1, verbose::Bool=false)
  
    # convert maxctrl_MHz to rad/ns per frequency
    # This is (approximately) the max amplitude of each control function (p & q)
    maxctrl_radns = 2*pi * maxctrl_MHz * 1e-3 
  
    pdim = length(Ne)
    
    # General case
    Hsys, Hsym_ops, Hanti_ops = hamiltonians(Nsys=pdim, Ness=Ne, Nguard=Ng, freq01=f01, anharm=xi, rot_freq=rot_freq, couple_coeff=couple_coeff, couple_type=couple_type, msb_order = msb_order)  

    is_ess, it2in = identify_essential_levels(Ne, Ne+Ng, msb_order)

    # println("Results from identify_essential_levels:")
    # for j = 1:length(is_ess)
    #     println("j: ", j, " it2in[j,:]: ", it2in[j,:], " is_ess[j]: ", is_ess[j])
    # end

    Nosc = length(Hsym_ops)
    @assert(Nosc == pdim) # Nosc must equal pdim
    Ness = prod(Ne)
    Ntot = prod(Ne + Ng)

    if (nTimeIntervals > 1 && Ness != Ntot)
        throw("The lifted approach with intermediate targets is only implemented for square unitaries")
    end

    if use_carrier_waves
        om, growth_rate, Utrans = get_resonances(is_ess, it2in, Ness=Ne, Nguard=Ng, Hsys=Hsys, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, cw_amp_thres=cw_amp_thres, cw_prox_thres=cw_prox_thres, rot_freq=rot_freq, verbose=verbose)
        println("Info: using carrier waves in control pulses")
    else
        Utrans = Matrix{Float64}(I, Ntot, Ntot)

        om = Vector{Vector{Float64}}(undef, Nosc)
        growth_rate = Vector{Vector{Float64}}(undef, Nosc)
        for q=1:Nosc
            om[q] = [0.0]
            growth_rate[q] = [1.0]
        end
        println("Info: NOT using carrier waves in control pulses")
    end

    Nfreq = zeros(Int64,Nosc) # Number of frequencies per control Hamiltonian
    for q=1:Nosc
        Nfreq[q] = length(om[q])
    end

    println("D1 = ", D1, " Ness = ", Ness, " Nosc = ", Nosc, " Nfreq = ", Nfreq)
  
    # Amplitude bounds to be imposed during optimization
    maxAmp = maxctrl_radns * ones(Nosc) # internally scaled by 1/(sqrt(2)*Nfreq[q]) in setup_ipopt() and Quandary

    Nfreq, om, growth_rate = sort_and_cull_carrier_freqs(Nosc, Nfreq, om, growth_rate, rot_freq)
  
    # Set the initial condition: Basis with guard levels
    Ubasis = initial_cond_general(is_ess, Ne, Ng)

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
    
    # Set up the initial control parameter (holding alpha and Winit)
    pcof0 = init_control(amp_frac = init_amp_frac, maxAmp=maxAmp, D1=D1, Nfreq=Nfreq, startFile=pcofFileName, seed=rand_seed, randomize=randomize_init_ctrl, growth_rate=growth_rate, nTimeIntervals=nTimeIntervals, U0=convert(Matrix{ComplexF64}, Ubasis), Utarget=Utarget)

    nCoeff = length(pcof0)

    # Estimate time step based on the number of time steps per shortest period
  
    # Note: calculate_timestep expects maxCoupled to have Nosc elements
    nsteps = calculate_timestep(T, Hsys, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, maxCoupled=maxAmp, Pmin=Pmin)
    println("Starting point: nsteps = ", nsteps, " maxAmp = ", maxAmp, " [rad/ns]")
    
    # create a linear solver object
    linear_solver = Juqbox.lsolver_object(solver=Juqbox.JACOBI_SOLVER, max_iter=100, tol=1e-12, nrhs=prod(Ne))
  
    # create diagonal W-matrix with weights for suppressing leakage
    w_diag_mat = wmatsetup(is_ess, it2in, Ne, Ng, wmatScale)

    # println("norm(wmat1 - wmat2): ", norm(w_diag_mat-w_diag_2))
    # println("w_diag_1: ", diag(w_diag_mat))
    # println("w_diag_2: ", diag(w_diag_2))
    # println("differen: ", diag(w_diag_mat-w_diag_2))

    # Set up parameter struct
    params = Juqbox.objparams(Ne, Ng, T, nsteps, Uinit=convert(Matrix{ComplexF64}, Ubasis), Utarget=Utarget, Cfreq=om, Rfreq=rot_freq, Hconst=Hsys, w_diag_mat=w_diag_mat, nCoeff=nCoeff, D1=D1, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, linear_solver=linear_solver, freq01=f01, self_kerr=xi, couple_coeff=couple_coeff, couple_type=couple_type, msb_order=msb_order, zeroCtrlBC=zeroCtrlBC, nTimeIntervals=nTimeIntervals)


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
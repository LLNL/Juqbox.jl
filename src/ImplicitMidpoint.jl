
function eval_forward_IM(U0::Array{Float64,2}, pcof0::Array{Float64,1}, params::objparams, saveAll:: Bool = false, verbose::Bool = false, order::Int64=2, stages=[])  
    N = params.N  

    Nguard = params.Nguard  
    T = params.T
    nsteps = params.nsteps
    H0 = params.Hconst

    Ntot = N + Nguard
    pcof = pcof0

    # We have 2*Ncoupled ctrl functions
    Ncoupled = params.Ncoupled
    Nunc = params.Nunc
    Nfreq = params.Nfreq
    Nsig = 2*(Ncoupled + Nunc)

    linear_solver = params.linear_solver    

    Psize = size(pcof,1) #must provide separate coefficients for the real and imaginary parts of the control fcn
    if Psize%2 != 0 || Psize < 6
        error("pcof must have an even number of elements >= 6, not ", Psize)
    end
    if params.use_bcarrier
        D1 = div(Psize, Nsig*Nfreq)  # 
        Psize = D1*Nsig*Nfreq # active part of the parameter array
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
        splinepar = bcparams(T, D1, Ncoupled, Nunc, params.Cfreq, pcof)
    else
        splinepar = splinepar(T, D1, Nsig, pcof)   # parameters for B-splines
    end

    # it is up to the user to estimate the number of time steps
    dt ::Float64 = T/nsteps

    gamma, stages = getgamma(order, stages)

    if verbose
        println("Final time: ", T, ", number of time steps: " , nsteps , ", time step: " , dt )
    end

    # the basis for the initial data as a matrix
    #Ident = params.Ident
    Ident = Matrix{Float64}(I, Ntot*2, Ntot*2)

    # Note: Initial condition is supplied as an argument

    #real and imaginary part of initial condition
    #vr   = U0[:,:]
    #vi   = zeros(Float64,Ntot,N)
    #vi05 = zeros(Float64,Ntot,N)
    v_full = zeros(Float64,Ntot*2,N)
    v_full[1:Ntot,:] .= U0[:,:]
    v_full_storage = zeros(Float64,Ntot*2,N)

    if saveAll # Only allocate solution memory for entire timespan if necessary
        usaver = zeros(Float64,Ntot,N,nsteps+1)
        usavei = zeros(Float64,Ntot,N,nsteps+1)
        usaver[:,:,1] = v_full[1:Ntot,:] # the rotation to the lab frame is the identity at t=0
        usavei[:,:,1] = -v_full[Ntot+1:end,:]
    end

    # Preallocate WHAT ABOUT SPARSE FORMAT!
    #K0   = zeros(Float64,Ntot,Ntot)
    #S0   = zeros(Float64,Ntot,Ntot)
    #K05  = zeros(Float64,Ntot,Ntot)
    #S05  = zeros(Float64,Ntot,Ntot)
    #K1   = zeros(Float64,Ntot,Ntot)
    #S1   = zeros(Float64,Ntot,Ntot)
    #κ₁   = zeros(Float64,Ntot,N)
    #κ₂   = zeros(Float64,Ntot,N)
    #ℓ₁   = zeros(Float64,Ntot,N)
    #ℓ₂   = zeros(Float64,Ntot,N)
    #rhs   = zeros(Float64,Ntot,N)

    #K0   = zeros(Float64,Ntot,Ntot)
    #S0   = zeros(Float64,Ntot,Ntot)
    #H0   = zeros(Float64,Ntot*2,Ntot*2)
    K05  = zeros(Float64,Ntot,Ntot)
    S05  = zeros(Float64,Ntot,Ntot)
    H05  = zeros(Float64,Ntot*2,Ntot*2)
    #K1   = zeros(Float64,Ntot,Ntot)
    #S1   = zeros(Float64,Ntot,Ntot)
    #H1   = zeros(Float64,Ntot*2,Ntot*2)
    #lhs   = zeros(Float64,Ntot*2,N)
    rhs   = zeros(Float64,Ntot*2,N)

    #initialize variables for time stepping
    t       ::Float64 = 0.0
    step    :: Int64 = 0

    #KS!(K0, S0, t, params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq) 
    # Forward time stepping loop
    for step in 1:nsteps

        # Störmer-Verlet
        for q in 1:stages
            
            # Update K and S matrices
            #KS!(K0, S0, t, params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq)
            KS_IM!(K05, S05, H05, t + 0.5*dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, Ntot, params.isSymm, splinepar, H0, params.Rfreq)
            #KS!(K1, S1, t + dt*gamma[q], params.Hsym_ops, params.Hanti_ops, params.Hunc_ops, Nunc, params.isSymm, splinepar, H0, params.Rfreq)

            # Take a step forward and accumulate weight matrix integral. Note the √2 multiplier is to account
            # for the midpoint rule in the numerical integration of the imaginary part of the signal.
            # @inbounds t, vr, vi, vi05 = step(t, vr, vi, dt*gamma[q], K0, S0, K05, S05, K1, S1, Ident)
            @inbounds t = step_IM!(t, v_full, v_full_storage, dt*gamma[q], H05, Ident, Ntot, rhs, linear_solver)

            # Keep prior value for next step (FG: will this work for multiple stages?)

        end # Stromer-Verlet
        
        # rotated frame
        if saveAll
            usaver[:,:, step + 1] = v_full[1:Ntot,:] # the rotation to the lab frame is the identity at t=0
            usavei[:,:, step + 1] = -v_full[Ntot+1:end,:]
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

    if saveAll
        return usaver + im*usavei
    else
        return v_full[1:Ntot,:] - im*v_full[Ntot+1:end,:]
    end

end

# FMG: This routine has been modified. This is for the forward evolution with no forcing.
@inline function step_IM!(t::Float64, uv::Array{Float64,N}, uv_storage::Array{Float64,N},  h::Float64,
                      H05::Array{Float64,N},
                      In::Array{Float64,N}, Ntot::Int64, rhs::Array{Float64,N},
                      linear_solver::lsolver_object) where N

    
    ## RHS = (I +0.5h*H05)*uv
 	rhs    .= uv
    mul!(rhs,H05,uv,0.5h,1)

	# uv     .= (In .-  0.5*h.*H05)\rhs
    linear_solver.solve(h,H05,rhs,uv_storage,uv)

	t      = t + h
	return t
end


function KS_IM!(K::Array{Float64,N}, S::Array{Float64,N}, H::Array{Float64,N}, t::Float64, Hsym_ops::Array{MyRealMatrix,1}, Hanti_ops::Array{MyRealMatrix, 1},
             Hunc_ops::Array{MyRealMatrix, 1}, Nunc::Int64, Ntot::Int64, isSymm::BitArray{1}, splinepar::BsplineParams, H0::Array{Float64,N}, Rfreq::Array{Float64,1}) where N

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
    H[1:Ntot,1:Ntot] .= S
    H[Ntot+1:end,1:Ntot] .= K
    H[1:Ntot,Ntot+1:end] .= .-K
    H[Ntot+1:end,Ntot+1:end] .= S
    
end

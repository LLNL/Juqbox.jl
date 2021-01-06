using LinearAlgebra
using JLD2

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

function timesteptest( cfl = 0.1, testcase = 2, order = 2, verbose = false)

    N = 2 # vector dimension	
    IN = Matrix{Float64}(I, N, N)
    
    if testcase == 1 || testcase == 2
	K0 = [0 1; 1 0]
	S0 = [0 0; 0 0]
    elseif testcase == 0 || testcase == 3
	K0 = [0 0; 0 0]
	S0 = [0 1; -1 0]
    end
    
    #Final time
    period = 1
    T = 5pi
    omega = 2*pi/period
    
    lamb = eigvals(K0 .+ S0)
    maxeig = maximum(abs.(lamb))
    
    #time step
    dt = cfl/maxeig  
    nsteps = ceil(Int64,T/dt) 
    dt = T/nsteps
    
    if verbose
        println("Testcase: ", testcase, "  Cfl: ", cfl, " Final time: ", T, " nsteps: ", nsteps )
    end
    
    # Initial conditions
    u = [1.0; 0.0]
    v = [0.0; 0.0]

    t = 0.0
    
    # timefunctions and forcing	
    if testcase == 1
	# timefunc1(t) = 0.5*(sin(0.5*omega*(t)))^2;
	timefunc1(t) = 0.25*(1.0 - cos(omega*t));
	uforce1(t::Float64) = [0.0; 0.0]
	vforce1(t::Float64) = [0.0; 0.0]

	timefunc =timefunc1
	uforce = uforce1
	vforce = vforce1

    elseif testcase == 0
	timefunc0(t::Float64) = 0.25*(1-sin(omega*t))
	uforce0(t::Float64) = [0.0; 0.0]
	vforce0(t::Float64) = [0.0; 0.0]

	timefunc =timefunc0
	uforce = uforce0
	vforce = vforce0
    elseif testcase == 2
	timefunc2(t::Float64) = 4/T^2 *t*(T-t)
	phi12(t::Float64) = 0.25*(t - sin(omega*t)/omega)
	phidot2(t::Float64) = 0.5*(sin(0.5*omega*(t)))^2
	uforce2(t::Float64) = [(timefunc2(t) - phidot2(t))*sin(phi12(t)); 0.0]
	vforce2(t::Float64) = [0.0; -(timefunc2(t) - phidot2(t)) * cos(phi12(t))]

	timefunc =timefunc2
	uforce = uforce2
	vforce = vforce2
    elseif testcase == 3
	timefunc3(t::Float64) =  4/T^2 *t*(T-t)
	phi13(t::Float64) = 0.25*(t - sin(omega*t)/omega)
	phidot3(t::Float64) = 0.5*(sin(0.5*omega*(t)))^2
	uforce3(t::Float64) = [-phidot3(t)*sin(phi13(t)); timefunc3(t)*cos(phi13(t))]
	vforce3(t::Float64) = [-timefunc3(t)*sin(phi13(t)); phidot3(t)*cos(phi13(t))]

	timefunc =timefunc3
	uforce = uforce3
	vforce = vforce3
    end	
    
    K(t::Float64) = timefunc(t)*K0
    S(t::Float64) = timefunc(t)*S0


    #Create time stepper
    gamma, stages = Juqbox.getgamma(order)
    timestepper = Juqbox.svparams(K,S,IN)
    
    #Time integration
    usave = u
    vsave = -v
    tsave = t


    # Stormer-Verlet
    start = time()

    for ii in 1:nsteps
	for jj in 1:stages 
	    t, u, v, v05 =  Juqbox.step(timestepper,t,u,v,dt*gamma[jj],uforce,vforce)
	    # t, u, v =  Juqbox.magnus(timestepper,t,u,v,dt,uforce,vforce)
	end
	usave = [usave u]
	vsave = [vsave -v]
	tsave = [tsave t]
    end
    elapsed = time() - start

    if verbose
        @show(elapsed)
    end

    # Evaluate analytical solution
    if testcase == 1 || testcase == 2 || testcase==3
#        phi = 0.25.*(tsave - 1/omega.*sin.(omega.*tsave))
        phi = 0.25*(t - 1.0/omega*sin(omega*t))
        cg = cos(phi)
        ce = -im*sin(phi)
    elseif testcase == 0
#        phi = 0.25.*( tsave + 1/omega.*(cos.(omega.*tsave) .- 1) );
        phi = 0.25*( t + 1/omega*(cos(omega*t) - 1.0) );
        cg = cos(phi);
        ce = -sin(phi);
    end

    #compute errors @ final time
    # cg_err = sqrt( (usave[1,end]-real(cg) )^2 + (vsave[1,end]-imag(cg) )^2 );
    # ce_err = sqrt( (usave[2,end]-real(ce) )^2 + (vsave[2,end]-imag(ce) )^2 );

    cg_err = sqrt( (u[1]-real(cg) )^2 + (v[1]+imag(cg) )^2 ); # negative imaginary part in v
    ce_err = sqrt( (u[2]-real(ce) )^2 + (v[2]+imag(ce) )^2 );

    if verbose
        println("cg-err = " , cg_err, " ce-err = ", ce_err)
    end

    return t,u,v,dt,cg_err,ce_err
end

function timestep_convergence( errFileName:: String, writeFile:: Bool=false)
    CFL_vec = 10.0.^(-1.0:-0.5:-2.0)

    ntests = 4
    err_mat  = zeros(length(CFL_vec),2,ntests)
    order    = 2

    verbose = false

    for j = 1:ntests
	for i = 1:length(CFL_vec)
	    t,u,v,dt,cg_err,ce_err = timesteptest(CFL_vec[i], j-1, order, verbose)
	    err_mat[i,1,j] = cg_err
	    err_mat[i,2,j] = ce_err
	end
    end

    if writeFile
        @save errFileName err_mat
        println("Saved final errors on file: ", errFileName)
        return true
    else
        # compare results with those on the reference file
        err_mat0 = copy(err_mat)
        @load errFileName err_mat
        # check sizes
        if length(err_mat0) != length(err_mat)
            printf("timestep_convergence: size missmatch with reference solution!")
            return false
        end
        # compare
        max_diff = maximum(abs.(err_mat - err_mat0))
        if verbose
            println("Max abs diff = ", max_diff)
        end
        success = (max_diff <= 1e-13)
        return success        

        # for j = 1:ntests
	#     for i = 1:length(CFL_vec)
        #         println("test #", j, " cfl = ", CFL_vec[i], " diff (cg) = ", abs(err_mat[i,1,j] - err_mat0[i,1,j]), " diff (ce) = ", abs(err_mat[i,2,j] - err_mat0[i,2,j]) )
        #     end
        # end
    end

end

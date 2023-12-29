using Printf
using LinearAlgebra

module OptimConstants
    export N_mnbrak, golden_ratio, N_para, brent_ib, brent_eps, Cr #, N_optim
    N_mnbrak::Int64 = 1e4;                     # number of maximum iterations for mnbrak
    golden_ratio = 0.5 * (1.0 + sqrt(5.0));
    N_para::Int64 = 1e4;                       # number of maximum iterations for linmin
    brent_ib = 1.0e-1;                         # the size of initial step in line search, if nothing specified
    brent_eps = 1e-14;
    Cr = 1. - 1/golden_ratio;
    # N_optim::Int64 = 100;                   # number of maximum iterations for conjugate-gradient. REPLACED by argument maxIter in cgmin()
end

using ..OptimConstants: N_mnbrak, golden_ratio, N_para, brent_ib, brent_eps, Cr

# Input arguments
# %inputObjective: function to evaluate the objective functional
# %pcof0: starting point
# %xi: minimizing direction (-grad)
# %b0: initial guess of step size
# tol: tolerance for bracket size

# Output arguments
# %p_min: local minimum along xi direction
# %J_min: function value at p_min
# %bmin: step size of minimum
function brent(inputObjective::Function, pcof0::Vector{Float64}, xi::Vector{Float64}, params::objparams, b0::Float64 = -1.0, line_tol::Float64 = 1.0e-1)

    if (b0 < 0.0)
        b0 = brent_ib;
    end
    
    #initial bracket
    step = zeros(N_mnbrak+1);
    J_brak = zeros(N_mnbrak+1);
    J_brak[1], _, _ = inputObjective(pcof0, params);
    # println("b: ", 0.0, ", J: ", J_brak[1])
    J1, _, _ = inputObjective(pcof0 + b0 * xi, params);
    # println("b: ", b0, ", J: ", J1)
    while (J1 > J_brak[1])
        b0 = b0 / golden_ratio;
        J1, _, _ = inputObjective(pcof0 + b0 * xi, params);
        # println("b: ", b0, ", J: ", J1)
    end
    # println("smaller J1 found");
    amp = b0; j0 = 0;
    a = -1.0; b = -1.0; c = -1.0;
    fa = -1.0; fb = -1.0; fc = -1.0;
    for j=1:N_mnbrak
        xf = pcof0 + amp * xi;
        step[j+1] = amp;
        J_brak[j+1], _, _ = inputObjective(xf, params);
        # println("b: ", amp, ", J: ", J_brak[j+1])
        
        if (J_brak[j+1] > J_brak[j])
            a = step[j-1]; b = step[j]; c = step[j+1];
            fa = J_brak[j-1]; fb = J_brak[j]; fc = J_brak[j+1];
            j0 = j0 + j;
            break;
        else
            amp = amp * golden_ratio;
        end
    end
    # println("bracket found");
    # println("a: ", a, ", b: ", b, ", c: ", c)
    @assert ((a >= 0) && (b > a) && (c > b))
    
    #parabolic estimation
    for j=1:N_para
        if ( (c-a) < b * line_tol + brent_eps )
            j0 = j0 + j;
            break;
        end
        
        b_new = b - 0.5 * ( (b-a)^2 * (fb-fc) - (b-c)^2 * (fb-fa) ) / ( (b-a) * (fb-fc) - (b-c) * (fb-fa) );
        if ((b_new > c) || (b_new < a) || (abs(log10((c-b) / (b-a))) > 1.0))
            if ( b > 0.5 * (a+c) )
                b_new = b - Cr * (b-a);
            else
                b_new = b + Cr * (c-b);
            end
        end
        
        xb = pcof0 + b_new * xi;
        fbx, _, _ = inputObjective(xb, params);
        # println("b: ", b_new, ", J: ", fbx)
        
        x_arr = zeros(4); x_arr[1] = a; x_arr[4] = c;
        J_arr = zeros(4); J_arr[1] = fa; J_arr[4] = fc;
        if (b_new > b)
            x_arr[2:3] = [b,b_new];
            J_arr[2:3] = [fb,fbx];
        else
            x_arr[2:3] = [b_new,b];
            J_arr[2:3] = [fbx,fb];
        end
        fb, idx = findmin(J_arr);
        fa = J_arr[idx-1]; fc = J_arr[idx+1];
        a = x_arr[idx-1]; b = x_arr[idx]; c = x_arr[idx+1];
    end
    
    p_min = pcof0 + b * xi;
    J_min = fb;
    bmin = b;

    # @printf("Brent: line steps: %d, step size: %.3E, Jmin: %.3E\n", j0, bmin, J_min);
    
    #@printf("step size: %.3E\n", bmin);
    #@printf("Jmin: %.3E\n", J_min);

    return p_min, J_min, bmin
end

# Input arguments
# inputObjective: function to evaluate the objective functional
# inputGradient: function to evaluate the objective gradient
# pcof0: initial guess for design parameters
# cgtol: gradient tolerance for optimization termination

# Output arguments
# %pmin: design variables at local minimum found from CG optimization
# %Jmin: objective function value at pmin
# %J_optim: Optimization history of objective functional (REMOVED)
# %grad_optim: Optimization history of objective gradient magnitude (REMOVED)
# j0: number of iterations executed to find the minimum
function cgmin(inputObjective::Function, inputGradient::Function, pcof0::Vector{Float64}, params::objparams; cgtol::Float64 = 1.0e-8, maxIter::Int64 = 100)
# NOTE: In the Augmented-Lagrangian method the minimization is over the functional J, which may become negative at intermediate iterations due to the lagrange multiplier terms.
# Setting the convergence criteria as J < Jtol is therefore not meaningful 
    pmin = copy(pcof0);

    nMat = params.Ntot^2
    @assert params.nWinit == 2 * params.Ntot^2

    nCoeff = length(pmin)
    grad_f = zeros(nCoeff)

    # Jmin, finalDist, _ = inputObjective(pmin, params);
    # J_optim = zeros(maxIter + 1, 2 + (params.nTimeIntervals-1));
    # J_optim[1, 1] = Jmin
    # J_optim[1, 2] = finalDist
    # J_optim[1, 3:end] = params.nrm2_Cjump[:]
    Jmin, infid, _ = inputGradient(pmin, params, grad_f);

    g = -grad_f; xi = copy(g); h = copy(g);
    gg0 = dot(g, g);

    push!(params.objHist, abs(Jmin))
    push!(params.dualInfidelityHist, sqrt(gg0))
    push!(params.primaryHist, infid)
    push!(params.secondaryHist, 0.0) # no leakage
    if params.constraintType == 0 && params.nTimeIntervals > 1 
        inf_jump = sqrt(maximum(params.nrm2_Cjump)) # norm of state discontinuity
    else
        inf_jump = 0.0
    end
    push!(params.constraintViolationHist, inf_jump)
    # grad_optim = zeros(maxIter + 1, 2 + (params.nTimeIntervals-1));
    # grad_optim[1, 1] = gg0;
    # grad_optim[1, 2] = dot(g[1:params.nAlpha], g[1:params.nAlpha]);
    # for itv = 1:params.nTimeIntervals-1
        # initial conditions from pcof0 (determined by optimization)
        # offc = params.nAlpha + (itv-1) * params.nWinit # for itv = 1 the offset should be nAlpha
        
        # grad_optim[1, 2+itv] = dot(g[offc+1:offc+params.nWinit], g[offc+1:offc+params.nWinit]);
    # end

    b0 = brent_ib / sqrt(gg0);

    j0 = 0;
    for j=1:maxIter
        pmin, Jmin, newStep = brent(inputObjective, pmin, xi, params, b0);
        b0 = newStep;

        # evaluate one more time to obtain sub-objective functional.
        # J_optim[j+1, 1], J_optim[j+1, 2], _ = inputObjective(pmin, params)
        # J_optim[j+1, 3:end] = params.nrm2_Cjump[:]

        # xi .= 0.0;
        Jmin, infid, _ = inputGradient(pmin, params, xi);
        gg = dot(g, g); # norm2 of previous gradient
        
        dgg = dot(xi + g, xi);
        dgg1 = dot(xi, xi); # norm2 of current gradient

        # @printf("Norm(grad)^2: %.3E\n", dgg1);

        push!(params.objHist, abs(Jmin))
        push!(params.dualInfidelityHist, sqrt(dgg1))
        push!(params.primaryHist, infid)
        push!(params.secondaryHist, 0.0) # no leakage
        if params.constraintType == 0 && params.nTimeIntervals > 1 
            inf_jump = sqrt(maximum(params.nrm2_Cjump)) # norm of state discontinuity
        else
            inf_jump = 0.0
        end
        push!(params.constraintViolationHist, inf_jump)
        # grad_optim[j+1, 1] = dgg1;
        # grad_optim[j+1, 2] = dot(xi[1:params.nAlpha], xi[1:params.nAlpha]);
        # for itv = 1:params.nTimeIntervals-1
            # initial conditions from pcof0 (determined by optimization)
            # offc = params.nAlpha + (itv-1) * params.nWinit # for itv = 1 the offset should be nAlpha
            
            # grad_optim[j+1, 2+itv] = dot(xi[offc+1:offc+params.nWinit], xi[offc+1:offc+params.nWinit]);
        # end
        gg1 = dot(g, xi);
        nu = abs(gg1 / dgg1);
        
        if dgg1 < cgtol # AP: could add other convergence criteria here, e.g., infid<threshold
            j0 = j;
            println("CG found local minima with norm^2(grad) = ", dgg1, " < ", cgtol)
            break;
        end
            
        gamma = dgg / gg;
        gamma_FR = dgg1 / gg;
        if (j >= 2)
            if (gamma < -gamma_FR)
                gamma = -gamma_FR;
            elseif (gamma > gamma_FR)
                gamma = gamma_FR;
            end
        end
        
        g = -xi; xi = g + gamma * h; h = copy(xi);
        j0 = j;
    end
    println("CG-min finished in: ", j0, " iterations, out of max: ", maxIter)
    # return pmin, Jmin, J_optim, grad_optim, stepsize, j0
    return pmin, Jmin, j0
end
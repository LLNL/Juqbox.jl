module OptimUtils
using Printf
using LinearAlgebra
export brent, cgmin

N_mnbrak::Int64 = 1e4;                     # number of maximum iterations for mnbrak
golden_ratio = 0.5*(1+sqrt(5));
N_para::Int64 = 1e4;                       # number of maximum iterations for linmin
brent_ib = 1.0e-1;                          # the size of initial step in line search, if nothing specified
brent_eps = 1e-14;
Cr = 1. - 1/golden_ratio;

N_optim::Int64 = 1e4;                          # number of maximum iterations for conjugate-gradient

# Input arguments
# %inputObjective: function to evaluate the objective functional
# %p: starting point
# %xi: minimizing direction (-grad)
# %b0: initial guess of step size
# tol: tolerance for bracket size

# Output arguments
# %p_min: local minimum along xi direction
# %J_min: function value at p_min
# %bmin: step size of minimum
function brent(inputObjective::Function, p::Vector{Float64}, xi::Vector{Float64}, b0::Float64 = -1.0, line_tol::Float64 = 1.0e-1)

    if (b0 < 0.0)
        b0 = brent_ib;
    end
    
    #initial bracket
    step = zeros(N_mnbrak+1);
    J_brak = zeros(N_mnbrak+1);
    J_brak[1], _, _ = inputObjective(p);
    # println("b: ", 0.0, ", J: ", J_brak[1])
    J1, _, _ = inputObjective(p + b0 * xi);
    # println("b: ", b0, ", J: ", J1)
    while (J1 > J_brak[1])
        b0 = b0 / golden_ratio;
        J1, _, _ = inputObjective(p + b0 * xi);
        # println("b: ", b0, ", J: ", J1)
    end
    # println("smaller J1 found");
    amp = b0; j0 = 0;
    a = -1.0; b = -1.0; c = -1.0;
    fa = -1.0; fb = -1.0; fc = -1.0;
    for j=1:N_mnbrak
        xf = p + amp * xi;
        step[j+1] = amp;
        J_brak[j+1], _, _ = inputObjective(xf);
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
        
        xb = p + b_new * xi;
        fbx, _, _ = inputObjective(xb);
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
    
    p_min = p + b * xi;
    J_min = fb;
    bmin = b;

    @printf("line steps: %d\n", j0);
    @printf("step size: %.3E\n", bmin);
    @printf("Jmin: %.3E\n", J_min);

    return p_min, J_min, bmin
end

# Input arguments
# %inputObjective: function to evaluate the objective functional
# %inputObjective: function to evaluate the objective gradient
# %p_init: initial guess for design parameters
# cgtol: gradient tolerance for optimization termination

# Output arguments
# %pmin: local minimum found from CG optimization
# %Jmin: function value at pmin
# %J_optim: Optimization history of objective functional
# %grad_optim: Optimization history of objective gradient magnitude
# j0: number of iterations executed to find the minimum
function cgmin(inputObjective::Function, inputGradient::Function, p_init::Vector{Float64}, cgtol::Float64 = 1.0e-8, Jtol::Float64 = 1.0e100)
    pmin = copy(p_init);

    nCoeff = length(pmin)
    grad_f = zeros(nCoeff)

    Jmin, finalDist, nrm2_Cjump = inputObjective(pmin);
    J_optim = zeros(N_optim+1, 2+size(nrm2_Cjump)[1]);
    J_optim[1, 1] = Jmin
    J_optim[1, 2] = finalDist
    J_optim[1, 3:end] = nrm2_Cjump
    inputGradient(pmin, grad_f);

    g = -grad_f; xi = copy(g); h = copy(g);
    gg0 = dot(g, g);
    grad_optim = zeros(N_optim+1);
    grad_optim[1] = gg0;

    stepsize = zeros(N_optim);

    b0 = brent_ib / sqrt(gg0);

    j0 = 0;
    for j=1:N_optim
        pmin, Jmin, stepsize[j] = brent(inputObjective, pmin, xi, b0);
        b0 = stepsize[j];

        # evaluate one more time to obtain sub-objective functional.
        J_optim[j+1, 1], J_optim[j+1, 2], J_optim[j+1, 3:end] = inputObjective(pmin)
        
        xi .= 0.0;
        inputGradient(pmin, xi);
        gg = dot(g, g);
        
        dgg = dot(xi + g, xi);
        dgg1 = dot(xi, xi);
        @printf("dgg1: %.3E\n", dgg1);
        grad_optim[j+1] = dgg1;
        gg1 = dot(g, xi);
        nu = abs(gg1 / dgg1);
        
        if ((dgg1 < cgtol) && (Jmin < Jtol))
            j0 = j;
            println("CG minimization finished.")
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

    return pmin, Jmin, J_optim, grad_optim, stepsize, j0
end

end
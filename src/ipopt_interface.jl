# mutable struct opt_params

#     nvar ::Int64    # number of variables in the optimization problem
#     neqconst ::Int64  # number of equality constraints
#     nieqconst ::Int64 # number of inequality constraints
#     maxIter ::Int64 # maximum number of iterations for optimizer

#     nele_jac ::Int64 # number of elements in constraint Jacobian
#     nele_hess ::Int64 # number of elements in the constrint Hessian
#     jacob_approx ::String # Jacobian computation (either exact or finite-difference-values)
#     hessian_approx ::String # Hessian computation (either exact or limited-memory)

#     x_L ::Array{Float64,1} # lower bound for design variables
#     x_U ::Array{Float64,1} # upper bound for design variables

#     g_L ::Array{Float64,1} # lower bound for constraints
#     g_U ::Array{Float64,1} # upper bound for constraints
    
# end #opt_params struct




function eval_f_g_grad!(pcof::Vector{Float64},params:: Juqbox.objparams, wa,
                       nodes::AbstractArray=[0.0], weights::AbstractArray=[1.0], compute_adjoint::Bool=true)

    params.last_pcof .= pcof
    params.last_infidelity = 0.0
    params.last_leak = 0.0
    params.last_infidelity_grad .= 0.0
    params.last_leak_grad .= 0.0


    # Loop over specified nodes and compute risk-neutral objective value. Default is usual optimization.
    nquad = length(nodes)

    # H0_old = copy(params.Hconst)
    for i = 1:nquad 
        ep = nodes[i]

        for j = 2:size(params.Hconst,2)
            # params.Hconst[j,j] += H0_old[j,j] + 0.01*ep*(10.0^(j-2))
            params.Hconst[j,j] += 0.01*ep*(10.0^(j-2))
        end

        if compute_adjoint
            _, totalgrad, primaryobjf, secondaryobjf, traceInfidelity, infidelitygrad, leakgrad = Juqbox.traceobjgrad(pcof,params,wa,false,compute_adjoint)
            for j in 1:length(infidelitygrad)
                params.last_infidelity_grad[j] += infidelitygrad[j]*weights[i]
            end
            for j in 1:length(leakgrad)            
                params.last_leak_grad[j] += leakgrad[j]*weights[i]
            end
        else
            _, primaryobjf, secondaryobjf = Juqbox.traceobjgrad(pcof,params,wa,false,compute_adjoint)
        end

        params.last_infidelity += primaryobjf[1]*weights[i]
        params.last_leak       += secondaryobjf[1]*weights[i]

        # Reset 
        for j = 2:size(params.Hconst,2)
            params.Hconst[j,j] -= 0.01*ep*(10.0^(j-2))
        end
    end

    params.lastTraceInfidelity = params.last_infidelity
    params.lastLeakIntegral = params.last_leak

end


# setup callback functions for Ipopt

# function eval_f_par(pcof::Vector{Float64},x_new:: Bool, params:: Juqbox.objparams, wa::Working_Arrays,
#                     nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])
function eval_f_par(pcof::Vector{Float64}, params:: Juqbox.objparams, wa,
    nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])


    #Return last stored
    # if x_new 
    pnorm =norm(pcof .- params.last_pcof)     
    if pnorm > 1.0e-15
        compute_adjoint = true
        eval_f_g_grad!(pcof,params,wa,nodes,weights,compute_adjoint)        
    end

    if params.objFuncType == 1
        f = params.last_infidelity + params.last_leak
    else
        f = params.last_infidelity
    end
    
    # Add in Tikhonov regularization
    tikhonovpenalty = Juqbox.tikhonov_pen(pcof, params)

    return f .+ tikhonovpenalty
  end


# function eval_g_par(pcof::Vector{Float64},x_new:: Bool,g::Vector{Float64},params:: Juqbox.objparams, wa::Working_Arrays,
#                     nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])
function eval_g_par(pcof::Vector{Float64},g::Vector{Float64},params:: Juqbox.objparams, wa,
    nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])

    #Return last stored
    # if x_new
    pnorm =norm(pcof .- params.last_pcof) 
    if pnorm > 1.0e-15
        compute_adjoint = true
        eval_f_g_grad!(pcof,params,wa,nodes,weights,compute_adjoint)
    end
    
    g[1] = params.last_leak

    return g[1]
end



# function eval_grad_f_par(pcof::Vector{Float64},x_new:: Bool, grad_f::Vector{Float64}, params:: Juqbox.objparams, wa::Working_Arrays,
#                         nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])
function eval_grad_f_par(pcof::Vector{Float64}, grad_f::Vector{Float64}, params:: Juqbox.objparams, wa,
    nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])
    
    #Return last stored
    # if x_new 
    pnorm =norm(pcof .- params.last_pcof) 
    if pnorm > 1.0e-15
        compute_adjoint = true
        eval_f_g_grad!(pcof,params,wa,nodes,weights,compute_adjoint)
    end

    #When params.objFuncType == 1, this stores the total grad
    grad_f .= params.last_infidelity_grad
    
    # Add in Tikhonov regularization gradient term
    wa.gr .= 0.0
    Juqbox.tikhonov_grad!(pcof, params, wa.gr)  
    axpy!(1.0,wa.gr,grad_f)


    # Save intermediate parameter vectors
    if params.save_pcof_hist
        push!(params.pcof_hist, copy(pcof)) #pcof_hist is and Array of Vector{Float64}
    end
end


# function eval_jac_g_par(pcof::Vector{Float64},x_new:: Bool, rows::Vector{Int32}, cols::Vector{Int32}, jac_g::Union{Nothing,Vector{Float64}},
#                         params:: Juqbox.objparams, wa::Working_Arrays, nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])
function eval_jac_g_par(pcof::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, jac_g::Union{Nothing,Vector{Float64}},
    params:: Juqbox.objparams, wa, nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])

    if jac_g === nothing 
        if length(rows)>0
            nvar = length(pcof)
            for i in 1:nvar
                rows[i] = 1;    cols[i] = i
            end
        end
    else


        #Return last stored
        # if x_new 
        pnorm =norm(pcof .- params.last_pcof) 
        if pnorm > 1.0e-15
            compute_adjoint = true    
            eval_f_g_grad!(pcof,params,wa,nodes,weights,compute_adjoint)    
            return
        end

        jac_g .= params.last_leak_grad 
    end        
    
    return               
end 


function eval_jac_g_empty(
    x::Vector{Float64},         # Current solution
    x_new:: Bool,               # If new vector x
    rows::Vector{Int32},        # Sparsity structure - row indices
    cols::Vector{Int32},        # Sparsity structure - column indices
    values::Union{Nothing,Vector{Float64}})    # The values of the Hessian
    return nothing
end




function eval_h(
    x::Vector{Float64},
    # x_new:: Bool,
    rows::Vector{Int32},
    cols::Vector{Int32},
    obj_factor::Float64,
    lambda::Vector{Float64},
    values::Union{Nothing,Vector{Float64}})

    if mode == :Structure
        # rows[...] = ...
        # ...
        # cols[...] = ...
    else
        # values[...] = ...
    end
end

function intermediate_par(
    alg_mod::Union{Int32,Int64},
    iter_count::Union{Int32,Int64},
    obj_value::Float64,
    inf_pr::Float64, inf_du::Float64,
    mu::Float64, d_norm::Float64,
    regularization_size::Float64,
    alpha_du::Float64, alpha_pr::Float64,
    ls_trials::Union{Int32,Int64},
    params:: Juqbox.objparams)
  # ...
    if params.saveConvHist 
        push!(params.objHist, obj_value)
        push!(params.dualInfidelityHist, inf_du)
        push!(params.primaryHist, params.lastTraceInfidelity)
        push!(params.secondaryHist,  params.lastLeakIntegral)
    end
    if obj_value < params.objThreshold
        println("Stopping because objective value = ", obj_value,
                " < threshold = ", params.objThreshold)        
        return false
    elseif params.lastTraceInfidelity < params.traceInfidelityThreshold
        println("Stopping because trace infidelity = ", params.lastTraceInfidelity,
                " < threshold = ", params.traceInfidelityThreshold)        
        return false
    else
        return true  # Keep going
    end
end

"""
    prob = setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff; maxIter=50, 
                            lbfgsMax=10, startFromScratch=true, ipTol=1e-5, acceptTol=1e-5, acceptIter=15,
                            nodes=[0.0], weights=[1.0])

Setup structure containing callback functions and convergence criteria for 
optimization via IPOPT. Note the last two inputs, `nodes', and 
`weights', are to be used when performing a simple risk-neutral optimization
where the fundamental frequency is random.

# Arguments
- `params:: objparams`: Struct with problem definition
- `wa::Working_Arrays`: Struct containing preallocated working arrays
- `nCoeff:: Int64`: Number of parameters in optimization
- `minCoeff:: Array{Float64, 1}`: Minimum allowable value for each parameter
- `maxCoeff:: Array{Float64, 1}`: Maximum allowable value for each parameter
- `maxIter:: Int64`: Maximum number of iterations to be taken by optimizer (keyword arg)
- `lbfgsMax:: Int64`: Maximum number of past iterates for Hessian approximation by L-BFGS (keyword arg)
- `startFromScratch:: Bool`: Specify whether the optimization is starting from file or not (keyword arg)
- `ipTol:: Float64`: Desired convergence tolerance (relative) (keyword arg)
- `acceptTol:: Float64`: Acceptable convergence tolerance (relative) (keyword arg)
- `acceptIter:: Int64`: Number of acceptable iterates before triggering termination (keyword arg)
- `nodes:: Array{Float64, 1}`: Risk-neutral opt: User specified quadrature nodes on the interval [-ϵ,ϵ] for some ϵ (keyword arg)
- `weights:: Array{Float64, 1}`: Risk-neutral opt: User specified quadrature weights on the interval [-ϵ,ϵ] for some ϵ (keyword arg)
"""
function setup_ipopt_problem(params:: Juqbox.objparams, wa, nCoeff:: Int64, minCoeff:: Array{Float64, 1}, maxCoeff:: Array{Float64, 1};
                             maxIter:: Int64=50, lbfgsMax:: Int64=10, 
                             startFromScratch:: Bool=true, ipTol:: Float64=1e-5, 
                             acceptTol:: Float64=1e-5, acceptIter:: Int64=15,
                             nodes::AbstractArray=[0.0], 
                             weights::AbstractArray=[1.0],
                             jacob_approx::String="exact")


    #Initialize the last fidelity and leak terms and gradients
    params.last_pcof = 1e9.*rand(nCoeff)
    params.last_infidelity_grad = 1e9.*rand(nCoeff)
    if params.objFuncType != 1 #Only allcoate for inequality opt...
        params.last_leak_grad = 1e9.*rand(nCoeff)        
    end

    # callback functions need access to the params object
    eval_f(pcof) = eval_f_par(pcof,params, wa, nodes, weights)
    eval_grad_f(pcof, grad_f) = eval_grad_f_par(pcof,grad_f, params, wa, nodes, weights)

    #Comment out to use xnew with later version of ipopt
    # eval_f(pcof,x_new) = eval_f_par(pcof, x_new,params, wa, nodes, weights)
    # eval_grad_f(pcof,x_new, grad_f) = eval_grad_f_par(pcof,x_new, grad_f, params, wa, nodes, weights)
    intermediate(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                 d_norm, regularization_size, alpha_du, alpha_pr, ls_trials) =
                     intermediate_par(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                                      d_norm, regularization_size, alpha_du, alpha_pr, ls_trials, params)

    # setup the Ipopt data structure
    if params.objFuncType == 3
        # treat the leakage as an inequality constraint
        nconst = 1 # One constraint
        nEleJac = nCoeff
        nEleHess = 0
        g_L = -2e19.*ones(nconst) # no lower bound needed because the leakage is always non-negative
        g_U = params.leak_ubound.*ones(nconst)
    else
        nconst = 0
        nEleJac = 0
        nEleHess = 0
        g_L = zeros(0);
        g_U = zeros(0);
    end

    #Create alias even if not used
    eval_g(pcof,g) = eval_g_par(pcof,g,params,wa,nodes,weights)
    eval_jac_g(pcof,rows,cols,jac_g) = eval_jac_g_par(pcof,rows,cols,jac_g,params,wa,nodes,weights)
    # eval_g(pcof,x_new,g) = eval_g_par(pcof,x_new,g,params,wa,nodes,weights)
    # eval_jac_g(pcof,x_new,rows,cols,jac_g) = eval_jac_g_par(pcof,x_new,rows,cols,jac_g,params,wa,nodes,weights)


    # tmp
    #    println("setup_ipopt_problem: nCoeff = ", nCoeff, " length(minCoeff) = ", length(minCoeff))
    if @isdefined createProblem
        prob = createProblem( nCoeff, minCoeff, maxCoeff, nconst, g_L, g_U, nEleJac, nEleHess, eval_f, eval_g, eval_grad_f, eval_jac_g);
    else
        # prob = CreateIpoptProblem( nCoeff, minCoeff, maxCoeff, nconst, g_L, g_U, nEleJac, nEleHess, eval_f, eval_g, eval_grad_f, eval_jac_g,eval_h,expose_xnew=true);
        prob = CreateIpoptProblem( nCoeff, minCoeff, maxCoeff, nconst, g_L, g_U, nEleJac, nEleHess, eval_f, eval_g, eval_grad_f, eval_jac_g,eval_h);
    end

    if @isdefined addOption
        addOption( prob, "hessian_approximation", "limited-memory");
        addOption( prob, "limited_memory_max_history", lbfgsMax);
        addOption( prob, "max_iter", maxIter);
        addOption( prob, "tol", ipTol);
        addOption( prob, "acceptable_tol", acceptTol);
        addOption( prob, "acceptable_iter", acceptIter);
        addOption( prob, "jacobian_approximation", jacob_approx);
        #addOption( prob, "derivative_test", "first-order");
        # addOption( prob, "derivative_test_tol", 0.0001);
        
        if !startFromScratch # enable warm start of Ipopt
            addOption( prob, "warm_start_init_point", "yes")
            # addOption( prob, "mu_init", 1e-6) # not sure how to set this parameter
            # addOption( prob, "nlp_scaling_method", "none") # what about scaling?
            #
            # the following settings prevent the initial parameters to be pushed away from their bounds
            addOption( prob, "warm_start_bound_push", 1e-16)
            addOption( prob, "warm_start_bound_frac", 1e-16)
            addOption( prob, "warm_start_slack_bound_frac", 1e-16)
            addOption( prob, "warm_start_slack_bound_push", 1e-16)

            if !params.quiet
                println("Ipopt: Enabling warm start option")
            end
        end
    else
        AddIpoptStrOption( prob, "hessian_approximation", "limited-memory");
        AddIpoptIntOption( prob, "limited_memory_max_history", lbfgsMax);
        AddIpoptIntOption( prob, "max_iter", maxIter);
        AddIpoptNumOption( prob, "tol", ipTol);
        AddIpoptNumOption( prob, "acceptable_tol", acceptTol);
        AddIpoptIntOption( prob, "acceptable_iter", acceptIter);
        AddIpoptStrOption( prob, "jacobian_approximation", jacob_approx);
        AddIpoptStrOption( prob, "derivative_test", "first-order");
        # AddIpoptNumOption( prob, "derivative_test_tol", 1.0e-4);
        # AddIpoptNumOption( prob, "derivative_test_perturbation", 1.0e-8);
        
        

        if !startFromScratch # enable warm start of Ipopt
            AddIpoptStrOption( prob, "warm_start_init_point", "yes")
            # AddIpoptNumOption( prob, "mu_init", 1e-6) # not sure how to set this parameter
            # AddIpoptStrOption( prob, "nlp_scaling_method", "none") # what about scaling?
            #
            # the following settings prevent the initial parameters to be pushed away from their bounds
            AddIpoptNumOption( prob, "warm_start_bound_push", 1e-16)
            AddIpoptNumOption( prob, "warm_start_bound_frac", 1e-16)
            AddIpoptNumOption( prob, "warm_start_slack_bound_frac", 1e-16)
            AddIpoptNumOption( prob, "warm_start_slack_bound_push", 1e-16)

            if !params.quiet
                println("Ipopt: Enabling warm start option")
            end
        end        
    end

    # intermediate callback function
    if @isdefined setIntermediateCallback
        setIntermediateCallback(prob, intermediate)
    else 
        SetIntermediateCallback(prob,intermediate)
    end

# output some of the settings
    if !params.quiet
        println("Ipopt parameters: max # iterations = ", maxIter)
        println("Ipopt parameters: max history L-BFGS = ", lbfgsMax)
        println("Ipopt parameters: tol = ", ipTol)
        println("Ipopt parameters: atol = ", acceptTol)
        println("Ipopt parameters: accept # iter = ", acceptIter)
    end
    
    return prob
end

"""
    pcof = run_optimizer(prob, pcof0 [, baseName:: String=""])

Call IPOPT to  optimizize the control functions.

# Arguments
- `prob:: IpoptProblem`: Struct containing Ipopt parameters callback functions
- `pcof0:: Array{Float64, 1}`: Initial guess for the parameter values
- `baseName:: String`: Name of file for saving the optimized parameters; extension ".jld2" is appended
"""
function run_optimizer(prob:: IpoptProblem, pcof0:: Array{Float64, 1}, baseName:: String="")
    # takes at most max_iter >= 0 iterations. Set with addOption(prob, "max_iter", nIter)

    # initial guess for IPOPT; make a copy of pcof0 to avoid overwriting it
    prob.x = copy(pcof0);

    # Ipopt solver
    println("*** Starting the optimization ***")
    if @isdefined solveProblem
        @time solveProblem(prob);
    else 
        @time IpoptSolve(prob);
    end
    pcof = prob.x;

    #save the b-spline coeffs on a JLD2 file
    if length(baseName)>0
        fileName = baseName * ".jld2"
        save_pcof(fileName, pcof)
        println("Saved B-spline parameters on binary jld2-file '", fileName, "'");
    end

    return pcof

end # run_optimizer

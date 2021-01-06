# setup callback functions for Ipopt
function eval_f_par(pcof::Vector{Float64}, params:: Juqbox.objparams, wa::Working_Arrays)
    #@show(pcof)
    f =Juqbox.traceobjgrad(pcof,params,wa,false,false)
    # @show(f)
    return f[1]
  end

function eval_g(pcof)
    return 0.0
end

function eval_grad_f_par(pcof::Vector{Float64}, grad_f::Vector{Float64}, params:: Juqbox.objparams, wa::Working_Arrays)
    objf, Gtemp, primaryobjf, secondaryobjf, traceinfid = Juqbox.traceobjgrad(pcof,params,wa,false, true)
        
    Gtemp = vcat(Gtemp...) 
    for i in 1:length(Gtemp)
        grad_f[i] = Gtemp[i]
    end
    # remember the value of the primary obj func (for termination in intermediate_par)
    params.lastTraceInfidelity = traceinfid
    params.lastLeakIntegral = secondaryobjf

end

function eval_jac_g(
    x::Vector{Float64},         # Current solution
    mode,                       # Either :Structure or :Values
    rows::Vector{Int32},        # Sparsity structure - row indices
    cols::Vector{Int32},        # Sparsity structure - column indices
    values::Vector{Float64})    # The values of the Hessian

    if mode == :Structure
        # rows[...] = ...
        # ...
        # cols[...] = ...
    else
        # values[...] = ...
    end
end

function intermediate_par(
    alg_mod::Int,
    iter_count::Int,
    obj_value::Float64,
    inf_pr::Float64, inf_du::Float64,
    mu::Float64, d_norm::Float64,
    regularization_size::Float64,
    alpha_du::Float64, alpha_pr::Float64,
    ls_trials::Int,
    params:: Juqbox.objparams)
  # ...
    if params.saveConvHist 
        push!(params.objHist, obj_value)
        push!(params.dualInfidelityHist, inf_du)
        push!(params.primaryHist, params.lastTraceInfidelity)
        push!(params.secondaryHist,  params.lastLeakIntegral)
    end
    if params.lastTraceInfidelity < params.traceInfidelityThreshold
        println("Stopping because trace infidelity = ", params.lastTraceInfidelity,
                " < threshold = ", params.traceInfidelityThreshold)
        return false
    else
        return true  # Keep going
    end
end


function setup_ipopt_problem(params:: Juqbox.objparams, wa::Working_Arrays, nCoeff:: Int64, minCoeff:: Array{Float64, 1}, maxCoeff:: Array{Float64, 1}, maxIter:: Int64=50, lbfgsMax:: Int64=10, startFromScratch:: Bool=true, ipTol:: Float64=1e-5, acceptTol:: Float64=1e-5, acceptIter:: Int64=15)
    # callback functions need access to the params object
    eval_f(pcof) = eval_f_par(pcof, params, wa)
    eval_grad_f(pcof, grad_f) = eval_grad_f_par(pcof, grad_f, params, wa)
    intermediate(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                 d_norm, regularization_size, alpha_du, alpha_pr, ls_trials) =
                     intermediate_par(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                                      d_norm, regularization_size, alpha_du, alpha_pr, ls_trials, params)
    # setup the Ipopt data structure
    mNLconstraints = 0;
    nEleJac = 0;
    nEleHess = 0;
    dum0 = zeros(0);
    dum1 = zeros(0);
    # tmp
    #    println("setup_ipopt_problem: nCoeff = ", nCoeff, " length(minCoeff) = ", length(minCoeff))
    prob = createProblem( nCoeff, minCoeff, maxCoeff, mNLconstraints, dum0, dum1, nEleJac, nEleHess, eval_f, eval_g, eval_grad_f, eval_jac_g);
    addOption( prob, "hessian_approximation", "limited-memory");
    addOption( prob, "limited_memory_max_history", lbfgsMax);
    addOption( prob, "max_iter", maxIter);
    addOption( prob, "tol", ipTol);
    addOption( prob, "acceptable_tol", acceptTol);
    addOption( prob, "acceptable_iter", acceptIter);
    if !startFromScratch # enable warm start of Ipopt
        addOption( prob, "warm_start_init_point", "yes")
        #        addOption( prob, "mu_init", 1e-6) # not sure how to set this parameter
        #        addOption( prob, "nlp_scaling_method", "none") # what about scaling?
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

    # intermediate callback function
    setIntermediateCallback(prob, intermediate)

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

function run_optimizer(prob:: IpoptProblem, pcof0:: Array{Float64, 1}, fileName:: String="")
    # takes at most max_iter >= 0 iterations. Set with addOption(prob, "max_iter", nIter)

    # initial guess for IPOPT
    prob.x = pcof0;

    # Ipopt solver
    println("*** Starting the optimization ***")
    @time solveProblem(prob);
    pcof = prob.x;

    #save the b-spline coeffs on a JLD2 file
    if length(fileName)>0
        save(fileName, "pcof", pcof)
        println("Saved B-spline parameters on binary jld2-file '", fileName, "'");
    end

    return pcof

end # run_optimizer

# mutable struct opt_params

#     nvar ::Int64    # number of variables in the optimization problem
#     neqconst ::Int64  # number of equality constraints
#     nieqconst ::Int64 # number of inequality constraints
#     maxIter ::Int64 # maximum number of iterations for optimizer

#     nele_jac ::Int64 # number of elements in constraint Jacobian
#     nele_hess ::Int64 # number of elements in the constrint Hessian
#     hessian_approx ::String # Hessian computation (either exact or limited-memory)

#     x_L ::Array{Float64,1} # lower bound for design variables
#     x_U ::Array{Float64,1} # upper bound for design variables

#     g_L ::Array{Float64,1} # lower bound for constraints
#     g_U ::Array{Float64,1} # upper bound for constraints
    
# end #opt_params struct




function eval_f_g_grad!(pcof::Vector{Float64}, params:: Juqbox.objparams, nodes::AbstractArray=[0.0], weights::AbstractArray=[1.0], compute_adjoint::Bool=true)

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
            _, totalgrad, primaryobjf, secondaryobjf, traceInfidelity, infidelitygrad, leakgrad = Juqbox.traceobjgrad(pcof, params, false, compute_adjoint)
            for j in 1:length(infidelitygrad)
                params.last_infidelity_grad[j] += infidelitygrad[j]*weights[i]
            end
            for j in 1:length(leakgrad)            
                params.last_leak_grad[j] += leakgrad[j]*weights[i]
            end
        else
            _, primaryobjf, secondaryobjf = Juqbox.traceobjgrad(pcof, params, false, compute_adjoint)
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

# function eval_f_par(pcof::Vector{Float64},x_new:: Bool, params:: Juqbox.objparams,
#                     nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])
function eval_f_par(pcof::Vector{Float64}, params:: Juqbox.objparams,
    nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])

    #Return last stored
    # if x_new 
    pnorm =norm(pcof .- params.last_pcof)     
    if pnorm > 1.0e-15
        compute_adjoint = true # could it be false?
        eval_f_g_grad!(pcof, params, nodes, weights, compute_adjoint)        
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


# function eval_g_par(pcof::Vector{Float64},x_new:: Bool,g::Vector{Float64},params:: Juqbox.objparams,
#                     nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])
function eval_g_par(pcof::Vector{Float64}, g::Vector{Float64}, params:: Juqbox.objparams,
    nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])

    #Return last stored
    # if x_new
    pnorm =norm(pcof .- params.last_pcof) 
    if pnorm > 1.0e-15
        compute_adjoint = true # could it be false?
        eval_f_g_grad!(pcof, params, nodes, weights, compute_adjoint)
    end
    
    g[1] = params.last_leak

    return g[1]
end


# function eval_grad_f_par(pcof::Vector{Float64},x_new:: Bool, grad_f::Vector{Float64}, params:: Juqbox.objparams, nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])
function eval_grad_f_par(pcof::Vector{Float64}, grad_f::Vector{Float64}, params:: Juqbox.objparams,
    nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])
    
    wa = params.wa

    # Return last stored
    # if x_new 
    pnorm =norm(pcof .- params.last_pcof) 
    if pnorm > 1.0e-15 # check if the gradient was calculated
        compute_adjoint = true
        eval_f_g_grad!(pcof, params, nodes, weights, compute_adjoint)
    end

    # When params.objFuncType == 1, this stores the total grad
    grad_f .= params.last_infidelity_grad

    # Add in Tikhonov regularization gradient term
    wa.gr .= 0.0
    Juqbox.tikhonov_grad!(pcof, params, wa.gr)  
    axpy!(1.0, wa.gr, grad_f)

    # Save intermediate parameter vectors
    if params.save_pcof_hist
        push!(params.pcof_hist, copy(pcof)) #pcof_hist is an Array of Vector{Float64}
    end
end

# function eval_jac_g_par(pcof::Vector{Float64},x_new:: Bool, rows::Vector{Int32}, cols::Vector{Int32}, jac_g::Union{Nothing,Vector{Float64}},
#                         params:: Juqbox.objparams, nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])
function eval_jac_g_par(pcof::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, jac_g::Union{Nothing,Vector{Float64}}, params:: Juqbox.objparams, nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])

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
            eval_f_g_grad!(pcof, params, nodes, weights, compute_adjoint)    
            return
        end

        jac_g .= params.last_leak_grad 
    end        
    
    return               
end 


# for objFuncType == 1
function eval_f_par1(pcof::Vector{Float64}, params:: Juqbox.objparams,
    nodes::AbstractArray=[0.0],weights::AbstractArray=[1.0])
    
    nquad = length(nodes)
    if nquad == 1 # Default deterministic optimization
        f, f1, f2 = Juqbox.traceobjgrad(pcof, params, false, false)
    else # Loop over specified nodes and compute risk-neutral objective value
        f = 0.0
        for i = 1:nquad 
            ep = nodes[i]
    
            # Perturb system Hamiltonian
            if ep != 0.0
                for j = 2:size(params.Hconst,2)
                    # params.Hconst[j,j] += H0_old[j,j] + 0.01*ep*(10.0^(j-2))
                    params.Hconst[j,j] += 0.01*ep*(10.0^(j-2))
                end
            end
    
            objf, infid, leak = Juqbox.traceobjgrad(pcof, params, false, false)
            f += objf * weights[i]
            f1 += infid * weights[i]
            f2 += leak * weights[i]

            # Reset system Hamiltonian
            if ep != 0.0
                for j = 2:size(params.Hconst,2)
                    params.Hconst[j,j] -= 0.01*ep*(10.0^(j-2))
                end
            end
        end  
    end

    params.lastTraceInfidelity = f1
    params.lastLeakIntegral = f2

    # Add in Tikhonov regularization
    #tikhonovpenalty = Juqbox.tikhonov_pen(pcof, params)

    return f #.+ tikhonovpenalty
  end

  # for objFuncType == 1
function eval_grad_f_par1(pcof::Vector{Float64}, grad_f::Vector{Float64}, params:: Juqbox.objparams,
    nodes::AbstractArray=[0.0], weights::AbstractArray=[1.0])

    grad_f .= 0.0 # initialize gradient

    nquad = length(nodes)
    if nquad == 1
        _, totalgrad, f1, f2, _, _, _ = Juqbox.traceobjgrad(pcof, params, false, true)
        axpy!(1.0, totalgrad, grad_f)
    else
        # initialize
        f1 = 0.0 # primary obj func
        f2 = 0.0 # secondary obj func

        # Loop over specified nodes and compute risk-neutral objective value
        for i = 1:nquad 
            ep = nodes[i]

            for j = 2:size(params.Hconst,2) # Perturb the Hamiltonian
                params.Hconst[j,j] += 0.01*ep*(10.0^(j-2))
            end

            _, totalgrad, primaryobjf, secondaryobjf, _, _, _ = Juqbox.traceobjgrad(pcof, params, false, true)

            f1 += primaryobjf * weights[i]
            f2 += secondaryobjf * weights[i]

            # grad_f += totalgrad * weights[i]
            axpy!(weights[i], totalgrad, grad_f)

            # Reset the Hamiltonian
            for j = 2:size(params.Hconst,2)
                params.Hconst[j,j] -= 0.01*ep*(10.0^(j-2))
            end
        end
    end

    params.lastTraceInfidelity = f1
    params.lastLeakIntegral = f2

    # Add in Tikhonov regularization gradient term
    #wa = params.wa # use pre-allocated torage
    #wa.gr .= 0.0
    #Juqbox.tikhonov_grad!(pcof, params, wa.gr)  
    #axpy!(1.0, wa.gr, grad_f)

    # Save intermediate parameter vectors
    if params.save_pcof_hist
        push!(params.pcof_hist, copy(pcof)) #pcof_hist is an Array of Vector{Float64}
    end

end


# for objFuncType == 1, intermediate initial conditions (no leak term and no qudrature)
function eval_f_par2(pcof::Vector{Float64}, params:: Juqbox.objparams)

    f, finalDist, _ = Juqbox.lagrange_obj(pcof, params, false)

    # NOTE: when the initial condition isn't unitary, the trace infidelity may be negative
    params.lastTraceInfidelity = max(1e-10, finalDist) 
    params.lastLeakIntegral = 0.0

    # debugging
    # println(pcof)
    # throw("Intentionally stopping here")
    
    # Add in Tikhonov regularization
    #tikhonovpenalty = Juqbox.tikhonov_pen(pcof, params)

    return f #.+ tikhonovpenalty
  end

  # for objFuncType == 1, intermediate initial conditions (no leak term)
function eval_grad_f_par2(pcof::Vector{Float64}, grad_f::Vector{Float64}, params:: Juqbox.objparams)

    grad_f .= 0.0 # initialize gradient storage
    
    #f1, totalgrad = Juqbox.lagrange_objgrad(pcof, params, false, true)
    #axpy!(1.0, totalgrad, grad_f) # AP: why is this needed? By directly assigning grad_f, the calling function reports grad_f = 0 ????

    # in-place grad_f
    _, finalDist, _ = Juqbox.lagrange_grad(pcof, params, grad_f, false)

    # test
    # println("eval_grad_f_par2: grad_f after calling lagrange_objgrad")
    # println(grad_f)

    # Note: the copying of the gradient can be avoided by passing grad_f as an argument to lagrange_objgrad, similar to how the constraints are handled, see the call to unitary_jacobian() below

    # debugging
    # println(pcof)
    # throw("Intentionally stopping here")
    
    # NOTE: when the initial condition isn't unitary, the trace infidelity may be negative
    params.lastTraceInfidelity = max(1e-10, finalDist) 
    params.lastLeakIntegral = 0.0

    # Add in Tikhonov regularization gradient term
    #wa = params.wa # use pre-allocated storage
    #wa.gr .= 0.0
    #Juqbox.tikhonov_grad!(pcof, params, wa.gr)  
    #axpy!(1.0, wa.gr, grad_f)

    # Save intermediate parameter vectors
    if params.save_pcof_hist
        push!(params.pcof_hist, copy(pcof)) #pcof_hist is an Array of Vector{Float64}
    end
    return true
end

###########################################
# IpOPT callback function for evaluating all unitary constraints
###########################################
function eval_g_par2(pcof::Vector{Float64}, g::Vector{Float64}, params::objparams)
    unitary_constraints(pcof, g, params, false)
end # function eval_g_par

###########################################
# IpOPT callback function for evaluating the Jacobian of the unitary constraints
###########################################
function eval_jac_g_par2(pcof::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, jac_g::Union{Nothing,Vector{Float64}}, p:: Juqbox.objparams)
    
    if jac_g === nothing 
        #println("eval_jac_g_par2: initialization, length(rows) = ", length(rows), "length(cols) = ", length(cols), " length(cols) = ", length(cols))
        unitary_jacobian_idx(rows, cols, p, false)
    else
        unitary_jacobian(pcof, jac_g, p, false) # enter all elements of the Jacobian
    end        
    
    return               
end 

function eval_f_par3(pcof::Vector{Float64}, params:: Juqbox.objparams)

    f, absInfid, _ = Juqbox.final_obj(pcof, params, false)

    # NOTE: when the initial condition isn't unitary, the trace infidelity may be negative
    params.lastTraceInfidelity = absInfid 
    params.lastLeakIntegral = 0.0

    return f
  end

function eval_grad_f_par3(pcof::Vector{Float64}, grad_f::Vector{Float64}, params:: Juqbox.objparams)

    grad_f[:] .= 0.0 # initialize gradient storage

    # in-place grad_f
    _, absInfid, _ = Juqbox.final_grad(pcof, params, grad_f, false)
    
    # NOTE: when the initial condition isn't unitary, the trace infidelity may be negative
    params.lastTraceInfidelity = absInfid
    params.lastLeakIntegral = 0.0

    # Save intermediate parameter vectors
    if params.save_pcof_hist
        push!(params.pcof_hist, copy(pcof)) #pcof_hist is an Array of Vector{Float64}
    end
    return true
end

###########################################
# IpOPT callback function for evaluating all jump constraints
###########################################
function eval_g_par3(pcof::Vector{Float64}, g::Vector{Float64}, params::objparams)
    c2norm_constraints(pcof, g, params, false)
end # function eval_g_par

###########################################
# IpOPT callback function for evaluating the Jacobian of the jump constraints
###########################################
function eval_jac_g_par3(pcof::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, jac_g::Union{Nothing,Vector{Float64}}, p:: Juqbox.objparams)
    
    if jac_g === nothing 
        #println("eval_jac_g_par2: initialization, length(rows) = ", length(rows), "length(cols) = ", length(cols), " length(cols) = ", length(cols))
        c2norm_jacobian_idx(rows, cols, p, false)
    else
        c2norm_jacobian(pcof, jac_g, p, false) # enter all elements of the Jacobian
    end        
    
    return true        
end 

###########################################
# IpOPT callback function for evaluating all pointwise jump constraints
###########################################
function eval_g_par5(pcof::Vector{Float64}, g::Vector{Float64}, params::objparams)
    state_constraints(pcof, g, params, false)
end # function eval_g_par

###########################################
# IpOPT callback function for evaluating the Jacobian of the pointwise jump constraints
###########################################
function eval_jac_g_par5(pcof::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, jac_g::Union{Nothing,Vector{Float64}}, p:: Juqbox.objparams)
    
    if jac_g === nothing 
        #println("eval_jac_g_par2: initialization, length(rows) = ", length(rows), "length(cols) = ", length(cols), " length(cols) = ", length(cols))
        state_jacobian_idx(rows, cols, p, false)
    else
        state_jacobian(pcof, jac_g, p, false) # enter all elements of the Jacobian
    end        
    
    return true        
end

###########################################
# IpOPT callback function for evaluating unitary and jump constraints
###########################################
function eval_g_par4(pcof::Vector{Float64}, g::Vector{Float64}, params::objparams)
    unitary_constraints(pcof, g, params, false)
    c2norm_constraints(pcof, g, params, false)
end # function eval_g_par

###########################################
# IpOPT callback function for evaluating the Jacobian of the unitary constraints
###########################################
function eval_jac_g_par4(pcof::Vector{Float64}, rows::Vector{Int32}, cols::Vector{Int32}, jac_g::Union{Nothing,Vector{Float64}}, p:: Juqbox.objparams)
    
    if jac_g === nothing 
        #println("eval_jac_g_par4: initialization, length(rows) = ", length(rows), "length(cols) = ", length(cols), " length(cols) = ", length(cols))
        unitary_jacobian_idx(rows, cols, p, false)
        c2norm_jacobian_idx(rows, cols, p, false)
    else
        unitary_jacobian(pcof, jac_g, p, false) # enter all elements of the Jacobian
        c2norm_jacobian(pcof, jac_g, p, false)
    end        
    
    return               
end 

###########################################
# IpOPT callback function for the case w/o any constraints
###########################################
function eval_g_empty(pcof::Vector{Float64}, g::Vector{Float64})
    return nothing
    #g[1] = 0.0
    #return g[1]
end


function eval_jac_g_empty(
    x::Vector{Float64},         # Current solution
    # x_new:: Bool,               # If new vector x
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
        push!(params.objHist, abs(obj_value)) # can be negative for Aug-Lag
        push!(params.dualInfidelityHist, inf_du)
        push!(params.primaryHist, params.lastTraceInfidelity) # infidelity
        push!(params.secondaryHist,  params.lastLeakIntegral)
        if params.constraintType == 0 && params.nTimeIntervals > 1 # Aug-Lagrange
            inf_jump = maximum(params.nrm2_Cjump) # norm-squared scales the same as the infidelity
        else
            inf_jump = inf_pr^2
        end
        push!(params.constraintViolationHist, inf_jump)
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
    prob = ipopt_setup(params, nCoeff, maxAmp; maxIter, lbfgsMax, coldStart, ipTol, acceptTol, acceptIter, nodes=[0.0], weights=[1.0])

Setup structure containing callback functions and convergence criteria for 
optimization via IPOPT. Note the last two inputs, `nodes', and 
`weights', are to be used when performing a simple risk-neutral optimization
where the fundamental frequency is random.

# Arguments
- `params:: objparams`: Struct with problem definition
- `nCoeff:: Int64`: Number of parameters in optimization
- `maxAmp:: Vector{Float64}`: Maximum amplitude for each control function (size Nctrl) 
- `maxIter:: Int64`: Maximum number of iterations to be taken by optimizer (keyword arg)
- `lbfgsMax:: Int64`: Maximum number of past iterates for Hessian approximation by L-BFGS (keyword arg)
- `coldStart:: Bool`: true (default): start a new optimization with ipopt, false: continue a previous optimization (keyword arg)
- `ipTol:: Float64`: Desired convergence tolerance (relative) (keyword arg)
- `acceptTol:: Float64`: Acceptable convergence tolerance (relative) (keyword arg)
- `acceptIter:: Int64`: Number of acceptable iterates before triggering termination (keyword arg)
- `nodes:: Array{Float64, 1}`: Risk-neutral opt: User specified quadrature nodes on the interval [-ϵ,ϵ] for some ϵ (optinal keyword arg)
- `weights:: Array{Float64, 1}`: Risk-neutral opt: User specified quadrature weights on the interval [-ϵ,ϵ] for some ϵ (optional keyword arg)
"""
function ipopt_setup(params:: Juqbox.objparams, nCoeff:: Int64, maxAmp:: Vector{Float64}; maxIter:: Int64, lbfgsMax:: Int64, coldStart:: Bool, ipTol:: Float64, acceptTol:: Float64, acceptIter:: Int64, nodes::AbstractArray=[0.0], weights::AbstractArray=[1.0])

    minCoeff, maxCoeff = control_bounds(params, maxAmp)
    
    intermediate(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials) =
                intermediate_par(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials, params)

    #testing
    #println("ipopt_setup(): params.objFuncType = ", params.objFuncType, " #intervals = ", params.nTimeIntervals)

    # if params.objFuncType == 3
    #     # treat the leakage as an inequality constraint
    #     nconst = 1 # One constraint
    #     nEleJac = nCoeff
    #     nEleHess = 0
    #     g_L = -2e19.*ones(nconst) # no lower bound needed because the leakage is always non-negative
    #     g_U = params.leak_ubound.*ones(nconst)

    #     #Initialize the last fidelity and leak terms and gradients
    #     params.last_pcof = 1e9.*rand(nCoeff)
    #     params.last_infidelity_grad = 1e9.*rand(nCoeff)
    #     params.last_leak_grad = 1e9.*rand(nCoeff)        
    
    #     # Comment out to use xnew with later version of ipopt
    #     # eval_f(pcof,x_new) = eval_f_par(pcof, x_new,params, nodes, weights)
    #     # eval_grad_f(pcof,x_new, grad_f) = eval_grad_f_par(pcof,x_new, grad_f, params, nodes, weights)
    #     # eval_g(pcof,x_new,g) = eval_g_par(pcof,x_new,g,params,nodes,weights)
    #     # eval_jac_g(pcof,x_new,rows,cols,jac_g) = eval_jac_g_par(pcof,x_new,rows,cols,jac_g,params,nodes,weights)
        
    #     # callback functions need access to the params object
    #     eval_f(pcof) = eval_f_par(pcof, params, nodes, weights)
    #     eval_grad_f(pcof, grad_f) = eval_grad_f_par(pcof,grad_f, params, nodes, weights)
    #     eval_g(pcof,g) = eval_g_par(pcof, g, params, nodes, weights)
    #     eval_jac_g(pcof,rows,cols,jac_g) = eval_jac_g_par(pcof, rows, cols, jac_g, params,nodes, weights)
        
    #     # setup the Ipopt data structure
    #     prob = CreateIpoptProblem( nCoeff, minCoeff, maxCoeff, nconst, g_L, g_U, nEleJac, nEleHess, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h);
    # else # params.objFuncType = 1 (add infidelity and leakage in the objective) 
       
    #println("ipopt_setup: imposing constraints of type = ", params.constraintType, ", # timeIntervals = ", params.nTimeIntervals)
    
    if (params.constraintType == 0) # Minimize the Lagrangian without imposing constraints
        nConst = 0
        nEleJac = 0
        nEleHess = 0
        g_L = zeros(0);
        g_U = zeros(0);

        # callback functions need access to the params object
        # eval_f1(pcof) = eval_f_par1(pcof, params, nodes, weights)
        # eval_grad_f1(pcof, grad_f) = eval_grad_f_par1(pcof, grad_f, params, nodes, weights)
        # to support fidType = 1, use lagrange_objgrad()
        eval_f1(pcof) = eval_f_par2(pcof, params)
        eval_grad_f1(pcof, grad_f) = eval_grad_f_par2(pcof, grad_f, params)
        
        # setup the Ipopt data structure
        prob = CreateIpoptProblem( nCoeff, minCoeff, maxCoeff, nConst, g_L, g_U, nEleJac, nEleHess, eval_f1, eval_g_empty, eval_grad_f1, eval_jac_g_empty, eval_h);
    elseif params.constraintType == 1 
        # unitary equality constraints on intermediate initial conditions and zero norm(jump)^2 constraints
        params.nConstUnitary = (params.nTimeIntervals - 1) * params.Ntot^2
        params.nEleJacUnitary = (params.nTimeIntervals - 1) * (2*params.Ntot^2 + 8*params.Ntot * div(params.Ntot*(params.Ntot - 1),2))
        # add in sizes for the jump constraints
        nConst = params.nConstUnitary + (params.nTimeIntervals - 1) # one constraint per intermediate initial condition
        nEleJac = params.nEleJacUnitary + (params.nTimeIntervals - 1) * (params.nAlpha + params.nWinit) # the Jacobian wrt B-spline coeffs and wrt Winit)
        if params.nTimeIntervals > 2
            nEleJac += (params.nTimeIntervals - 2) * params.nWinit 
        end

        # no Hessian info (using BFGS)
        nEleHess = 0

        g_L = zeros(nConst); # Equality constraints
        g_U = zeros(nConst);

        # callback functions need access to the params object
        eval_f2(pcof) = eval_f_par3(pcof, params) # eval_f_par2(pcof, params)
        eval_grad_f2(pcof, grad_f) = eval_grad_f_par3(pcof, grad_f, params) #eval_grad_f_par2(pcof, grad_f, params)
        
        # callbacks for evaluating the constraints and their Jacobian
        eval_g4(pcof, g) = eval_g_par4(pcof, g, params) 
        eval_jac_g4(pcof, rows, cols, jac_g) = eval_jac_g_par4(pcof, rows, cols, jac_g, params) 
    
        println("ipopt_setup, params.constraintType = ", params.constraintType)
        println("params.nConstUnitary = ", params.nConstUnitary, " nConst = ", nConst)
        println("params.nEleJacUnitary = ", params.nEleJacUnitary, " nEleJac = ", nEleJac)

        # setup the Ipopt data structure
        prob = CreateIpoptProblem( nCoeff, minCoeff, maxCoeff, nConst, g_L, g_U, nEleJac, nEleHess, eval_f2, eval_g4, eval_grad_f2, eval_jac_g4, eval_h)
    elseif params.constraintType == 2 
        # zero norm^2(jump) across time intervals
        nConst = (params.nTimeIntervals - 1) # one constraint per intermediate initial condition
        # begin with the Jacobian wrt B-spline coeffs, and wrt Winit_next)
        nEleJac = (params.nTimeIntervals - 1) * (params.nAlpha + params.nWinit) 
        if params.nTimeIntervals > 2
            nEleJac += (params.nTimeIntervals - 2) * params.nWinit 
        end
        nEleHess = 0
        g_L = zeros(nConst); # Equality constraints
        g_U = zeros(nConst);

        # callback functions need access to the params object
        # testing the finalDist + gamma*norm^2(jump) objective
        # combined with equality constraints for norm^2(jump)=0
        
        # call lagrange_obj/grad, penalizes the c2norm and includes lagrange multipliers
        #eval_f3(pcof) = eval_f_par2(pcof, params)
        #eval_grad_f3(pcof, grad_f) = eval_grad_f_par2(pcof, grad_f, params)
        
        # call final_obj/grad
        eval_f3(pcof) = eval_f_par3(pcof, params)
        eval_grad_f3(pcof, grad_f) = eval_grad_f_par3(pcof, grad_f, params)
        
        # callbacks for evaluating the constraints and their Jacobian
        # call c2norm_constraint
        eval_g3(pcof, g) = eval_g_par3(pcof, g, params) 
        # call c2norm_jacobian or c2norm_jacobian_idx
        eval_jac_g3(pcof, rows, cols, jac_g) = eval_jac_g_par3(pcof, rows, cols, jac_g, params)

        # setup the Ipopt data structure
        prob = CreateIpoptProblem( nCoeff, minCoeff, maxCoeff, nConst, g_L, g_U, nEleJac, nEleHess, eval_f3, eval_g3, eval_grad_f3, eval_jac_g3, eval_h)
    elseif params.constraintType == 4 
        # zero jump in state across time intervals
       nConst = (params.nTimeIntervals - 1)*params.nWinit # 2*N^2 constraint per intermediate initial condition

        nEleJac = 0
        for q = 1: params.nTimeIntervals - 1
            nEleJac += 2 * params.NfreqTot * (params.d1_end[q] - params.d1_start[q] + 1) * params.nWinit # Jacobian wrt alpha (Overestimating)
        end
        nEleJac += (params.nTimeIntervals - 1) * params.nWinit # Jacobian wrt target state (next) Winit
        if params.nTimeIntervals > 2 # Jacobian wrt initial conditions, Winit
            nEleJac += (params.nTimeIntervals - 2) * params.nWinit * 2 * params.N
        end

        nEleHess = 0
        g_L = zeros(nConst); # Equality constraints
        g_U = zeros(nConst);

        # callback functions need access to the params object
        # testing the finalDist + gamma*norm^2(jump) objective
        # combined with equality constraints for norm^2(jump)=0
        
        # call final_obj/grad
        eval_f5(pcof) = eval_f_par3(pcof, params)
        eval_grad_f5(pcof, grad_f) = eval_grad_f_par3(pcof, grad_f, params)
        
        # callbacks for evaluating the constraints and their Jacobian
        # call jacobian_constraint
        eval_g5(pcof, g) = eval_g_par5(pcof, g, params) 
        # call state_jacobian or state_jacobian_idx
        eval_jac_g5(pcof, rows, cols, jac_g) = eval_jac_g_par5(pcof, rows, cols, jac_g, params)

        # setup the Ipopt data structure
        prob = CreateIpoptProblem( nCoeff, minCoeff, maxCoeff, nConst, g_L, g_U, nEleJac, nEleHess, eval_f5, eval_g5, eval_grad_f5, eval_jac_g5, eval_h)
    else
        println("ConstraintType = ", params.constraintType, " is not yet implemented" )
        throw("IPOPT interface can not be defined")
    end

    # tmp
    # if @isdefined createProblem
    #     prob = createProblem( nCoeff, minCoeff, maxCoeff, nconst, g_L, g_U, nEleJac, nEleHess, eval_f, eval_g, eval_grad_f, eval_jac_g);
    # else
    #     # prob = CreateIpoptProblem( nCoeff, minCoeff, maxCoeff, nconst, g_L, g_U, nEleJac, nEleHess, eval_f, eval_g, eval_grad_f, eval_jac_g,eval_h,expose_xnew=true);
    #     prob = CreateIpoptProblem( nCoeff, minCoeff, maxCoeff, nconst, g_L, g_U, nEleJac, nEleHess, eval_f, eval_g, eval_grad_f, eval_jac_g,eval_h);
    # end

    if @isdefined addOption
        addOption( prob, "hessian_approximation", "limited-memory");
        addOption( prob, "limited_memory_max_history", lbfgsMax);
        addOption( prob, "max_iter", maxIter);
        addOption( prob, "tol", ipTol);
        addOption( prob, "acceptable_tol", acceptTol);
        addOption( prob, "acceptable_iter", acceptIter);
        addOption( prob, "jacobian_approximation", "exact");
        # addOption( prob, "derivative_test", "first-order");
        # addOption( prob, "derivative_test_tol", 0.0001);
        
        if !coldStart # enable warm start of Ipopt
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
        AddIpoptStrOption( prob, "jacobian_approximation", "exact");
        # AddIpoptStrOption( prob, "derivative_test", "first-order");
        # AddIpoptNumOption( prob, "derivative_test_tol", 1.0e-4);
        AddIpoptNumOption( prob, "derivative_test_perturbation", 1.0e-7);
        
        

        if !coldStart # enable warm start of Ipopt
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
        SetIntermediateCallback(prob, intermediate)
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
end # ipopt_setup


"""
    pcof = run_optimizer(params, pcof0, maxAmp; maxIter=50, lbfgsMax=200, coldStart=true, ipTol=1e-5, acceptTol=1e-5, acceptIter=15, print_level=5, print_frequency_iter=1, nodes=[0.0], weights=[1.0])

Call IPOPT to  optimizize the control functions.

# Arguments
- `params:: objparams`: Struct with problem definition
- `pcof0:: Vector{Float64}`: Initial guess for the control vector
- `maxAmp:: Vector{Float64}`: Maximum amplitude for each control function (size Nctrl)
- `maxIter:: Int64`: (Optional-kw) Maximum number of iterations to be taken by optimizer
- `lbfgsMax:: Int64`: (Optional-kw) Maximum number of past iterates for Hessian approximation by L-BFGS
- `coldStart:: Bool`: (Optional-kw) true (default): start a new optimization with ipopt; false: continue a previous optimization
- `ipTol:: Float64`: (Optional-kw) Desired convergence tolerance (relative)
- `acceptTol:: Float64`: (Optional-kw) Acceptable convergence tolerance (relative)
- `acceptIter:: Int64`: (Optional-kw) Number of acceptable iterates before triggering termination
- `print_level:: Int64`: (Optional-kw) Ipopt verbosity level (5)
- `print_frequency_iter:: Int64`: (Optional-kw) Ipopt printout frequency (1)
- `nodes:: AbstractArray`: (Optional-kw) Risk-neutral opt: User specified quadrature nodes on the interval [-ϵ,ϵ] for some ϵ
- `weights:: AbstractArray`: (Optional-kw) Risk-neutral opt: User specified quadrature weights on the interval [-ϵ,ϵ] for some ϵ
- `derivative_test:: Bool`: (Optional-kw) Set to true to check the gradient against a FD approximation (default is false)
"""
function run_optimizer(params:: objparams, pcof0:: Vector{Float64}, maxAmp:: Vector{Float64}; maxIter::Int64 = 50, lbfgsMax:: Int64=200, coldStart:: Bool=true, ipTol:: Float64=1e-5, acceptTol:: Float64=1e-5, acceptIter:: Int64=15, print_level:: Int64=5, print_frequency_iter:: Int64=1, nodes::AbstractArray=[0.0], weights::AbstractArray=[1.0], derivative_test::Bool=false)
    
    # start by setting up the Ipopt object: prob
    #println("Ipopt initialization timing:")
    #@time 

    # println("run_optimizer: length(pcof0) = ", length(pcof0))

    prob = Juqbox.ipopt_setup(params, length(pcof0), maxAmp, maxIter=maxIter, lbfgsMax=lbfgsMax, coldStart=coldStart, ipTol=ipTol, acceptTol=acceptTol, acceptIter=acceptIter, nodes=nodes, weights=weights)

    AddIpoptIntOption(prob, "print_level", print_level)
    AddIpoptIntOption(prob, "print_frequency_iter", print_frequency_iter)

    if derivative_test # for testing the gradient
        if @isdefined addOption
            addOption( prob, "derivative_test", "first-order");
        else
            AddIpoptStrOption( prob, "derivative_test", "first-order")
        end
    end        
    # initial guess for IPOPT; make a copy of pcof0 to avoid overwriting it
    prob.x = copy(pcof0);

    # Ipopt solver
    if print_level > 0
        println("*** Starting the optimization ***")
        if @isdefined solveProblem
            @time solveProblem(prob);
        else 
            @time IpoptSolve(prob);
        end
    else
        if @isdefined solveProblem
            solveProblem(prob);
        else 
            IpoptSolve(prob);
        end
    end
    pcof = prob.x;

    #save the b-spline coeffs on a JLD2 file (replace by an explicit call to save_pcof)
    # if length(baseName)>0
    #     fileName = baseName * ".jld2"
    #     save_pcof(fileName, pcof)
    #     println("Saved B-spline parameters on binary jld2-file '", fileName, "'");
    # end

    return pcof

end # run_optimizer

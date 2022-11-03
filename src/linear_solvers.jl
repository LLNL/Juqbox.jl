using SparseArrays
using LinearAlgebra


const NEUMANN_SOLVER = 1
const JACOBI_SOLVER  = 2
const GAUSSIAN_ELIM_SOLVER = 3


"""
    linear_solver = lsolver_object(; tol  = tol,
                                     max_iter = max_iter,
                                     nrhs = nrhs,
                                     solver = NEUMANN_SOLVER)

Constructor for the mutable struct lsolver_object. That allcoates arrays and sets up the function pointers
	for the different linear solvers supported.
 
# Arguments
- `tol::Float64  = 1e-10` : Convergence tolerance of the iterative solver (only needed for Jacobi)
- `max_iter::Int64   = 3` : Max number of iterations for the linear solver
- `nrhs::Int64   = 1` : Number of right-hand sides (used for tolerance scaling)
- `solver::Int64 = NEUMANN_SOLVER` : (keyword) ID of the iterative solver.
                                     Can take the value of NEUMANN_SOLVER (i.e. 1) or JACOBI_SOLVER (i.e. 2)
                                     See examples/cnot2-jacobi-setup.jl
"""
mutable struct lsolver_object

    tol   ::Float64
    max_iter  ::Int64    
    solver_id ::Int64
    solver_name ::String
    solve ::Function
    print_info ::Function
    
    function lsolver_object(;tol::Float64 = 1e-10, max_iter::Int64 = 3, nrhs::Int64=1, solver::Int64 = NEUMANN_SOLVER)
        
        if solver == JACOBI_SOLVER
			tol *= sqrt(nrhs)
			solve = (a,b,c,d,e) -> jacobi!(a,b,c,d,e,max_iter,tol)
            solver_name = "Jacobi"
            print_info = () -> println("*** Using linear solver: ", solver_name," with max_iter = ", max_iter, ", tol = ", tol)

        elseif solver == NEUMANN_SOLVER
            solve = (a,b,c,d,e) -> neumann!(a,b,c,d,e,max_iter)
            solver_name = "Neumann"
            print_info = () -> println("*** Using linear solver: ", solver_name," with max_iter = ", max_iter)
        elseif solver == GAUSSIAN_ELIM_SOLVER
            solve = (a,b,c,d,e) -> gaussian_elim_solve!(a,b,c,d)
            solver_name = "Built-in"
            print_info = () -> println("*** Using linear solver: ", solver_name," with max_iter = ", max_iter)
        else
            error("Please specify a supported linear solver")
        end
	
        new(tol,max_iter,solver,solver_name,solve,print_info)
    end

end #mutable struct lsolver_object    

#Routine to recreate closured for updated max_iter and tol values
function recreate_linear_solver_closure!(lsolver::lsolver_object)
    
    if lsolver.solver_id == JACOBI_SOLVER
        lsolver.solve = (a,b,c,d,e) -> jacobi!(a,b,c,d,e,lsolver.max_iter,lsolver.tol)
    elseif lsolver.solver_id == NEUMANN_SOLVER
        lsolver.solve = (a,b,c,d,e) -> neumann!(a,b,c,d,e,lsolver.max_iter)
    elseif lsolver.solver_id == GAUSSIAN_ELIM_SOLVER
        lsolver.solve = (a,b,c,d,e) -> gaussian_elim_solve!(a,b,c,e)
    end

end


@inline function neumann!(h::Float64, S::SparseMatrixCSC{Float64,Int64}, 
						 B::Array{Float64,N}, T::Array{Float64,N}, X::Array{Float64,N},nterms::Int64) where N
	copy!(X,B)
	copy!(T,B)
	coeff = 1.0
	for j = 1:nterms
		LinearAlgebra.mul!(T,S,B)
		coeff *= (0.5*h)
		LinearAlgebra.axpy!(coeff, T, X)
		copy!(B,T)
	end
end

@inline function neumann!(h::Float64, S::Array{Float64,N}, B::Array{Float64,N}, 
						T::Array{Float64,N}, X::Array{Float64,N}, nterms::Int64,) where N

	copy!(X,B)
	copy!(T,B)
	coeff = 1.0
	for j = 1:nterms
		LinearAlgebra.mul!(T,S,B)
		coeff *= (0.5*h)
		LinearAlgebra.axpy!(coeff, T, X)
		copy!(B,T)
	end
end



@inline function jacobi!(h::Float64, S::Array{Float64,N}, B::Array{Float64,N},
                         T::Array{Float64,N}, X::Array{Float64,N}, max_iter::Int64,tol::Float64) where N
						 
	X .= B #copy!(X,B)
	coeff = -0.5*h
	o_coeff = 1.0/coeff
	S .*= coeff
	for j = 1:max_iter
		LinearAlgebra.mul!(T,S,X)
		T = B - T
		err = norm(T - X)
		X .= T #copy!(X,T)
		if err < tol
			S .*= o_coeff
			return
		end
	end
	S .*= o_coeff
    @warn "WARNING: Jacobi solver reached maximum number of iterations ($max_iter), final norm(residual) = $(err)"
end


@inline function jacobi!(h::Float64, S::SparseMatrixCSC{Float64,Int64}, B::Array{Float64,N},
                         T::Array{Float64,N}, X::Array{Float64,N}, max_iter::Int64,tol::Float64) where N
				 
	X .= B #copy!(X,B)
	coeff = -0.5*h
	o_coeff = 1.0/coeff
	S .*= coeff
	for j = 1:max_iter
		LinearAlgebra.mul!(T,S,X)
		T = B - T
		err = norm(T - X)
		X .= T #copy!(X,T)
		if err < tol
			S .*= o_coeff
			return
		end
	end
	S .*= o_coeff
    @warn "WARNING: reached maximum number of iterations ($max_iter) in Jacobi solver!"
end

"""
For comparing other solvers with the built-in linear solver using Gaussian
elimination.
"""
@inline function gaussian_elim_solve!(h::Float64, S::Array{Float64,N}, B::Array{Float64,N},
                         X::Array{Float64,N}) where N
    X .= (LinearAlgebra.I - (0.5h.*S))\B
end

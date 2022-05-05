using SparseArrays
using LinearAlgebra


const NEUMANN_SOLVER = 1
const JACOBI_SOLVER  = 2

mutable struct lsolver_object

    nvar  ::Int64
    tol   ::Float64
    iter  ::Int64
    rwork ::Array{Float64,2}
    
    solver_id ::Int64
    solve ::Function
    

    function lsolver_object(;nvar::Int64 = 0, ndim::Int64 = 0, tol::Float64 = 1e-10, 
                            iter::Int64 = 3, solver::Int64 = NEUMANN_SOLVER)
        
        rwork = zeros(Float64,nvar,ndim)
        if solver == JACOBI_SOLVER
            solve = (a,b,c,d,e) -> jacobi!(a,b,c,d,e,rwork,iter,tol)
        elseif solver == NEUMANN_SOLVER
            solve = (a,b,c,d,e) -> neumann!(a,b,c,d,e,iter)
        else
            error("Please specify a correct linear solver")
        end
        new(nvar,tol,iter,rwork,solver,solve)

    end

end #mutable struct lsolver_object    



#Routine to resize appropriate work array and recreate closured for updated iter and tol values
function allocate_linear_solver_arrays!(lsolver::lsolver_object, nvar::Int64, ndim ::Int64)

    
    if lsolver.solver_id == JACOBI_SOLVER
        #Resize work array
        lsolver.rwork = zeros(Float64,nvar,ndim)

        #Recreate closure using new resized work array
        lsolver.solve = (a,b,c,d,e) -> jacobi!(a,b,c,d,e,lsolver.rwork,lsolver.iter,lsolver.tol)
    elseif lsolver.solver_id == NEUMANN_SOLVER
        lsolver.solve = (a,b,c,d,e) -> neumann!(a,b,c,d,e,lsolver.iter)
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
						 T::Array{Float64,N}, X::Array{Float64,N},xold::Array{Float64,N},
                         iter::Int64,tol::Float64) where N
    
	copy!(X,B)
	coeff = -0.5*h
  	# nsize, _ = size(X)
    S .*= coeff
	for j = 1:iter
		copy!(xold,X)
        LinearAlgebra.mul!(T,S,xold)
		# for i in 1:nsize
		# 	# D = 1.0 + coeff*S[i,i]
		# 	# Ax = T[i,:] .- D*xold[i,:]
		# 	# X[i,:] = 1.0/D .* (B[i,:] .- Ax)
		# end
		# Since diag(S) = 0	
		X .= B .- T
		err = norm(xold .- X)
		if err < tol
            S ./= coeff
			return
		end
	end
    S ./= coeff
end


@inline function jacobi!(h::Float64, S::SparseMatrixCSC{Float64,Int64}, B::Array{Float64,N}, 
						T::Array{Float64,N}, X::Array{Float64,N},xold::Array{Float64,N},
                        iter::Int64,tol::Float64) where N

	copy!(X,B)
	coeff = -0.5*h
    S .*= coeff
	# nsize, _ = size(X)
	for j = 1:iter
		copy!(xold,X)
		LinearAlgebra.mul!(T,S,xold)
		# for i in 1:nsize
		# 	# D = 1.0 + coeff*S[i,i]
		# 	# Ax = T[i,:] .- D*xold[i,:]
		# 	# X[i,:] = 1.0/D .* (B[i,:] .- Ax)
		# end
		# Since diag(S) = 0	
		X .= B .- T
		err = norm(xold .- X)
		if err < tol
            S ./= coeff
			return
		end
	end
    S ./= coeff    
end
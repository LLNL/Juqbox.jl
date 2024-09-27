using SparseArrays
using LinearAlgebra

# Julia versions prior to v"1.3.1" can't use LinearAlgebra's 5 argument mul!, routines
# included here for backwards compatability
if(VERSION < v"1.3.1")
    include("backwards_compat.jl")
end

#Used in testing Implicit Midpoint
@inline function step_midpoint(stepper::svparams, t::Float64, u::Array{Float64,N},
	v::Array{Float64,N}, h::Float64, uforce::Function, vforce::Function) where N

    uforce05 = uforce(t + 0.5*h)
    vforce05 = vforce(t + 0.5*h)

    t,u,v = step_midpoint(stepper, t, u, v, h, uforce05, vforce05)

	return t, u, v
end


@inline function step_midpoint(stepper:: svparams, t::Float64, u::Array{Float64,N}, v::Array{Float64,N},
	h::Float64, uforce05::Array{Float64, N}, vforce05::Array{Float64, N}) where N

	S = stepper.S
	K = stepper.K
	In = stepper.In


    u05 = uforce05
    v05 = vforce05

	K05 = K(t + 0.5*h)
	S05 = S(t + 0.5*h)

	A = h/2 * K05
	B = h/2 * K05 * u
	C = h/2 * S05
	D = h/2 * S05 * v
	E = h*v05
	F = h/2 * S05
	G = h/2 * S05 * u
	H = h/2 * K05 * v
	J = h*u05

	Q = (In - C)

	u_lhs = In - F + A*(Q\A)
	u_rhs = G - A*(Q\(B + D + E + v)) - H + J + u
	
	u = u_lhs\u_rhs

	v = Q\(A*u + B + D + E + v)

	t = t + h

	return t, u, v
end

@inline function m_step_slow!(t::Float64, u::Array, v::Array, h::Float64,
	K05,S05, uforce::Array, vforce::Array)
	
	row, col = size(K05)
	In = Matrix(1.0I, row, col)
	A = h/2 * K05
	B = h/2 * K05 * u
	C = h/2 * S05
	D = h/2 * S05 * v
	E = h*vforce
	F = h/2 * S05
	G = h/2 * S05 * u
	H = h/2 * K05 * v
	J = h*uforce

	Q = (In - C)

	u_lhs = In .- F .+ A*(Q\A)
	u_rhs = G .- A*(Q\(B .+ D .+ E .+ v)) .- H .+ J .+ u
	
	u .= u_lhs\u_rhs

	v .= Q\(A*u .+ B .+ D .+ E .+ v)

	t = t + h

	return t
end

@inline function m_step_no_forcing_slow!(t::Float64, u::Array, v::Array, h::Float64,
	K05,S05)
	
	row, col = size(K05)
	In = Matrix(1.0I, row, col)
	A = h/2 * K05
	B = h/2 * K05 * u
	C = h/2 * S05
	D = h/2 * S05 * v
	E = 0.0
	F = h/2 * S05
	G = h/2 * S05 * u
	H = h/2 * K05 * v
	J = 0.0

	Q = (In - C)

	u_lhs = In .- F .+ A*(Q\A)
	u_rhs = G .- A*(Q\(B .+ D .+ E .+ v)) .- H .+ J .+ u
	
	u .= u_lhs\u_rhs

	v .= Q\(A*u .+ B .+ D .+ E .+ v)

	t = t + h

	return t
end


#Sparse implementation of Implicit Midpoint Rule with no forcing
@inline function m_step_no_forcing!(t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, h::Float64,
	K05::SparseMatrixCSC{Float64,Int64},S05::SparseMatrixCSC{Float64,Int64}, rhs_u::Array{Float64,N}, rhs_v::Array{Float64,N},
	x0_u, x0_v, norm_matrix_u, norm_matrix_v, linear_solver::lsolver_object) where N

	#Create the right hand side of the system Ax = b
    #Create the real part of the right hand side
	mul!(rhs_u, S05, u, h/2, 0)
	axpy!(1, u, rhs_u)
	mul!(rhs_u, K05, v, -h/2, 1)
	
	#Create the imaginary part of the right hand side
	mul!(rhs_v, S05, v, h/2, 0)
	axpy!(1, v, rhs_v)
	mul!(rhs_v, K05, u, h/2, 1)

	#Solve the system using Jacobi's method
	linear_solver.solve(h, rhs_u, rhs_v, S05, K05, u, v, x0_u, x0_v, norm_matrix_u, norm_matrix_v)

	u .= x0_u
	v .= x0_v

	t = t + h

	return t
end

#Dense version of above function
@inline function m_step_no_forcing!(t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, h::Float64,
	K05::Array{Float64, N},S05::Array{Float64, N}, rhs_u::Array{Float64,N}, rhs_v::Array{Float64,N},
	x0_u, x0_v, norm_matrix_u, norm_matrix_v, linear_solver::lsolver_object) where N

	mul!(rhs_u, S05, u, h/2, 0)
	axpy!(1, u, rhs_u)
	mul!(rhs_u, K05, v, -h/2, 1)

	mul!(rhs_v, S05, v, h/2, 0)
	axpy!(1, v, rhs_v)
	mul!(rhs_v, K05, u, h/2, 1)

	linear_solver.solve(h, rhs_u, rhs_v, S05, K05, u, v, x0_u, x0_v, norm_matrix_u, norm_matrix_v)
    
	u .= x0_u
    v .= x0_v
	
	t = t + h

	return t
end


#Sparse implmentation of Implicit Midpoint rule with forcing
@inline function m_step!(t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, h::Float64,
	K05::SparseMatrixCSC{Float64,Int64},S05::SparseMatrixCSC{Float64,Int64}, uforce::Array{Float64,N}, vforce::Array{Float64,N}, 
	rhs_u::Array{Float64,N}, rhs_v::Array{Float64,N}, x0_u, x0_v, norm_matrix_u, norm_matrix_v, linear_solver::lsolver_object) where N

	#Create the right hand side of the system Ax = b
    #Create the real part of the right hand side
	mul!(rhs_u, S05, u, h/2, 0)
	axpy!(1, u, rhs_u)
	mul!(rhs_u, K05, v, -h/2, 1)

	#Add real forcing term to real right hand side
	axpy!(h, uforce, rhs_u)

	#Create the imaginary part of the right hand side
	mul!(rhs_v, S05, v, h/2, 0)
	axpy!(1, v, rhs_v)
	mul!(rhs_v, K05, u, h/2, 1)

	#Add imaginary forcing term to imaginary right hand side
	axpy!(h, vforce, rhs_v)
	
	#Solve the system using Jacobi's method
	linear_solver.solve(h, rhs_u, rhs_v, S05, K05, u, v, x0_u, x0_v, norm_matrix_u, norm_matrix_v)

	u .= x0_u
	v .= x0_v
	
	t = t + h

	return t
end

#Dense version of above function
@inline function m_step!(t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, h::Float64,
	K05::Array{Float64, N},S05::Array{Float64, N}, uforce::Array{Float64,N}, vforce::Array{Float64,N}, 
	rhs_u::Array{Float64,N}, rhs_v::Array{Float64,N}, x0_u, x0_v, norm_matrix_u, norm_matrix_v,
	linear_solver::lsolver_object) where N

	mul!(rhs_u, S05, u, h/2, 0)
	axpy!(1, u, rhs_u)
	mul!(rhs_u, K05, v, -h/2, 1)
	axpy!(h, uforce, rhs_u)

	mul!(rhs_v, S05, v, h/2, 0)
	axpy!(1, v, rhs_v)
	mul!(rhs_v, K05, u, h/2, 1)
	axpy!(h, vforce, rhs_v)
	
	linear_solver.solve(h, rhs_u, rhs_v, S05, K05, u, v, x0_u, x0_v, norm_matrix_u, norm_matrix_v)
    
	u .= x0_u
    v .= x0_v
	
	t = t + h

	return t
end
using SparseArrays
using LinearAlgebra

struct svparams
	K::Function
	S::Function
	In::Array{Float64,2}
end

# Julia versions prior to v"1.3.1" can't use LinearAlgebra's 5 argument mul!, routines
# included here for backwards compatability
if(VERSION < v"1.3.1")
    include("backwards_compat.jl")
end

# Method to generate the coefficients for the adjoint
# method for a given RK scheme (not PRK)
function adjoint_tableau(A,b,c)
	#If any weights vanish, exit immediately
	if(any(b.==0.0))
		println("Please feed me non-zero weights.")
		return
	end

	m,n = size(A)
	B = zeros(m,n)
	for j = 1:n
		for i = 1:m
			B[i,j] = b[j] - b[j]*A[j,i]/b[i]
		end
	end
	return B
end

# Force function
@inline function step(stepper::svparams, t::Float64, u::Array{Float64,N},
	v::Array{Float64,N}, h::Float64, uforce::Function, vforce::Function) where N
	uforce0 = uforce(t)
	vforce0 = vforce(t)

	vforce05 = vforce(t+0.5*h)
	vforce1 = vforce(t + h)
	uforce1 = uforce(t + h)

    t,u,v,v05 = step(stepper, t, u, v, h, uforce0, vforce05, uforce1)

	return t, u, v, v05
end

# wWthout forcing
@inline function step(stepper::svparams, t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, h::Float64) where N
	Ntot = size(u)
	vforce0 = zeros(Ntot)
	uforce05 = zeros(Ntot)
	vforce1 = zeros(Ntot)


    t,u,v, v05 = step(stepper, t, u, v, h, vforce0, uforce05, vforce1)

	return t, u, v, v05
end

# Forcing as array
@inline function step(stepper::svparams, t::Float64, u::Array{Float64,N}, v::Array{Float64,N},
	h::Float64, uforce0::Array{Float64,N}, vforce05::Array{Float64,N}, uforce1::Array{Float64,N}) where N
	S = stepper.S
	K = stepper.K
	In = stepper.In

	K0  = K(t)
	S0  = S(t)
	K05 = K(t +0.5*h)
	S05 = S(t + 0.5*h)
	K1  = K(t+h)
	S1  = S(t+h)

 	rhs = K05*u .+  S05*v + vforce05
	l1 = (In .-  0.5*h.*S05)\rhs
	v05 = (v .+ 0.5*h.*l1)
	kappa1 = S0*u .- K0*v05 + uforce0
	rhs = S1*(u .+ 0.5*h*kappa1) .- K1*v05 + uforce1
	kappa2 = (In .- (0.5*h).*S1)\rhs

	u = u .+ (0.5*h).*(kappa1 .+ kappa2)
	l2 = K05*u .+ S05*v05 + vforce05

	v = v .+ 0.5*h.*(l1 .+ l2)
	t = t + h
	return t, u, v, v05
end


###########################################################
# FMG
# Here we attempt an implementation of an explicit RK time
# integrator. Remember: return intermediate stages
@inline function explicit_step(stepper::svparams,t::Float64, u::Array{Float64,N},
	    v::Array{Float64,N}, h::Float64, uforce::Function, vforce::Function) where N
	S  = stepper.S
	K  = stepper.K
	In = stepper.In
	A  = stepper.A
	b  = stepper.b
	c  = stepper.c
	n_stages = stepper.n_stages

	# Arrays for storing intermediate stages
	u_stage = zeros(length(u),n_stages)
	k_mat_u = zeros(length(u),n_stages)
	v_stage = zeros(length(v),n_stages)
	k_mat_v = zeros(length(v),n_stages)

	St = S(t)
	Kt = K(t)

	# First stage is always the current solution
	k_mat_u[:,1] = St*u - Kt*v
	k_mat_v[:,1] = Kt*u + St*v
	u_stage[:,1] = u
	v_stage[:,1] = v
	for s = 2:n_stages
		# Build intermediate stage
		u_stage[:,s] = u
		v_stage[:,s] = v
		for ss = 1:s-1
			u_stage[:,s] += h*A[s,ss]*k_mat_u[:,ss]
			v_stage[:,s] += h*A[s,ss]*k_mat_v[:,ss]
		end

		St = S(t+c[s]*h)
		Kt = K(t+c[s]*h)
		k_mat_u[:,s] = St*u_stage[:,s] - Kt*v_stage[:,s]
		k_mat_v[:,s] = Kt*u_stage[:,s] + St*v_stage[:,s]
	end

	# Step forward in time
	for s = 1:n_stages
		u += h*b[s]*k_mat_u[:,s]
		v += h*b[s]*k_mat_v[:,s]
	end

	# FMG: Must return intermediate stages.
	t = t + h
	return t, u, v
end
###########################################################



# Forward gradient calculation.
@inline function step_fwdGrad!(t::Float64, u::Array{Float64,N}, v::Array{Float64,N},v05::Array{Float64,N},
	h::Float64, uforce0::Array{Float64,N}, vforce0::Array{Float64,N},uforce1::Array{Float64,N},
	vforce1::Array{Float64,N},K0::Array{Float64,N},
	S0::Array{Float64,N},K05::Array{Float64,N},S05::Array{Float64,N},
 	K1::Array{Float64,N},S1::Array{Float64,N},In::Array{Float64,N},
 	κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, 
 	ℓ₂::Array{Float64,N},rhs::Array{Float64,N},linear_solver::lsolver_object) where N

 	# rhs .= K05*u .+  S05*v .+ vforce0
 	LinearAlgebra.mul!(rhs,K05,u)
 	mul!(rhs,S05,v,1.0,1.0)
	LinearAlgebra.axpy!(1.0,vforce0,rhs)

	# ℓ₁     .= (In .-  0.5*h.*S05)\rhs
	linear_solver.solve(h,S05,rhs,v05,ℓ₁)

	# v05    .= v .+ 0.5*h*ℓ₁
	copy!(v05,v)
	LinearAlgebra.axpy!(0.5*h,ℓ₁,v05)

	# κ₁  .= S0*u .- K0*v05 .+ uforce0
	LinearAlgebra.mul!(κ₁,S0,u)
	mul!(κ₁,K0,v05,-1.0,1.0)
	LinearAlgebra.axpy!(1.0,uforce0,κ₁)

	# rhs    .= S1*(u .+ 0.5*h*κ₁) .- K1*v05
	LinearAlgebra.mul!(rhs, S1,u)
	mul!(rhs,S1,κ₁,0.5*h,1.0)
	mul!(rhs,K1,v05,-1.0,1.0)
	LinearAlgebra.axpy!(1.0,uforce1,rhs)

	# u      .= u .+ (0.5*h).*κ₁
	LinearAlgebra.axpy!(0.5*h,κ₁,u)

	# κ₂     .= (In .- (0.5*h)*S1)\rhs
	linear_solver.solve(h,S1,rhs,κ₁,κ₂)

	# u      .= u .+ (0.5*h).*κ₂
	LinearAlgebra.axpy!(0.5*h,κ₂,u)

	# ℓ₂  .= K05*u .+ S05*v05 .+ vforce1
	LinearAlgebra.mul!(ℓ₂,K05,u)
	mul!(ℓ₂,S05,v05,1.0,1.0)
	LinearAlgebra.axpy!(1.0,vforce1,ℓ₂)

	v      .= v .+ 0.5*h.*(ℓ₁ .+ ℓ₂)

	t      = t + h
end


# sparse version of step_fwdGrad above
@inline function step_fwdGrad!(t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, v05::Array{Float64,N},
	h::Float64, uforce0::Array{Float64,N}, vforce0::Array{Float64,N},uforce1::Array{Float64,N},
	vforce1::Array{Float64,N},K0::SparseMatrixCSC{Float64,Int64},
	S0::SparseMatrixCSC{Float64,Int64},K05::SparseMatrixCSC{Float64,Int64},S05::SparseMatrixCSC{Float64,Int64},
 	K1::SparseMatrixCSC{Float64,Int64},S1::SparseMatrixCSC{Float64,Int64},In::SparseMatrixCSC{Float64,Int64},
 	κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, 
 	ℓ₂::Array{Float64,N},rhs::Array{Float64,N},linear_solver::lsolver_object) where N

 	# rhs .= K05*u .+  S05*v .+ vforce0
 	LinearAlgebra.mul!(rhs,K05,u)
 	mul!(rhs,S05,v,1.0,1.0)
	LinearAlgebra.axpy!(1.0,vforce0,rhs)

	# ℓ₁     .= (In .-  0.5*h.*S05)\rhs
	linear_solver.solve(h,S05,rhs,v05,ℓ₁)

	# v05    .= v .+ 0.5*h*ℓ₁
	copy!(v05,v)
	LinearAlgebra.axpy!(0.5*h,ℓ₁,v05)

	# κ₁  .= S0*u .- K0*v05 .+ uforce0
	LinearAlgebra.mul!(κ₁,S0,u)
	mul!(κ₁,K0,v05,-1.0,1.0)
	LinearAlgebra.axpy!(1.0,uforce0,κ₁)

	# rhs    .= S1*(u .+ 0.5*h*κ₁) .- K1*v05
	LinearAlgebra.mul!(rhs, S1,u)
	mul!(rhs,S1,κ₁,0.5*h,1.0)
	mul!(rhs,K1,v05,-1.0,1.0)
	LinearAlgebra.axpy!(1.0,uforce1,rhs)

	# u      .= u .+ (0.5*h).*κ₁
	LinearAlgebra.axpy!(0.5*h,κ₁,u)

	# κ₂     .= (In .- (0.5*h)*S1)\rhs
	linear_solver.solve(h,S1,rhs,κ₁,κ₂)

	# u      .= u .+ (0.5*h).*κ₂
	LinearAlgebra.axpy!(0.5*h,κ₂,u)

	# ℓ₂  .= K05*u .+ S05*v05 .+ vforce1
	LinearAlgebra.mul!(ℓ₂,K05,u)
	mul!(ℓ₂,S05,v05,1.0,1.0)
	LinearAlgebra.axpy!(1.0,vforce1,ℓ₂)

	v      .= v .+ 0.5*h.*(ℓ₁ .+ ℓ₂)

	t      = t + h
end


# This is for the adjoint solve which contains forcing. We note that h is negative in this case.
@inline function step!(t::Float64, μ::Array{Float64,N}, ν::Array{Float64,N}, X::Array{Float64,N}, h::Float64,
  					uforce0::Array{Float64,N}, vforce0::Array{Float64,N}, uforce1::Array{Float64,N},
  					vforce1::Array{Float64,N},
  					K0::Array{Float64,N},S0::Array{Float64,N},K05::Array{Float64,N},S05::Array{Float64,N},
 					K1::Array{Float64,N},S1::Array{Float64,N},In::Array{Float64,N},
 					κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, ℓ₂::Array{Float64,N},
  					rhs::Array{Float64,N},linear_solver::lsolver_object) where N

    # rhs .= S0*μ .- K05*ν .+ uforce0
	LinearAlgebra.mul!(rhs,S0,μ)
	mul!(rhs,K05,ν,-1.0,1.0)
	LinearAlgebra.axpy!(1.0, uforce0, rhs)

	# solve linear system
	linear_solver.solve(h,S0,rhs,κ₁,κ₂)

	# μ  .= μ .+ 0.5*h.*κ₂
	LinearAlgebra.axpy!(0.5*h, κ₂, μ)
	
	# X   .= (μ .+ 0.5*h.*κ₂)
	copy!(X,μ)

	# ℓ₂  .= K0*X .+ S05*ν .+ vforce0
	LinearAlgebra.mul!(ℓ₂,K0,X)
	mul!(ℓ₂,S05,ν,1.0,1.0)
	LinearAlgebra.axpy!(1.0, vforce0, ℓ₂)


	# rhs .= S05*(ν .+ 0.5*h*ℓ₂) .+ K1*X .+ vforce1
	LinearAlgebra.mul!(rhs,S05,ν)
	mul!(rhs,S05,ℓ₂,0.5*h,1.0)
	mul!(rhs,K1,X,1.0,1.0)
	LinearAlgebra.axpy!(1.0, vforce1, rhs)

	# solve linear system
	linear_solver.solve(h,S05,rhs,κ₂,ℓ₁)

	ν  .= ν .+ (0.5*h).*(ℓ₂ .+ ℓ₁)

	# κ₁ .= -K05*ν .+ S1*X .+ uforce1
	LinearAlgebra.mul!(κ₁,S1,X)
	mul!(κ₁,K05,ν,-1.0,1.0)
	LinearAlgebra.axpy!(1.0, uforce1, κ₁)

	# μ  .= μ .+ 0.5*h.*κ₁
	LinearAlgebra.axpy!(0.5*h, κ₁, μ)

	t  = t + h
end

# sparse version of step function above
@inline function step!(t::Float64, μ::Array{Float64,N}, ν::Array{Float64,N}, X::Array{Float64,N}, h::Float64,
  					uforce0::Array{Float64,N}, vforce0::Array{Float64,N}, uforce1::Array{Float64,N},
  					vforce1::Array{Float64,N},
  					K0::SparseMatrixCSC{Float64,Int64},S0::SparseMatrixCSC{Float64,Int64},K05::SparseMatrixCSC{Float64,Int64},S05::SparseMatrixCSC{Float64,Int64},
 					K1::SparseMatrixCSC{Float64,Int64},S1::SparseMatrixCSC{Float64,Int64},In::SparseMatrixCSC{Float64,Int64},
					κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, ℓ₂::Array{Float64,N},
  					rhs::Array{Float64,N},linear_solver::lsolver_object) where N

    # rhs .= S0*μ .- K05*ν .+ uforce0
	LinearAlgebra.mul!(rhs,S0,μ)
	mul!(rhs,K05,ν,-1.0,1.0)
	LinearAlgebra.axpy!(1.0, uforce0, rhs)

	# solve linear system
	linear_solver.solve(h,S0,rhs,κ₁,κ₂)

	# μ  .= μ .+ 0.5*h.*κ₂
	LinearAlgebra.axpy!(0.5*h, κ₂, μ)
	
	# X   .= (μ .+ 0.5*h.*κ₂)
	copy!(X,μ)

	# ℓ₂  .= K0*X .+ S05*ν .+ vforce0
	LinearAlgebra.mul!(ℓ₂,K0,X)
	mul!(ℓ₂,S05,ν,1.0,1.0)
	LinearAlgebra.axpy!(1.0, vforce0, ℓ₂)


	# rhs .= S05*(ν .+ 0.5*h*ℓ₂) .+ K1*X .+ vforce1
	LinearAlgebra.mul!(rhs,S05,ν)
	mul!(rhs,S05,ℓ₂,0.5*h,1.0)
	mul!(rhs,K1,X,1.0,1.0)
	LinearAlgebra.axpy!(1.0, vforce1, rhs)

	# solve linear system
	linear_solver.solve(h,S05,rhs,κ₂,ℓ₁)

	ν  .= ν .+ (0.5*h).*(ℓ₂ .+ ℓ₁)

	# κ₁ .= -K05*ν .+ S1*X .+ uforce1
	LinearAlgebra.mul!(κ₁,S1,X)
	mul!(κ₁,K05,ν,-1.0,1.0)
	LinearAlgebra.axpy!(1.0, uforce1, κ₁)

	# μ  .= μ .+ 0.5*h.*κ₁
	LinearAlgebra.axpy!(0.5*h, κ₁, μ)

	t  = t + h


end







# This is for the adjoint solve Without forcing. We note that h is negative in this case.
@inline function step_no_forcing!(t::Float64, μ::Array{Float64,N}, ν::Array{Float64,N}, X::Array{Float64,N}, h::Float64,
  					K0::Array{Float64,N},S0::Array{Float64,N},K05::Array{Float64,N},S05::Array{Float64,N},
 					K1::Array{Float64,N},S1::Array{Float64,N},In::Array{Float64,N},
 					κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, ℓ₂::Array{Float64,N},
  					rhs::Array{Float64,N},linear_solver::lsolver_object) where N

    # rhs .= S0*μ .- K05*ν .+ uforce0
	LinearAlgebra.mul!(rhs,S0,μ)
	mul!(rhs,K05,ν,-1.0,1.0)

	# solve linear system
	linear_solver.solve(h,S0,rhs,κ₁,κ₂)

	# μ  .= μ .+ 0.5*h.*κ₂
	LinearAlgebra.axpy!(0.5*h, κ₂, μ)
	
	# X   .= (μ .+ 0.5*h.*κ₂)
	copy!(X,μ)

	# ℓ₂  .= K0*X .+ S05*ν .+ vforce0
	LinearAlgebra.mul!(ℓ₂,K0,X)
	mul!(ℓ₂,S05,ν,1.0,1.0)
	
	# rhs .= S05*(ν .+ 0.5*h*ℓ₂) .+ K1*X .+ vforce1
	LinearAlgebra.mul!(rhs,S05,ν)
	mul!(rhs,S05,ℓ₂,0.5*h,1.0)
	mul!(rhs,K1,X,1.0,1.0)
	
	# solve linear system
	linear_solver.solve(h,S05,rhs,κ₂,ℓ₁)

	ν  .= ν .+ (0.5*h).*(ℓ₂ .+ ℓ₁)

	# κ₁ .= -K05*ν .+ S1*X .+ uforce1
	LinearAlgebra.mul!(κ₁,S1,X)
	mul!(κ₁,K05,ν,-1.0,1.0)

	# μ  .= μ .+ 0.5*h.*κ₁
	LinearAlgebra.axpy!(0.5*h, κ₁, μ)

	t  = t + h
end

# sparse version of step function above
@inline function step_no_forcing!(t::Float64, μ::Array{Float64,N}, ν::Array{Float64,N}, X::Array{Float64,N}, h::Float64,
  					K0::SparseMatrixCSC{Float64,Int64},S0::SparseMatrixCSC{Float64,Int64},K05::SparseMatrixCSC{Float64,Int64},S05::SparseMatrixCSC{Float64,Int64},
 					K1::SparseMatrixCSC{Float64,Int64},S1::SparseMatrixCSC{Float64,Int64},In::SparseMatrixCSC{Float64,Int64},
					κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, ℓ₂::Array{Float64,N},
  					rhs::Array{Float64,N},linear_solver::lsolver_object) where N

    # rhs .= S0*μ .- K05*ν .+ uforce0
	LinearAlgebra.mul!(rhs,S0,μ)
	mul!(rhs,K05,ν,-1.0,1.0)
	
	# solve linear system
	linear_solver.solve(h,S0,rhs,κ₁,κ₂)

	# μ  .= μ .+ 0.5*h.*κ₂
	LinearAlgebra.axpy!(0.5*h, κ₂, μ)
	
	# X   .= (μ .+ 0.5*h.*κ₂)
	copy!(X,μ)

	# ℓ₂  .= K0*X .+ S05*ν .+ vforce0
	LinearAlgebra.mul!(ℓ₂,K0,X)
	mul!(ℓ₂,S05,ν,1.0,1.0)

	# rhs .= S05*(ν .+ 0.5*h*ℓ₂) .+ K1*X .+ vforce1
	LinearAlgebra.mul!(rhs,S05,ν)
	mul!(rhs,S05,ℓ₂,0.5*h,1.0)
	mul!(rhs,K1,X,1.0,1.0)

	# solve linear system
	linear_solver.solve(h,S05,rhs,κ₂,ℓ₁)

	ν  .= ν .+ (0.5*h).*(ℓ₂ .+ ℓ₁)

	# κ₁ .= -K05*ν .+ S1*X .+ uforce1
	LinearAlgebra.mul!(κ₁,S1,X)
	mul!(κ₁,K05,ν,-1.0,1.0)

	# μ  .= μ .+ 0.5*h.*κ₁
	LinearAlgebra.axpy!(0.5*h, κ₁, μ)

	t  = t + h

end








# FMG: This routine has been modified. This is for the forward evolution with no forcing.
@inline function step!(t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, v05::Array{Float64,N}, h::Float64,
  					  K0::Array{Float64,N},S0::Array{Float64,N},K05::Array{Float64,N},S05::Array{Float64,N},
  					  K1::Array{Float64,N},S1::Array{Float64,N},In::Array{Float64,N},
  					  κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, ℓ₂::Array{Float64,N},
  					  rhs::Array{Float64,N},linear_solver::lsolver_object) where N

 	# rhs    .= (K05*u .+  S05*v)
 	LinearAlgebra.mul!(rhs,K05,u)
 	mul!(rhs,S05,v,1.0,1.0)

	# ℓ₁     .= (In .-  0.5*h.*S05)\rhs
	linear_solver.solve(h,S05,rhs,v05,ℓ₁)
	
	# v05    .= v .+ 0.5*h*ℓ₁
	copy!(v05,v)
	LinearAlgebra.axpy!(0.5*h,ℓ₁,v05)

	# κ₁     .= S0*u .- K0*v05
	LinearAlgebra.mul!(κ₁,S0,u)
	mul!(κ₁,K0,v05,-1.0,1.0)

	# rhs    .= S1*(u .+ 0.5*h*κ₁) .- K1*v05
	LinearAlgebra.mul!(rhs, S1,u)
	mul!(rhs,S1,κ₁,0.5*h,1.0)
	mul!(rhs,K1,v05,-1.0,1.0)

	# u      .= u .+ (0.5*h).*κ₁
	LinearAlgebra.axpy!(0.5*h,κ₁,u)

	# κ₂     .= (In .- (0.5*h)*S1)\rhs
	linear_solver.solve(h,S1,rhs,κ₁,κ₂)

	# u      .= u .+ (0.5*h).*κ₂
	LinearAlgebra.axpy!(0.5*h,κ₂,u)

	# ℓ₂     .= K05*u .+ S05*v05
	LinearAlgebra.mul!(ℓ₂,K05,u)
	mul!(ℓ₂,S05,v05,1.0,1.0)

	v      .= v .+ 0.5*h.*(ℓ₁ .+ ℓ₂)

	t      = t + h
	return t
end

# sparse version of step function above
@inline function step!(t::Float64, u::Array{Float64,N}, v::Array{Float64,N}, v05::Array{Float64,N}, h::Float64,
  					  K0::SparseMatrixCSC{Float64,Int64},S0::SparseMatrixCSC{Float64,Int64},K05::SparseMatrixCSC{Float64,Int64},S05::SparseMatrixCSC{Float64,Int64},
  					  K1::SparseMatrixCSC{Float64,Int64},S1::SparseMatrixCSC{Float64,Int64},In::SparseMatrixCSC{Float64,Int64},
  					  κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, ℓ₂::Array{Float64,N},
  					  rhs::Array{Float64,N},linear_solver::lsolver_object) where N

 	# rhs    .= (K05*u .+  S05*v)
 	LinearAlgebra.mul!(rhs,K05,u)
 	mul!(rhs,S05,v,1.0,1.0)

	# ℓ₁     .= (In .-  0.5*h.*S05)\rhs
	linear_solver.solve(h,S05,rhs,v05,ℓ₁)

	# v05    .= v .+ 0.5*h*ℓ₁
	copy!(v05,v)
	LinearAlgebra.axpy!(0.5*h,ℓ₁,v05)

	# κ₁     .= S0*u .- K0*v05
	LinearAlgebra.mul!(κ₁,S0,u)
	mul!(κ₁,K0,v05,-1.0,1.0)

	# rhs    .= S1*(u .+ 0.5*h*κ₁) .- K1*v05
	LinearAlgebra.mul!(rhs, S1,u)
	mul!(rhs,S1,κ₁,0.5*h,1.0)
	mul!(rhs,K1,v05,-1.0,1.0)

	# u      .= u .+ (0.5*h).*κ₁
	LinearAlgebra.axpy!(0.5*h,κ₁,u)

	# κ₂     .= (In .- (0.5*h)*S1)\rhs
	linear_solver.solve(h,S1,rhs,κ₁,κ₂)

	# u      .= u .+ (0.5*h).*κ₂
	LinearAlgebra.axpy!(0.5*h,κ₂,u)

	# ℓ₂     .= K05*u .+ S05*v05
	LinearAlgebra.mul!(ℓ₂,K05,u)
	mul!(ℓ₂,S05,v05,1.0,1.0)

	v      .= v .+ 0.5*h.*(ℓ₁ .+ ℓ₂)

	t      = t + h
	return t
end

function stepseparable(stepper::svparams,u,v,t,h)
  if S(stepper.t) != 0
  	error("S has to be zero for separble time-stepping to ework")
  end

	S = stepper.S
	K = stepper.K
	In = stepper.In
	h = stepper.h
	uforce = stepper.uforce
	vforce = stepper.vforce

	l1 = (K(t)*u .+ vforce(t))
	kappa1 = - K(t + 0.5*h)*(v .+ 0.5*h.*l1) .+ uforce(t + 0.5*h)
	kappa2 = kappa1

	u = u .+ (h/2).*(kappa1 .+ kappa2)
	l2 = K(t +h)*u  .+ vforce(t + h)

	v = v .+ 0.5*h.*(l1 .+ l2)
	t = t + h

	return t, u, v
end


"""
    gamma, used_stages = getgamma(order, stages)

Obtain step size coefficients (gamma) for the composition method with the given
order and number of stages. 
"""
function getgamma(order::Int64, stages::Int64=0)
    # Check if given order and stages are implemented, round stages to nearest valid value if not
    valid_orders = (2,4,6,8,10)
    valid_stages = ((1), (3,5), (7,9), (15, 17), (35))

    if !(order in valid_orders)
        throw(DomainError(order, "Provided order $order invalid. Please use one of the following values: $(valid_orders)."))
    end
    order_index = findfirst(x -> x==order, valid_orders)

    # If stages provided but invalid, use nearest valid number of stages
    if stages == 0
        used_stages = valid_stages[order_index][end]
        #@warn "WARNING: Default value 0 provided for stages. Using $(used_stages)-stage, order $order method."
    elseif !(stages in valid_stages[order_index])
        stages_index = findmin(abs.(valid_stages[order_index] .- stages))[2]
        used_stages = valid_stages[order_index][stages_index]
        @warn "WARNING: invalid number of stages $stages specified for compositional method of order $(order)! Using closest valid number of stages $used_stages instead."
    else
        used_stages = stages
    end

    # Calculate gamma 
    if order == 2 # 2nd order basic Stormer-Verlet
        if used_stages == 1
          gamma = [1.0]
        end
    elseif order == 4
        if used_stages==3 # 4th order Composition of Stormer-Verlet methods
            gamma = zeros(used_stages)
            gamma[1] = gamma[3] = 1/(2 - 2^(1/3))
            gamma[2] = -2^(1/3)*gamma[1]
        elseif used_stages==5
            gamma = zeros(used_stages)
            gamma[1] = gamma[2] = gamma[4] = gamma[5] = 1/(4-4^(1/3))
            gamma[3] = -4^(1/3)*gamma[1]
        end
    elseif order == 6
        if used_stages==7  # Yoshida (1990) 6th order, 7 stage method
            gamma = zeros(used_stages)
            gamma[1] = gamma[7] = 0.78451361047755726381949763
            gamma[2] = gamma[6] = 0.23557321335935813368479318
            gamma[3] = gamma[5] = -1.17767998417887100694641568
            gamma[4] = 1.31518632068391121888424973
        elseif used_stages==9 # Kahan + Li 6th order, 9 stage method
            gamma = zeros(used_stages)
            gamma[1]= gamma[9]= 0.39216144400731413927925056
            gamma[2]= gamma[8]= 0.33259913678935943859974864
            gamma[3]= gamma[7]= -0.70624617255763935980996482
            gamma[4]= gamma[6]= 0.08221359629355080023149045
            gamma[5]= 0.79854399093482996339895035
        end
    elseif order == 8
        if used_stages == 15
            gamma = zeros(used_stages)
            gamma[1] = gamma[15] = 0.74167036435061295344822780
            gamma[2] = gamma[14] = -0.40910082580003159399730010
            gamma[3] = gamma[13] = 0.19075471029623837995387626
            gamma[4] = gamma[12] = -0.57386247111608226665638773
            gamma[5] = gamma[11] = 0.29906418130365592384446354
            gamma[6] = gamma[10] = 0.33462491824529818378495798
            gamma[7] = gamma[9]  = 0.31529309239676659663205666
            gamma[8] = -0.79688793935291635401978884
        elseif used_stages==17
            gamma = zeros(used_stages)
            gamma[1] = gamma[17] = 0.13020248308889008087881763
            gamma[2] = gamma[16] = 0.56116298177510838456196441
            gamma[3] = gamma[15] = -0.38947496264484728640807860
            gamma[4] = gamma[14] = 0.15884190655515560089621075
            gamma[5] = gamma[13] = -0.39590389413323757733623154
            gamma[6] = gamma[12] = 0.18453964097831570709183254
            gamma[7] = gamma[11] = 0.25837438768632204729397911
            gamma[8] = gamma[10] = 0.29501172360931029887096624
            gamma[9] = -0.60550853383003451169892108  
        end
    elseif order==10
        if used_stages==35
            gamma = zeros(used_stages)
            gamma[1]  = gamma[35] = 0.07879572252168641926390768
            gamma[2]  = gamma[34] = 0.31309610341510852776481247
            gamma[3]  = gamma[33] = 0.02791838323507806610952027
            gamma[4]  = gamma[32] = -0.22959284159390709415121340
            gamma[5]  = gamma[31] = 0.13096206107716486317465686
            gamma[6]  = gamma[30] = -0.26973340565451071434460973
            gamma[7]  = gamma[29] = 0.07497334315589143566613711
            gamma[8]  = gamma[28] = 0.11199342399981020488957508
            gamma[9]  = gamma[27] = 0.36613344954622675119314812
            gamma[10] = gamma[26] = -0.39910563013603589787862981
            gamma[11] = gamma[25] = 0.10308739852747107731580277
            gamma[12] = gamma[24] = 0.41143087395589023782070412
            gamma[13] = gamma[23] = -0.00486636058313526176219566
            gamma[14] = gamma[22] = -0.39203335370863990644808194
            gamma[15] = gamma[21] = 0.05194250296244964703718290
            gamma[16] = gamma[20] = 0.05066509075992449633587434
            gamma[17] = gamma[19] = 0.04967437063972987905456880
            gamma[18] = 0.04931773575959453791768001
        end
    end

    return gamma, used_stages
end



# Routine to advance the solution by one time step with a second order Magnus integrator. Note that 
# we use this in a very special case where we have an evolution matrix that is 
# real, orthogonal, and skew-symmetric which gives a single complex conjugate 
# pair of eigenvalues of ±i. This allows an incredibly simplified approach 
# to compute the matrix exponential.
@inline function magnus(stepper::svparams, t::Float64, u::Array{Float64,N}, 
	v::Array{Float64,N}, h::Float64, uforce::Function, vforce::Function) where N
	uforce = uforce(t+0.5*h)
	vforce = vforce(t+0.5*h)

	S = stepper.S
	K = stepper.K
	In = stepper.In

	# Only evaluate the matrix at half-steps
	K = K(t +0.5*h)
	S = S(t + 0.5*h)

	# Single application of matrix
	# u1 = S*u - K*v
	# v1 = K*u + S*v
	# u += sin(h)*u1 + (1.0-cos(h))*(S*u1-K*v1)
	# v += sin(h)*v1 + (1.0-cos(h))*(K*u1+S*v1)

	# # Add in forcing 
	# u += sin(h)*uforce + (1.0-cos(h))*(S*uforce - K*vforce)
	# v += sin(h)*vforce + (1.0-cos(h))*(K*uforce + S*vforce)
	# t = t + h

	# Brute force 
	A = [S -K;K S]
	E = exp(h*A)
	Ndeg = length(u)
	tmp = E*[u;v] + inv(A)*(E - [In zeros(Ndeg,Ndeg);zeros(Ndeg,Ndeg) In])*[uforce;vforce]

	u = tmp[1:length(u)]
	v = tmp[length(u)+1:end]
	t = t + h
	return t, u, v
end


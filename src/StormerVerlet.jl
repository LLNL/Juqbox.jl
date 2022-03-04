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
@inline function step_fwdGrad!(t::Float64, nNeumann::Int64, u::Array{Float64,N}, v::Array{Float64,N},v05::Array{Float64,N},
	h::Float64, uforce0::Array{Float64,N}, vforce0::Array{Float64,N},uforce1::Array{Float64,N},
	vforce1::Array{Float64,N},K0::Array{Float64,N},
	S0::Array{Float64,N},K05::Array{Float64,N},S05::Array{Float64,N},
 	K1::Array{Float64,N},S1::Array{Float64,N},In::Array{Float64,N},
 	κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, 
 	ℓ₂::Array{Float64,N},rhs::Array{Float64,N}) where N

 	# rhs .= K05*u .+  S05*v .+ vforce0
 	LinearAlgebra.mul!(rhs,K05,u)
 	mul!(rhs,S05,v,1.0,1.0)
	LinearAlgebra.axpy!(1.0,vforce0,rhs)

	# ℓ₁     .= (In .-  0.5*h.*S05)\rhs
	neumann!(nNeumann,h,S05,rhs,v05,ℓ₁)

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
	neumann!(nNeumann,h,S1,rhs,κ₁,κ₂)

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
@inline function step_fwdGrad!(t::Float64, nNeumann::Int64, u::Array{Float64,N}, v::Array{Float64,N}, v05::Array{Float64,N},
	h::Float64, uforce0::Array{Float64,N}, vforce0::Array{Float64,N},uforce1::Array{Float64,N},
	vforce1::Array{Float64,N},K0::SparseMatrixCSC{Float64,Int64},
	S0::SparseMatrixCSC{Float64,Int64},K05::SparseMatrixCSC{Float64,Int64},S05::SparseMatrixCSC{Float64,Int64},
 	K1::SparseMatrixCSC{Float64,Int64},S1::SparseMatrixCSC{Float64,Int64},In::SparseMatrixCSC{Float64,Int64},
 	κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, 
 	ℓ₂::Array{Float64,N},rhs::Array{Float64,N}) where N

 	# rhs .= K05*u .+  S05*v .+ vforce0
 	LinearAlgebra.mul!(rhs,K05,u)
 	mul!(rhs,S05,v,1.0,1.0)
	LinearAlgebra.axpy!(1.0,vforce0,rhs)

	# ℓ₁     .= (In .-  0.5*h.*S05)\rhs
	neumann!(nNeumann,h,S05,rhs,v05,ℓ₁)

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
	neumann!(nNeumann,h,S1,rhs,κ₁,κ₂)

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
@inline function step!(t::Float64, nNeumann::Int64, μ::Array{Float64,N}, ν::Array{Float64,N}, X::Array{Float64,N}, h::Float64,
  					uforce0::Array{Float64,N}, vforce0::Array{Float64,N}, uforce1::Array{Float64,N},
  					vforce1::Array{Float64,N},
  					K0::Array{Float64,N},S0::Array{Float64,N},K05::Array{Float64,N},S05::Array{Float64,N},
 					K1::Array{Float64,N},S1::Array{Float64,N},In::Array{Float64,N},
 					κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, ℓ₂::Array{Float64,N},
  					rhs::Array{Float64,N}) where N

    # rhs .= S0*μ .- K05*ν .+ uforce0
	LinearAlgebra.mul!(rhs,S0,μ)
	mul!(rhs,K05,ν,-1.0,1.0)
	LinearAlgebra.axpy!(1.0, uforce0, rhs)

	# Neumann Series to invert linear system
	neumann!(nNeumann,h,S0,rhs,κ₁,κ₂)

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

	# Neumann Series to invert linear system
	neumann!(nNeumann,h,S05,rhs,κ₂,ℓ₁)

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
@inline function step!(t::Float64, nNeumann::Int64, μ::Array{Float64,N}, ν::Array{Float64,N}, X::Array{Float64,N}, h::Float64,
  					uforce0::Array{Float64,N}, vforce0::Array{Float64,N}, uforce1::Array{Float64,N},
  					vforce1::Array{Float64,N},
  					K0::SparseMatrixCSC{Float64,Int64},S0::SparseMatrixCSC{Float64,Int64},K05::SparseMatrixCSC{Float64,Int64},S05::SparseMatrixCSC{Float64,Int64},
 					K1::SparseMatrixCSC{Float64,Int64},S1::SparseMatrixCSC{Float64,Int64},In::SparseMatrixCSC{Float64,Int64},
					κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, ℓ₂::Array{Float64,N},
  					rhs::Array{Float64,N}) where N

    # rhs .= S0*μ .- K05*ν .+ uforce0
	LinearAlgebra.mul!(rhs,S0,μ)
	mul!(rhs,K05,ν,-1.0,1.0)
	LinearAlgebra.axpy!(1.0, uforce0, rhs)

	# Neumann Series to invert linear system
	neumann!(nNeumann,h,S0,rhs,κ₁,κ₂)

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

	# Neumann Series to invert linear system
	neumann!(nNeumann,h,S05,rhs,κ₂,ℓ₁)

	ν  .= ν .+ (0.5*h).*(ℓ₂ .+ ℓ₁)

	# κ₁ .= -K05*ν .+ S1*X .+ uforce1
	LinearAlgebra.mul!(κ₁,S1,X)
	mul!(κ₁,K05,ν,-1.0,1.0)
	LinearAlgebra.axpy!(1.0, uforce1, κ₁)

	# μ  .= μ .+ 0.5*h.*κ₁
	LinearAlgebra.axpy!(0.5*h, κ₁, μ)

	t  = t + h


end


# FMG: This routine has been modified. This is for the forward evolution with no forcing.
@inline function step!(t::Float64, nNeumann::Int64, u::Array{Float64,N}, v::Array{Float64,N}, v05::Array{Float64,N}, h::Float64,
  					  K0::Array{Float64,N},S0::Array{Float64,N},K05::Array{Float64,N},S05::Array{Float64,N},
  					  K1::Array{Float64,N},S1::Array{Float64,N},In::Array{Float64,N},
  					  κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, ℓ₂::Array{Float64,N},
  					  rhs::Array{Float64,N}) where N

 	# rhs    .= (K05*u .+  S05*v)
 	LinearAlgebra.mul!(rhs,K05,u)
 	mul!(rhs,S05,v,1.0,1.0)

	# ℓ₁     .= (In .-  0.5*h.*S05)\rhs
	neumann!(nNeumann,h,S05,rhs,v05,ℓ₁)
	
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
	neumann!(nNeumann,h,S1,rhs,κ₁,κ₂)

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
@inline function step!(t::Float64, nNeumann::Int64, u::Array{Float64,N}, v::Array{Float64,N}, v05::Array{Float64,N}, h::Float64,
  					  K0::SparseMatrixCSC{Float64,Int64},S0::SparseMatrixCSC{Float64,Int64},K05::SparseMatrixCSC{Float64,Int64},S05::SparseMatrixCSC{Float64,Int64},
  					  K1::SparseMatrixCSC{Float64,Int64},S1::SparseMatrixCSC{Float64,Int64},In::SparseMatrixCSC{Float64,Int64},
  					  κ₁::Array{Float64,N}, κ₂::Array{Float64,N},ℓ₁::Array{Float64,N}, ℓ₂::Array{Float64,N},
  					  rhs::Array{Float64,N}) where N

 	# rhs    .= (K05*u .+  S05*v)
 	LinearAlgebra.mul!(rhs,K05,u)
 	mul!(rhs,S05,v,1.0,1.0)

	# ℓ₁     .= (In .-  0.5*h.*S05)\rhs
	neumann!(nNeumann,h,S05,rhs,v05,ℓ₁)

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
	neumann!(nNeumann,h,S1,rhs,κ₁,κ₂)

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

function getgamma(order,stages = [])
 if order == 2	# 2nd order basic verlet
	stages = 1
    gamma = [1]
  elseif order == 4 # 4th order Composition of Stromer-Verlet methods
    stages=3
    gamma = zeros(stages)
    gamma[1] = gamma[3] = 1/(2 - 2^(1/3))
    gamma[2] = -2^(1/3)*gamma[1]
  elseif order == 6 # Yoshida (1990) 6th order, 7 stage method
    if stages==7
      gamma = zeros(stages)
      gamma[2] = gamma[6] = 0.23557321335935813368479318
      gamma[1] = gamma[7] = 0.78451361047755726381949763
      gamma[3] = gamma[5] = -1.17767998417887100694641568
      gamma[4] = 1.31518632068391121888424973
    else # Kahan + Li 6th order, 9 stage method
      stages=9;
      gamma = zeros(stages)
      gamma[1]= gamma[9]= 0.39216144400731413927925056
      gamma[2]= gamma[8]= 0.33259913678935943859974864
      gamma[3]= gamma[7]= -0.70624617255763935980996482
      gamma[4]= gamma[6]= 0.08221359629355080023149045
      gamma[5]= 0.79854399093482996339895035
    end
   end

   return gamma, stages
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


@inline function neumann!(nterms::Int64, h::Float64, S::SparseMatrixCSC{Float64,Int64}, 
						 B::Array{Float64,N}, T::Array{Float64,N}, X::Array{Float64,N}) where N
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

@inline function neumann!(nterms::Int64, h::Float64, S::Array{Float64,N}, B::Array{Float64,N}, 
						T::Array{Float64,N}, X::Array{Float64,N}) where N
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


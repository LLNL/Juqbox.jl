"""
    spar = splineparams(T, D1, Nseg, pcof)

Constructor for struct splineparams, which sets up the parameters for a regular B-spline function
(without carrier waves).

# Arguments
- `T:: Float64`: Duration of spline function
- `D1:: Int64`: Number of basis functions in each spline
- `Nseg:: Int64`:  Number of splines (real, imaginary, different ctrl func)
- `pcof:: Array{Float64, 1}`: Coefficient vector. Must have D1*Nseg elements

# External links
* [Spline Wavelet](https://en.wikipedia.org/wiki/Spline_wavelet#Quadratic_B-spline) on Wikipedia.
"""
struct splineparams 
    T::Float64
    D1::Int64 # Number of coefficients per spline
    tcenter::Array{Float64,1}
    dtknot::Float64
    pcof::Array{Float64,1} # pcof should have D1*Nseg elements
    Nseg::Int64 # Number of segments (real, imaginary, different ctrl func)
    Ncoeff:: Int64 # Total number of coefficients

# new, simplified constructor
    function splineparams(T, D1, Nseg, pcof)
        dtknot = T/(D1 -2)
        tcenter = dtknot.*(collect(1:D1) .- 1.5)
        new(T, D1, tcenter, dtknot, pcof, Nseg, Nseg*D1)
    end

end

# bspline2: Evaluate quadratic bspline function
"""
    f = bspline2(t, splineparam, splinefunc)

Evaluate a B-spline function. See also the `splineparams` constructor.

# Arguments
- `t::Float64`: Evaluate spline at parameter t ∈ [0, param.T]
- `param::splineparams`: Parameters for the spline
- `splinefunc::Int64`: Spline function index ∈ [0, param.Nseg-1]
"""
@inline function bspline2(t::Float64, param::splineparams, splinefunc::Int64)
  f = 0.0

  dtknot = param.dtknot
  width = 3*dtknot

  offset = splinefunc*param.D1

  k = max.(3, ceil.(Int64,t./dtknot + 2)) # Unsure if this line does what it is supposed to
  k = min.(k, param.D1)

  # 1st segment of nurb k
  tc = param.tcenter[k]
  tau = (t .- tc)./width
  f = f + param.pcof[offset+k] * (9/8 .+ 4.5*tau + 4.5*tau^2) # test to remove square for extra speed

  # 2nd segment of nurb k-1
  tc = param.tcenter[k-1]
  tau = (t - tc)./width
  f = f .+ param.pcof[offset+k.-1] .* (0.75 - 9 *tau^2)

  # 3rd segment of nurb k-2
  tc = param.tcenter[k.-2]
  tau = (t .- tc)./width
  f = f + param.pcof[offset+k.-2] * (9/8 - 4.5*tau + 4.5*tau.^2)
end


"""
    g = gradbspline2(t, param, splinefunc)

Evaluate the gradient of a spline function with respect to all coefficient.
NOTE: the index of the spline functions are 0-based. For a set of 
coupled controls, mod(`splinefunc`,2)=0 corresponds to ∇ p_j(t) and mod(`splinefunc`,2) = 1 
corresponds to ∇ q_j(t), where j = div(splinefunc,2).

# Arguments
- `t::Float64`: Evaluate spline at parameter t ∈ [0, param.T]
- `param::splineparams`: Spline parameter object
- `splinefunc::Int64`: Spline function index ∈ [0, param.Nseg-1]
"""
@inline function gradbspline2(t::Float64,param::splineparams, splinefunc::Int64)

# NOTE: param.Nseg used to be '2'
  g = zeros(param.Nseg*param.D1) # real and imag parts for both f and g 

  dtknot = param.dtknot
  width = 3*dtknot

  offset = splinefunc*param.D1
  
  k = max.(3, ceil.(Int64,t./dtknot .+ 2)) # t_knot(k-1) < t <= t_knot(k), but t=0 needs to give k=3
  k = min.(k, param.D1) # protect agains roundoff that sometimes makes t/dt > N_nurbs-2

  #1st segment of nurb k
  tc = param.tcenter[k]
  tau = (t .- tc)./width
  g[offset .+ k] = (9/8 .+ 4.5.*tau .+ 4.5.*tau.^2);

  #2nd segment of nurb k-1
  tc = param.tcenter[k.-1]
  tau = (t .- tc)./width
  g[offset .+ k.-1] = (0.75 .- 9 .*tau.^2)

  # 3rd segment og nurb k-2
  tc = param.tcenter[k.-2]
  tau = (t .- tc)./width
  g[offset .+ k.-2] = (9/8 .- 4.5.*tau .+ 4.5.*tau.^2);
  return g
end


"""
    bcpar = bcparams(T, D1, Ncoupled, Nunc, omega, pcof)

General constructor of struct bcparams for setting up B-splines with carrier waves.

    bcpar = bcparams(T, D1, omega, pcof)

Simplified constructor for the case when there are no uncoupled controls and `Ncoupled = size(omega,1)`.

# Arguments
- `T:: Float64`: Duration of spline function
- `D1:: Int64`: Number of basis functions in each segment
- `Ncoupled::Int64`: Number of coupled controls in the simulation
- `Nunc::Int64`: Number of uncoupled controls in the simulation
- `omega::Array{Float64,2}`: Carrier wave frequencies
- `pcof:: Array{Float64, 1}`: Coefficient vector. Must have D1*Nseg elements

# First dimensions of the `omega` array:
- With no uncoupled controls, `Nunc=0` and `size(omega,1) = Ncoupled`.
- With uncoupled controls, `Nunc > 0` and `size(omega,1) = Ncoupled + Nunc`.

# Second dimension of the `omega` array:
- `size(omega, 2) = Nfreq`

# Ordering of the `pcof` array:
First consider the case without uncoupled control functions, `Nunc = 0`: 
Then the `pcof` array then has `2*Ncoupled*Nfreq*D1` elements. 
Each `ctrl ∈ [1,Ncoupled]` and `freq ∈ [1,Nfreq]` corresponds to `D1` elements in 
the `pcof` vector. For the case `Ncoupled = 2` and `Nfreq = 2`, the elements are ordered according to

| ctrl    | freq    | α_1  | α_2 |
| ------- | -------- | -------- | -------|
| 1      | 1        | 1:D1             | D1+1:2 D1 |
| 1      | 2        | 2 D1+1: 3 D1 | 3 D1+1:4 D1 |
| 2      | 1        | 4 D1+1: 5 D1 | 5 D1+1:6 D1 |
| 2      | 2        | 6 D1+1: 7 D1 | 7 D1+1: 8D1 |

If there are uncoupled controls, `Nunc > 0`, the `pcof` array should have `(2*Ncoupled + Nunc)*Nfreq*D1` elements. 
The last `Nunc*Nfreq*D1` elements correspond to the uncoupled control functions and are ordered in a corresponding way.

# External links
* [Spline Wavelet](https://en.wikipedia.org/wiki/Spline_wavelet#Quadratic_B-spline) on Wikipedia.
"""
struct bcparams
    T ::Float64
    D1::Int64 # number of B-spline coefficients per control function
    om::Array{Float64,2} #Carrier wave frequencies [rad/s], size Nfreq
    tcenter::Array{Float64,1}
    dtknot::Float64
    pcof::Array{Float64,1} # coefficients for all 2*Ncoupled splines, size Ncoupled*D1*Nfreq*2 (*2 because of sin/cos)
    Nfreq::Int64 # Number of frequencies
    Ncoeff:: Int64 # Total number of coefficients
    Ncoupled::Int64 # Number of coupled B-splines functions
    Nunc::Int64 # Number of uncoupled B-spline functions

    # # simplified constructor (assumes no uncoupled terms)
    # function bcparams(T::Float64, D1::Int64, omega::Array{Float64,2}, pcof::Array{Float64,1})
    #     dtknot = T/(D1 -2)
    #     tcenter = dtknot.*(collect(1:D1) .- 1.5)
    #     Ncoupled = size(omega,1) # should check that Ncoupled >=1
    #     Nfreq = size(omega,2)
    #     Nunc = 0
    #     nCoeff = Nfreq*D1*(2*Ncoupled + Nunc)
    #     if(nCoeff != length(pcof))
    #         throw(DimensionMismatch("Inconsistent number of coefficients and size of parameter vector (nCoeff ≠ length(pcof)."))
    #     end
    #     new(T, D1, omega, tcenter, dtknot, pcof, Nfreq, nCoeff, Ncoupled, Nunc)
    # end

    # New constructor to allow defining number of symmetric Hamiltonian terms
    function bcparams(T::Float64, D1::Int64, Ncoupled::Int64, Nunc::Int64, omega::Array{Float64,2}, pcof::Array{Float64,1})
        dtknot = T/(D1 -2)
        tcenter = dtknot.*(collect(1:D1) .- 1.5)
        Nfreq = size(omega,2)
        nCoeff = Nfreq*D1*(2*Ncoupled + Nunc)
        if(nCoeff != length(pcof))
            throw(DimensionMismatch("Inconsistent number of coefficients and size of parameter vector (nCoeff ≠ length(pcof)."))
        end
        new(T, D1, omega, tcenter, dtknot, pcof, Nfreq, nCoeff, Ncoupled, Nunc)
    end

end

# simplified constructor (assumes no uncoupled terms)
function bcparams(T::Float64, D1::Int64, omega::Array{Float64,2}, pcof::Array{Float64,1})
  dtknot = T/(D1 -2)
  tcenter = dtknot.*(collect(1:D1) .- 1.5)
  Ncoupled = size(omega,1) # should check that Ncoupled >=1
  Nfreq = size(omega,2)
  Nunc = 0
  nCoeff = Nfreq*D1*(2*Ncoupled + Nunc)
  if(nCoeff != length(pcof))
    throw(DimensionMismatch("Inconsistent number of coefficients and size of parameter vector (nCoeff ≠ length(pcof)."))
  end
  bcparams(T, D1, Ncoupled, Nunc, omega, pcof)
end

"""
    f = bcarrier2(t, params, func)

Evaluate a B-spline function with carrier waves. See also the `bcparams` constructor.

# Arguments
- `t::Float64`: Evaluate spline at parameter t ∈ [0, param.T]
- `param::params`: Parameters for the spline
- `func::Int64`: Spline function index ∈ [0, param.Nseg-1]
"""
@inline function bcarrier2(t::Float64, params::bcparams, func::Int64)
    # for a single oscillator, func=0 corresponds to p(t) and func=1 to q(t)
    # in general, 0 <= func < 2*Ncoupled + Nunc

    # compute basic offset: func 0 and 1 use the same spline coefficients, but combined in a different way
    osc = div(func, 2) # osc is base 0; 0<= osc < Ncoupled
    q_func = func % 2 # q_func = 0 for p and q_func=1 for q
    
    f = 0.0 # initialize
    
    dtknot = params.dtknot
    width = 3*dtknot
    
    k = max.(3, ceil.(Int64,t./dtknot + 2)) # pick out the index of the last basis function corresponding to t
    k = min.(k, params.D1) #  Make sure we don't access outside the array
    
    if(func < 2*params.Ncoupled)
        # Coupled controls
        @fastmath @inbounds @simd for freq in 1:params.Nfreq
            fbs1 = 0.0 # initialize
            fbs2 = 0.0 # initialize
            # offset in parameter array (osc = 0,1,2,...
            # Vary freq first, then osc
            offset1 = 2*osc*params.Nfreq*params.D1 + (freq-1)*2*params.D1
            offset2 = 2*osc*params.Nfreq*params.D1 + (freq-1)*2*params.D1 + params.D1

            # 1st segment of nurb k
            tc = params.tcenter[k]
            tau = (t .- tc)./width
            fbs1 += params.pcof[offset1+k] * (9/8 .+ 4.5*tau + 4.5*tau^2)
            fbs2 += params.pcof[offset2+k] * (9/8 .+ 4.5*tau + 4.5*tau^2)
            
            # 2nd segment of nurb k-1
            tc = params.tcenter[k-1]
            tau = (t - tc)./width
            fbs1 += params.pcof[offset1+k.-1] .* (0.75 - 9 *tau^2)
            fbs2 += params.pcof[offset2+k.-1] .* (0.75 - 9 *tau^2)
            
            # 3rd segment of nurb k-2
            tc = params.tcenter[k.-2]
            tau = (t .- tc)./width
            fbs1 += params.pcof[offset1+k-2] * (9/8 - 4.5*tau + 4.5*tau.^2)
            fbs2 += params.pcof[offset2+k-2] * (9/8 - 4.5*tau + 4.5*tau.^2)

            #    end # for carrier phase
            # p(t)
            if q_func==1
                f += fbs1 * sin(params.om[osc+1,freq]*t) + fbs2 * cos(params.om[osc+1,freq]*t) # q-func
            else
                f += fbs1 * cos(params.om[osc+1,freq]*t) - fbs2 * sin(params.om[osc+1,freq]*t) # p-func
            end
        end # for freq
    else 

      # uncoupled control
      @fastmath @inbounds @simd for freq in 1:params.Nfreq
        fbs = 0.0 # initialize
        
        # offset to uncoupled parameters
        # offset = (2*params.Ncoupled)*params.D1*params.Nfreq + (fun - 2*params.Ncoupled)*params.D1*params.Nfreq  +  (freq-1)*params.D1
        offset = func*params.D1*params.Nfreq  +  (freq-1)*params.D1

        # 1st segment of nurb k
        if k <= params.D1  # could do the if statement outside the for-loop instead
          tc = params.tcenter[k]
          tau = (t .- tc)./width
          fbs += params.pcof[offset+k] * (9/8 .+ 4.5*tau + 4.5*tau^2)
        end
      
        # 2nd segment of nurb k-1
        if k >= 2 && k <= params.D1 + 1
          tc = params.tcenter[k-1]
          tau = (t - tc)./width
          fbs += params.pcof[offset+k.-1] .* (0.75 - 9 *tau^2)
        end
        
        # 3rd segment of nurb k-2
        if k >= 3 && k <= params.D1 + 2
          tc = params.tcenter[k.-2]
          tau = (t .- tc)./width
          fbs += params.pcof[offset+k-2] * (9/8 - 4.5*tau + 4.5*tau.^2)
        end

        # FMG: For now this would assume that every oscillator would have 
        # an uncoupled term if uncoupled terms exist in the problem. Perhaps 
        # update parameter struct to track terms if they are only present on some 
        # oscillators?
        ind = func - 2*params.Ncoupled
        f += fbs * cos(params.om[ind+1,freq]*t) # spl is 0-based
      end 


    end
    return f
end


"""
    gradbcarrier2!(t, params, func, g) -> g
Evaluate the gradient of a control function with respect to all coefficient.

NOTE: the index of the control functions is 0-based. For a set of 
coupled controls, mod(`func`,2)=0 corresponds to ∇ p_j(t) and mod(`func`,2) = 1 
corresponds to ∇ q_j(t), where j = div(func,2).

# Arguments
- `t::Float64`: Evaluate spline at parameter t ∈ [0, param.T]
- `params::bcparams`: Parameters for the spline
- `func::Int64`: Control function index ∈ [0, param.Nseg-1]
- `g::Array{Float64,1}`: Preallocated array to store calculated gradient
"""
function gradbcarrier2!(t::Float64, params::bcparams, func::Int64, g::Array{Float64,1})

    # compute basic offset: func 0 and 1 use the same spline coefficients, but combined in a different way
    osc = div(func, 2)
    q_func = func % 2 # q_func = 0 for p and q_func=1 for q

    # allocate array for returning the results
    # g = zeros(length(params.pcof)) # cos and sin parts 

    g .= 0.0

    dtknot = params.dtknot
    width = 3*dtknot

    k = max.(3, ceil.(Int64,t./dtknot .+ 2)) # t_knot(k-1) < t <= t_knot(k), but t=0 needs to give k=3
    k = min.(k, params.D1) # protect agains roundoff that sometimes makes t/dt > N_nurbs-2
    
    if(func < 2*params.Ncoupled)
        @fastmath @inbounds @simd for freq in 1:params.Nfreq

            # offset in parameter array (osc = 0,1,2,...
            # Vary freq first, then osc
            offset1 = 2*osc*params.Nfreq*params.D1 + (freq-1)*2*params.D1
            offset2 = 2*osc*params.Nfreq*params.D1 + (freq-1)*2*params.D1 + params.D1

            #1st segment of nurb k
            tc = params.tcenter[k]
            tau = (t .- tc)./width
            bk = (9/8 .+ 4.5.*tau .+ 4.5.*tau.^2)
            if q_func==1
                g[offset1 .+ k] = bk * sin(params.om[osc+1,freq]*t)
                g[offset2 .+ k] = bk * cos(params.om[osc+1,freq]*t) 
            else # p-func
                g[offset1 .+ k] = bk * cos(params.om[osc+1,freq]*t)
                g[offset2 .+ k] = -bk * sin(params.om[osc+1,freq]*t) 
            end          

            #2nd segment of nurb k-1
            tc = params.tcenter[k.-1]
            tau = (t .- tc)./width
            bk = (0.75 .- 9 .*tau.^2)
            if q_func==1
                g[offset1 .+ (k-1)] = bk * sin(params.om[osc+1,freq]*t)
                g[offset2 .+ (k-1)] = bk * cos(params.om[osc+1,freq]*t) 
            else # p-func
                g[offset1 .+ (k-1)] = bk * cos(params.om[osc+1,freq]*t)
                g[offset2 .+ (k-1)] = -bk * sin(params.om[osc+1,freq]*t) 
            end
      
            # 3rd segment og nurb k-2
            tc = params.tcenter[k.-2]
            tau = (t .- tc)./width
            bk = (9/8 .- 4.5.*tau .+ 4.5.*tau.^2)
            if q_func==1
                g[offset1 .+ (k-2)] = bk * sin(params.om[osc+1,freq]*t)
                g[offset2 .+ (k-2)] = bk * cos(params.om[osc+1,freq]*t) 
            else # p-func
                g[offset1 .+ (k-2)] = bk * cos(params.om[osc+1,freq]*t)
                g[offset2 .+ (k-2)] = -bk * sin(params.om[osc+1,freq]*t) 
            end

        end #for freq
    else
        # uncoupled control case 
      @fastmath @inbounds @simd for freq in 1:params.Nfreq

        # offset
        offset = func*params.D1*params.Nfreq  +  (freq-1)*params.D1
        ind = func - 2*params.Ncoupled

        #1st segment of nurb k
        if k <= params.D1  # could do the if statement outside the for-loop instead
          tc = params.tcenter[k]
          tau = (t .- tc)./width
          g[offset .+ k] = (9/8 .+ 4.5.*tau .+ 4.5.*tau.^2)*cos(params.om[ind+1,freq]*t)
        end

        #2nd segment of nurb k-1
        if k >= 2 && k <= params.D1 + 1
          tc = params.tcenter[k.-1]
          tau = (t .- tc)./width
          g[offset .+ k.-1] = (0.75 .- 9 .*tau.^2)*cos(params.om[ind+1,freq]*t)
        end
      
        # 3rd segment og nurb k-2
        if k >= 3 && k <= params.D1 + 2
          tc = params.tcenter[k.-2]
          tau = (t .- tc)./width
          g[offset .+ k.-2] = (9/8 .- 4.5.*tau .+ 4.5.*tau.^2)*cos(params.om[ind+1,freq]*t)
        end
      
      end #for freq
    end
end



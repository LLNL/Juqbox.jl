module Juqbox

using LinearAlgebra
using Plots
pyplot()
using Printf
using Random
using LaTeXStrings
using SparseArrays
using Ipopt
using FileIO

export splineparams, bspline2, gradbspline2
export bcparams, bcarrier2, gradbcarrier2

export objparams, traceobjgrad, identify_guard_levels, identify_forbidden_levels, plotunitary, plotspecified, evalctrl
export setup_ipopt_problem, Working_Arrays, estimate_Neumann!, assign_thresholds, setup_rotmatrices
export run_optimizer, plot_conv_hist
export setup_prior!, wmatsetup, assign_thresholds_old, assign_thresholds_freq 
export init_adjoint!, tracefidabs2, tracefidreal,tracefidcomplex, trace_operator
export adjoint_trace_operator!, penalf2a, penalf2aTrap, penalf2adj, penalf2adj!
export penalf2grad, tikhonov_pen, tikhonov_grad!
export KS!, accumulate_matrix!, controlfunc, controlfuncgrad!, rotmatrices!
export fgradforce!, adjoint_grad_calc!, eval_forward, estimate_Neumann
export calculate_timestep, KS_alloc, time_step_alloc, grad_alloc
export eval_f_par, eval_g, eval_grad_f_par, eval_jac_g, intermediate_par
export plot_forward, specify_level3, marginalize3
export adjoint_tableau, step, step!, explicit_step, step_fwdGrad!, stepseparable, getgamma, magnus, neumann!

# Julia versions prior to v"1.3.1" can't use LinearAlgebra's 5 argument mul!, routines
# included here for backwards compatability
if(VERSION < v"1.3.1")
    include("backwards_compat.jl")
end

include("bsplines.jl") # add all B-spline functionality to the Juqbox module

include("StormerVerlet.jl") # add in time-stepping functionality

# union type for different ctrl parameterizations
BsplineParams = Union{splineparams, bcparams}
# union type for different matrix representations
MyRealMatrix = Union{Array{Float64,2}, SparseMatrixCSC{Float64, Int64}}

include("evalobjgrad.jl")

include("plotstatectrl.jl")

include("ipopt_interface.jl")

end # module

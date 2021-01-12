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
export bcparams, bcarrier2, gradbcarrier2!

export objparams, traceobjgrad, identify_guard_levels, identify_forbidden_levels, plotunitary, plotspecified, evalctrl
export setup_ipopt_problem, Working_Arrays, estimate_Neumann!, assign_thresholds, setup_rotmatrices
export run_optimizer, plot_conv_hist, neumann!
export wmatsetup, assign_thresholds_freq 
export calculate_timestep, marginalize3

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

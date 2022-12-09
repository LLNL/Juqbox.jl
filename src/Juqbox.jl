module Juqbox

using LinearAlgebra
using Plots
#pyplot()
using Printf
using Random
using LaTeXStrings
using SparseArrays
using Ipopt
using FileIO
using FFTW
using DelimitedFiles

export splineparams, bspline2, gradbspline2
export bcparams, bcarrier2, gradbcarrier2!

export objparams, traceobjgrad, identify_guard_levels, identify_forbidden_levels, initial_cond
export plotunitary, plotspecified, evalctrl, plot_results, plot_energy, plot_final_unitary
export setup_ipopt_problem, Working_Arrays, estimate_Neumann!, setup_rotmatrices
export run_optimizer, plot_conv_hist, wmatsetup
export zero_start_end!, assign_thresholds, assign_thresholds_freq, assign_thresholds_ctrl_freq 
export calculate_timestep, marginalize3, change_target!, set_adjoint_Sv_type!
export save_pcof, read_pcof, juq2qis
export lsolver_object
export hamiltonians_one_sys, get_resonances, init_control, control_bounds
export hamiltonians_two_sys, transformHamiltonians!, hamiltonians_three_sys

# Julia versions prior to v"1.3.1" can't use LinearAlgebra's 5 argument mul!, routines
# included here for backwards compatability
if(VERSION < v"1.3.1")
    include("backwards_compat.jl")
end

include("bsplines.jl") # add all B-spline functionality to the Juqbox module

include("linear_solvers.jl")

include("StormerVerlet.jl") # add in time-stepping functionality

# union type for different ctrl parameterizations
BsplineParams = Union{splineparams, bcparams}
# union type for different matrix representations
MyRealMatrix = Union{Array{Float64,2}, SparseMatrixCSC{Float64, Int64}}

WeightMatrix = Union{Array{Float64,2}, Diagonal{Float64, Vector{Float64}}}

include("evalobjgrad.jl")

include("ipopt_interface.jl")

include("plotstatectrl.jl")

include("plot-results.jl")

include("save_pcof.jl")

include("setup_problem.jl")

end # module

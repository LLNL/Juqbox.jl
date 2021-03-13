using Test
using LinearAlgebra
using Printf
using Random

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

import Juqbox

include("evalGrad.jl")
include("test-stormer-verlet.jl")

refDir = "reference_solutions"

# test Stormer-Verlet time stepping
refFile = refDir * "/" * "err-mat-ref.jld2"
success = timestep_convergence(refFile)
println("time stepping test: pass=", success)
@test success

# test traceobjgrad for various setups
caseNames =["rabi", "swap02", "flux", "cnot2", "cnot3"]
caseDir = "cases"
nCases = length(caseNames)
pass = BitArray(undef, nCases)
pass .= false

q = 0
for case in caseNames
    global q += 1
    
     # setup the testcase (assign pcof0, params and wa
    juliaFile = caseDir * "/" * case * "-setup.jl"
    include(juliaFile);

    # evaluate fidelity
    caseRefFile = refDir * "/" * case * "-ref.jld2"
    pass[q] = evalObjGrad(pcof0, params, wa, caseRefFile) 

    println("case=", case, " pass=", pass[q])
    @test pass[q]
    
end
    

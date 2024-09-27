using Test
using LinearAlgebra
using Printf
using Random

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

# include("../src/Juqbox.jl")
import Juqbox # quantum control module

include("evalGrad.jl")
include("test-stormer-verlet.jl")
include("test-implicit-midpoint.jl")

refDir = "reference_solutions"

# test Stormer-Verlet time stepping
refFile = refDir * "/" * "err-mat-ref.jld2"
success = timestep_convergence(refFile)
println("Stormer-Verlet time stepping test: pass=", success)
@test success

#test Implicit Midpoint Time stepping
refFile_imr = refDir * "/" * "err-mat-imr-ref.jld2"
success_imr = timestep_convergence_implicit(refFile_imr)
println("Implicit Midpoint time stepping test: pass=", success_imr)
@test success_imr

# test traceobjgrad for various setups
caseNames =["rabi", "swap02", "flux", "cnot2", "cnot3", "cnot2-leakieq", "cnot2-jacobi"]
caseDir = "cases"
nCases = length(caseNames)
pass = BitArray(undef, nCases)
pass .= false

q = 0
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("Testing Stormer Verlet adjoint method")
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
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
pass .= false
p = 0

println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("Testing Implicit Midpoint Rule adjoint method")
println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
for case in caseNames
    global p += 1
    
     # setup the testcase (assign pcof0, params and wa
    juliaFile = caseDir * "/" * case * "-setup.jl"
    include(juliaFile);
    
    params.Integrator_id = 2
    params.linear_solver = Juqbox.lsolver_object(solver=Juqbox.JACOBI_SOLVER_M,max_iter=100,tol=1e-12,nrhs=prod(N))
    
    local wa = Juqbox.Working_Arrays_M(params, nCoeff)

    # evaluate fidelity
    caseRefFile = refDir * "/" * case * "-ref-imr.jld2"
    pass[p] = evalObjGrad(pcof0, params, wa, caseRefFile) 

    println("case=", case, " pass=", pass[p])
    @test pass[p]
    
end
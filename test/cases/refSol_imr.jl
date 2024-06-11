# This script can be used to generate reference solutions for new cases for the Implicit Midpoint Rule
using DelimitedFiles
using Printf
using LinearAlgebra
import Juqbox
using Random

include("evalGrad.jl")

caseNames =["rabi", "swap02", "flux", "cnot2", "cnot3", "cnot2-leakieq", "cnot2-jacobi"]
dirName = "reference_solutions"

nCases = length(caseNames)
pass = BitArray(undef, nCases)
pass .= false

q = 0
for case in caseNames
    global q += 1
    
    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    println("Case: ", case)

     # setup the testcase (assign pcof0, params and wa)
    juliaFile = "cases" * "/" * case * "-setup.jl"
    include(juliaFile);

    params.Integrator_id = 2
    params.linear_solver = Juqbox.lsolver_object(solver=Juqbox.JACOBI_SOLVER_M,max_iter=100,tol=1e-12,nrhs=prod(N))
    
    local wa = Juqbox.Working_Arrays_M(params, nCoeff)

    # evaluate fidelity & save reference sol
    refFile = dirName * "/" * case * "-ref-imr.jld2"
    evalObjGrad(pcof0, params, wa, refFile, true) # true for saving a reference jld2 file

    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

end

println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("    TEST SUMMARY")

for q in 1:nCases
    println("case=", caseNames[q], " pass=", pass[q])
end
    
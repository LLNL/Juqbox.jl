using DelimitedFiles
using Printf
import Juqbox

include("evalGrad.jl")

caseNames =["rabi", "swap02", "flux", "cnot-lab", "cnot2", "cnot3"]
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
    juliaFile = case * "-setup.jl"
    include(juliaFile);

    # evaluate fidelity & save reference sol
    refFile = dirName * "/" * case * "-ref.jld2"
    evalObjGrad(pcof0, params, wa, refFile, true) # true for saving a reference jld2 file

    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

end

println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
println("    TEST SUMMARY")

for q in 1:nCases
    println("case=", caseNames[q], " pass=", pass[q])
end
    
    

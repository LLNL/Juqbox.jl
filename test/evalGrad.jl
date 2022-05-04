using FileIO

function evalObjGrad( pcof0:: Array{Float64, 1}, params:: Juqbox.objparams, wa:: Juqbox.Working_Arrays, refFileName:: String, writeFile:: Bool=false)
    rtol = 1e-10
    atol = 1e-14
    verbose = false

    # return flag
    success=false

    # evaluate fidelity
    nCoeff = length(pcof0)
    grad = zeros(nCoeff)
    objv = Juqbox.eval_f_par(pcof0,params, wa, [0.0], [1.0]);    # Use these functions to compute obj
    Juqbox.eval_grad_f_par(pcof0,grad,params, wa, [0.0], [1.0])  # and grad since they now include the tikhonov terms

    # objv = Juqbox.eval_f_par(pcof0,true,params, wa, [0.0], [1.0]);    # Use these functions to compute obj 
    # Juqbox.eval_grad_f_par(pcof0,false,grad,params, wa, [0.0], [1.0])  # and grad since they now include the tikhonov terms


    if writeFile
        save(refFileName, "obj0", objv, "grad0", grad)
        success=true
    else
        # read reference solution
        # pcofRef = vec(readdlm(refFileName)) # change to JLD2 file to get full accuracy
        # objvRef = pcofRef[1]
        # gradRef = pcofRef[2:end]
        dict = load(refFileName)
        objvRef = dict["obj0"]
        gradRef = dict["grad0"]

        objDiff = abs(objv - objvRef)

        refNorm = norm(gradRef)
        aNorm = norm(grad-gradRef)

        pass1 = false
        if objDiff < atol
            pass1 = true
        elseif abs(objvRef) >= atol && objDiff/abs(objvRef) < rtol
            pass1 = true
        end

        pass2 = false
        if aNorm < atol
            pass2 = true
        elseif refNorm >= atol &&  aNorm/refNorm < rtol
            pass2 = true
        end

        success = pass1 && pass2

        if !success
            println("rel objfunc error =", objDiff, " absolute grad error =", aNorm, " relative grad error =", aNorm/refNorm)
        end
        
    end
    
    return success
end

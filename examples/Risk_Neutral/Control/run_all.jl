#================================================
This routine performs an optimization to 
find control functions to enact the X gate
on a single qudit
    U_T = [0     1;
           1     0]
We let e∼Unif(-0.02,0.02) and model uncertain 
control Hamiltonians via
        H_cᵘ = (1+e)*p(t)*(a+a')
In this script we perform first a noise-free (usual) 
optimization in which we set e = 0. We then 
perform a risk-neutral optimization in which 
we aim to minimize the expected value of the 
usual objective function J,
        E[J] = ∫ J dμ ≈ ∑ wⱼ J[eⱼ],
where we use a Gauss quadrature rule to 
approximate the integral E[J].
================================================#

using LaTeXStrings
using Juqbox
using Plots

# Function to loop over ϵ values and create a plot of the objective function for pertrubed Hamiltonians
function ep_plot(pcof::Vector{Float64}, params:: Juqbox.objparams, wa::Working_Arrays, ep_vals::AbstractArray)
    results = zeros(length(ep_vals),4)

    for i =1:length(ep_vals)
        ep = ep_vals[i]
        
        # Multiplicative noise 
        for j = 1:length(params.Hsym_ops)
            params.Hsym_ops[j] .*= ep
        end

        obj, _, _, secondaryobjf, traceinfid = Juqbox.traceobjgrad(pcof,params,wa,false, true)
        results[i,1] = obj
        results[i,2] = secondaryobjf
        results[i,3] = traceinfid
        results[i,4] = traceinfid+secondaryobjf
        
        # Reset
        for j = 1:length(params.Hsym_ops)
            params.Hsym_ops[j] ./= ep
        end

    end

    pl1 = Plots.plot(ep_vals,results[:,1],yaxis=:log,xlabel = L"\epsilon",ylabel = "Objective Function")
    return results,pl1
end



function evalctrl_no_carrier(params::objparams, pcof0:: Array{Float64, 1}, jFunc:: Int64, D1::Int64) 
    
    # Evaluate the ctrl functions on this grid in time
    nplot = round(Int64, params.T*32)
    # is this resolution sufficient for the lab frame ctrl functions so we can get meaningful FFTs?
    td = collect(range(0, stop = params.T, length = nplot+1))

    nfreq = length(params.Cfreq)
    offset = (jFunc-1)*2*D1
    nCoeff = 2*D1
    pcof = copy(pcof0[offset+1:offset+nCoeff])

    if (params.use_bcarrier)
        # B-splines with carrier waves (zero out the frequency to plot just the splines)
        splinepar = Juqbox.bcparams(params.T, D1, params.Ncoupled, params.Nunc, zeros(1,1), pcof)
    else
        # regular B-splines
        splinepar = Juqbox.splineparams(params.T, D1, 2*(params.Ncoupled + params.Nunc), pcof)
    end

    # define inline function to enable vectorization over t
    controlplot(t, splinefunc) = Juqbox.controlfunc(t, splinepar, splinefunc)

    fact = 1.0/(2*pi) # conversion factor to GHz
    fact = 1.0 # conversion factor to rad/ns
    pj = fact.*controlplot.(td, 0)
    qj = fact.*controlplot.(td, 1)
    return pj, qj
    
end

# Range of perturbation values to evaluate the objective function
# (larger than desired range of optimization)
min_ep_sweep = 1.0-0.05
max_ep_sweep = 1.0+0.05
len = 1001
ep_vals = range(min_ep_sweep,stop=max_ep_sweep,length=len)
freshOptim = true


#================================================
Perform a noise-free (usual) optimization.
================================================#
nquad = 1
startFromScratch = true
Nfreq = 1
ep_max = 0.00
include("x-risk-neutral.jl")
if(freshOptim)

    # Perform noise-free optimization
    pcof = Juqbox.run_optimizer(prob, pcof0)

    # Loop over given range of control perturbations and calculate objective function
    results,pl1 = ep_plot(pcof, params, wa, ep_vals)

    # Process and save data
    data = zeros(size(results,1),4)
    data[:,1] = ep_vals
    for j = 1:3
        data[:,j+1] = results[:,j]
    end
    writedlm("usual_control_GLQ2.dat", pcof)
    writedlm("usual_optim_OF_sweep_GLQ2.dat", data)
else
    pcof = vec(readdlm("usual_control_GLQ2.dat"))
    results = readdlm("usual_optim_OF_sweep_GLQ2.dat")
end

pcof_old = copy(pcof)


#================================================
Evaluate control functions without carrier waves 
and plot
================================================#
scalefactor = 1000/(2*pi)
unitStr = "MHz"

nplot = round(Int64, params.T*32)
td = collect(range(0, stop = params.T, length = nplot+1))
p_NF_1,q_NF_1 = evalctrl_no_carrier(params, pcof_old, 1, D1) 

pfunc1 = scalefactor .* p_NF_1
qfunc1 = scalefactor .* q_NF_1
pmax = maximum(abs.(pfunc1))
qmax = maximum(abs.(qfunc1))

fnt = Plots.font("Helvetica", 12)
lfnt = Plots.font("Helvetica", 12)
Plots.default(titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=lfnt, linewidth=1.5, size=(650, 350))

titlestr = "Rotating frame NF ctrl " * " Max-p=" *@sprintf("%.3e", pmax) * " Max-q=" *@sprintf("%.3e", qmax) * " " * unitStr

pl_ctrl_NF = Plots.plot(td, pfunc1, lab=L"p_{1}(t)", title = titlestr, xlabel="Time [ns]",
                                  ylabel=unitStr, legend= :outerright, linewidth=1.5, legendfontsize=14)
# add in the control function for the anti-symmetric Hamiltonian
Plots.plot!(pl_ctrl_NF,td, qfunc1, lab=L"q_{1}(t)", linewidth=1.5, legendfontsize=14)

# save plots of control functions in rotating frame without carrier waves
Plots.savefig(pl_ctrl_NF,  "robust_comparison_"*@sprintf("%3d",T)*"_T_"*@sprintf("%2d",maxIter)*"_iters_"*@sprintf("%d",nquad)*"_N_"*@sprintf("%d",D1)*"_D1_NF_ctrl.png")

#================================================
Perform a risk-neutral optimization. The strategy
here is to first optimize for a smaller maximum
dispersion value for the |2⟩-|3⟩ transition.
We use this solution as an initial guess for the
optimization using the full specified range.
================================================#
nquad = 5
Nfreq = 1
ep_max = 0.02
include("x-risk-neutral.jl")
if(freshOptim)
    
    # initial optimization to get initial guess for wider noise range
    pcof = Juqbox.run_optimizer(prob, pcof0)
    # save_pcof(startFile,pcof)

    # Loop over range of dispersion values and plot the objective function
    results2,pl2 = ep_plot(pcof, params, wa, ep_vals)
    data2 = zeros(size(results2,1),4)
    data2[:,1] = ep_vals
    for j = 1:3
        data2[:,j+1] = results2[:,j]
    end
    writedlm("robust_control_GLQ2.dat", pcof)
    writedlm("robust_optim_OF_sweep_GLQ2.dat", data2)
else
    pcof = vec(readdlm("robust_control_GLQ2.dat"))
    results2 = readdlm("robust_optim_OF_sweep_GLQ2.dat")
end


#================================================
Plot objective functions for various Hamiltonian 
perturbations.
================================================#
plc = Plots.plot(ep_vals,abs.(results[:,3]),yaxis=:log,xlabel = "Control function scaling",ylabel = "Objective Function", lab="NF Infidelity")
Plots.plot!(plc,ep_vals,abs.(results2[:,3]),yaxis=:log,xlabel = "Control function scaling",ylabel = "Objective Function", lab="RN Infidelity")
Plots.plot!(plc,ep_vals,abs.(results[:,2]),yaxis=:log,xlabel = "Control function scaling",ylabel = "Objective Function", lab="NF Guard Level Pop.",linestyle=:dash)
Plots.plot!(plc,ep_vals,abs.(results2[:,2]),yaxis=:log,xlabel = "Control function scaling",ylabel = "Objective Function", lab="RN Guard Level Pop.",linestyle=:dash,legend= :outerright)

plc2 = Plots.plot(ep_vals,abs.(results[:,2].+results[:,3]),yaxis=:log,xlabel ="Control function scaling",ylabel = "Objective Function", lab="Total NF Objective")
Plots.plot!(plc2,ep_vals,abs.(results2[:,2].+results2[:,3]),yaxis=:log,xlabel ="Control function scaling",ylabel = "Objective Function", lab="Total RN Objective",legend= :outerright)

# Save plots to file
Plots.savefig(plc,  "robust_comparison_"*@sprintf("%3d",T)*"_T_"*@sprintf("%2d",maxIter)*"_iters_"*@sprintf("%d",nquad)*"_N_"*@sprintf("%d",D1)*"_D1_separate.png")
Plots.savefig(plc2,  "robust_comparison_"*@sprintf("%3d",T)*"_T_"*@sprintf("%2d",maxIter)*"_iters_"*@sprintf("%d",nquad)*"_N_"*@sprintf("%d",D1)*"_D1_total.png")


#================================================
Plot control functions (no carrier waves) of the
risk-neutral control function and save to file.
================================================#
p_RN_1,q_RN_1 = evalctrl_no_carrier(params, pcof, 1, D1) 
pfunc1 = scalefactor .* p_RN_1
qfunc1 = scalefactor .* q_RN_1
pmax = maximum(abs.(pfunc1))
qmax = maximum(abs.(qfunc1))

titlestr = "Rotating frame RN ctrl " * " Max-p=" *@sprintf("%.3e", pmax) * " Max-q=" *@sprintf("%.3e", qmax) * " " * unitStr
pl_ctrl_RN = Plots.plot(td, pfunc1, lab=L"p_(t)", title = titlestr, xlabel="Time [ns]",
                                  ylabel=unitStr, legend= :topleft, linewidth=1.5, legendfontsize=12)

# add in the control function for the anti-symmetric Hamiltonian
Plots.plot!(pl_ctrl_RN,td, qfunc1, lab=L"q_(t)", linewidth=1.5, legendfontsize=12)
Plots.savefig(pl_ctrl_RN,  "robust_comparison_"*@sprintf("%3d",T)*"_T_"*@sprintf("%2d",maxIter)*"_iters_"*@sprintf("%d",nquad)*"_N_"*@sprintf("%d",D1)*"D1_RN_ctrl.png")

# Save coefficients
save_pcof("robust_comparison_"*@sprintf("%3d",T)*"_T_"*@sprintf("%2d",maxIter)*"_iters_"*@sprintf("%d",nquad)*"_N_"*@sprintf("%d",D1)*"D1_NF_pcof.jld2",pcof_old)
save_pcof("robust_comparison_"*@sprintf("%3d",T)*"_T_"*@sprintf("%2d",maxIter)*"_iters_"*@sprintf("%d",nquad)*"_N_"*@sprintf("%d",D1)*"D1_RN_pcof.jld2",pcof)

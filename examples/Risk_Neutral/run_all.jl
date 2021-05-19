using Plots
using LaTeXStrings
using Juqbox

# Function to loop over ϵ values and create a plot of the objective function for pertrubed Hamiltonians
function ep_plot(pcof::Vector{Float64}, params:: Juqbox.objparams, wa::Working_Arrays, ep_vals::AbstractArray)
    results = zeros(length(ep_vals),3)

    H0_old = copy(params.Hconst)

    for i =1:length(ep_vals)
        epsh = ep_vals[i]
        
        # Additive noise
        for j = 2:size(params.Hconst,2)
            params.Hconst[j,j] = H0_old[j,j] + 0.01*epsh*(10.0^(j-2))
        end

        obj, _, _, secondaryobjf, traceinfid = Juqbox.traceobjgrad(pcof,params,wa,false, true)
        results[i,1] = obj
        results[i,2] = secondaryobjf
        results[i,3] = traceinfid
    end
    copy!(params.Hconst,H0_old)

    pl1 = plot(ep_vals,results[:,1],yaxis=:log,xlabel = L"\epsilon",ylabel = "Objective Function")
    return results,pl1
end

# Maximum shift in Hamiltonian (in rad*GHz)
ep_max = 2*pi*3e-2

# For plotting purposes 
max_ep_sweep = 1e-1
len = 300
ep_vals = range(-max_ep_sweep,stop=max_ep_sweep,length=len)
freshOptim = true 

# Usual optimization
nquad = 1
include("swap-02-risk-neutral.jl")
if(freshOptim)
    pcof = Juqbox.run_optimizer(prob, pcof0)
else
    pcof = vec(readdlm("usual_control_GLQ.dat"))
end
results,pl1 = ep_plot(pcof, params, wa, ep_vals)
data = zeros(size(results,1),4)
data[:,1] = ep_vals
for j = 1:3
    data[:,j+1] = results[:,j]
end
writedlm("usual_control_GLQ.dat", pcof)
writedlm("usual_optim_OF_sweep_GLQ.dat", data)


# Risk-neutral optimization
nquad = 20
include("swap-02-risk-neutral.jl")
if(freshOptim)
    pcof = Juqbox.run_optimizer(prob, pcof0)
else
    pcof = vec(readdlm("robust_control_GLQ.dat"))
end
results2,pl2 = ep_plot(pcof, params, wa, ep_vals)
data2 = zeros(size(results,1),4)
data2[:,1] = ep_vals
for j = 1:3
    data2[:,j+1] = results2[:,j]
end
writedlm("robust_control_GLQ.dat", pcof)
writedlm("robust_optim_OF_sweep_GLQ.dat", data)

# Plot all results on single plot
fnt = Plots.font("Helvetica", 12)
lfnt = Plots.font("Helvetica", 10)
Plots.default(titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=lfnt, linewidth=1, size=(650, 350))
plc = plot(ep_vals,results[:,1],yaxis=:log,xlabel = L"\epsilon",ylabel = "Objective Function", lab=L"\mathcal{J}")
plot!(plc,ep_vals,results2[:,1],yaxis=:log,xlabel = L"\epsilon",ylabel = "Objective Function", lab=L"\mathbb{E}[\mathcal{J}]",legend= :outerright)
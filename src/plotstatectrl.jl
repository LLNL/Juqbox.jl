# Works for Nosc={1, 2, 3}
"""
    plt = plotunitary(us, params, guardlev)

Plot the evolution of the state vector.
 
# Arguments
- `us:: Array{Complex{Float64},3})`: State vector history for each timestep
- `param:: objparams`: Struct with problem definition
- `guardlev:: Array{Bool,1})`: Boolean array indicating if a certain level is a guard level
"""
function plotunitary(us, params, guardlev)
    nsteps = length(us[1,1,:])
    Ntot = length(us[:,1,1])
    N =  length(us[1,:,1])
    t = range(0, stop = params.T, length = nsteps)
    if nsteps < 10000
        stride = 1
    elseif nsteps < 20000
        stride = 2
    else 
        stride = 4 # only plot every 4th data point
    end
    rg = 1:stride:nsteps # range object

  # one figure for the response of each basis vector
    plotarray = Array{Plots.Plot}(undef, N) #empty array for separate plots

    for ii in 1:N # N = Ne[1] * Ne[2], ordered as
        if params.Nosc == 1
            statestr = string(ii-1)
        elseif params.Nosc == 2
            # Example: Ne[2] = 3, Ne[1]=4 
            #   0,   1,  2,   3,   4,  5,   6,   7,   8,  9,  10, 11  (ii-1)
            # 00, 01, 02, 03, 10, 11, 12, 13, 20, 21, 22, 23
            statestr = string( div((ii-1), params.Ne[1]), mod(ii-1, params.Ne[1]))
        elseif params.Nosc == 3
            # Example: Ne[3] = 2, Ne[2] = 3, Ne[1]=4 
            #     0,    1,     2,    3,    4,     5,     6,    7,     8,    9,   10,   11,   12,   13,   14,  15,   16,   17,   18,  19,   20,   21,   22,   23  (ii-1)
            # 000, 001, 002, 003, 010, 011, 012, 013, 020, 021, 022, 023,  100, 101, 102, 103, 110, 111, 112, 113, 120, 121, 122, 123
            s3 = div((ii-1), params.Ne[1]*params.Ne[2])
            s12 = (ii-1) % (params.Ne[1]*params.Ne[2])
            s2 = div(s12, params.Ne[1])
            s1 = s12 %  params.Ne[1]
            statestr = string( s3, s2, s1 )
        end
        titlestr = raw"Evolution from state $|" * statestr * raw"\rangle$"
#        h = plot(title = titlestr, size=(650, 400), legend= :outerright)
        h = plot(title = titlestr, legend= :outerright)
        for jj in 1:Ntot
            # Is jj an essential level?
            if !guardlev[jj]
                if params.Nosc == 1
                    labstr = string(jj-1) # "" for no labels
                elseif params.Nosc == 2
                    #   for s2 = 0:Nt[2]-1
                    #     for s1 = 0:Nt[1]-1
                    #        jj = s2*Nt[1] + s1 + 1
                    labstr = string( div((jj-1),params.Nt[1]), mod(jj-1, params.Nt[1]) )
                elseif params.Nosc == 3
                    # for s3 = 0:Nt[3]-1
                    #   for s2 = 0:Nt[2]-1
                    #     for s1 = 0:Nt[1]-1
                    #        jj = s3*Nt[1]*Nt[2] + s2*Nt[1] + s1 + 1
                    s3 = div((jj-1), params.Nt[1]*params.Nt[2])
                    s12 = jj -1 - s3 * params.Nt[1]*params.Nt[2]
                    s2 = div(s12, params.Nt[1])
                    s1 = s12 - s2 * params.Nt[1]
                    labstr = string( s3, s2, s1 )
                end
                plot!(t[rg], abs.(us[jj,ii,rg]).^2, lab = labstr, xlabel = "Time [ns]", ylabel="Population")
            end
        end
        plotarray[ii] = h
    end
    if N <= 2
        plt = plot(plotarray..., layout = (N,1))
    else
        plt = plot(plotarray..., layout = N, size=(650, 400))
    end
    
    return plt
end

# Works for Nosc={1, 2, 3}
"""
    plt = plotspecified(us, params, guardlev, specifiedlev)

Plot the evolution of the state vector for specified levels.
 
# Arguments
- `us:: Array{Complex{Float64},3})`: State vector history for each timestep
- `param:: objparams`: Struct with problem definition
- `us:: Array{Bool,1})`: Boolean array indicating if a certain level is a guard level
- `specifiedlev:: Array{Bool,1}`: Boolean array indicating which levels to be plotted
"""
function plotspecified(us, params, guardlev::Array{Bool,1}, specifiedlev::Array{Bool,1})
    nsteps = length(us[1,1,:])
    Ntot = length(us[:,1,1])
    N =  length(us[1,:,1])
    t = range(0, stop = params.T, length = nsteps)
    if nsteps < 10000
        stride = 1
    elseif nsteps < 20000
        stride = 2
    else 
        stride = 4 # only plot every 4th data point
    end
    rg = 1:stride:nsteps # range object

    # How many specified levels are there?
    nForb = 0
    gLev=-1
    for jj in 1:Ntot
        if specifiedlev[jj]
            nForb += 1
            gLev = jj-1
        end
    end
    
    # plot rows corresponding to specified guard levels = specified levels
#    plotarray = Array{Plots.Plot}(undef, N) #empty array for separate plots

    if nForb == 1
        titlestr = raw"Population of state $|" * string(gLev) * raw"\rangle$"
    else
        titlestr = "Population of guard levels"
    end
    h = plot(title = titlestr, size=(700, 350), legend= :outerright) # make it big to fit the legend
#    h = plot(title = titlestr, size=(700, 350)) # put legend inside plot

    iplot = 0
    for col in 1:Ntot
        if !guardlev[col] # col is an essential level: plot this column in the 'us' array
            iplot += 1
            if params.Nosc == 1
                statestr = string(col-1)
            elseif params.Nosc == 2
                s2 = div((col-1), params.Nt[1])
                s1 = (col-1) % params.Nt[1]
                statestr = string( s2, s1 )
            elseif params.Nosc == 3
                s3 = div((col-1), params.Nt[1]*params.Nt[2])
                s12 = (col-1) % (params.Nt[1]*params.Nt[2])
                s2 = div(s12, params.Nt[1])
                s1 = s12 %  params.Nt[1]
                statestr = string( s3, s2, s1 )
            end
            # tmp
            # println("col=", col, " guardlev=", guardlev[col], " statestr= ", statestr)
            # end tmp
            for row in 1:Ntot
                # Is row a specified level?
                if specifiedlev[row] # only plot the specified = forbidden levels (rows in 'us')
                    if params.Nosc == 1
                        labstr = string(row-1)
                    elseif params.Nosc == 2
                        s2 = div((row-1), params.Nt[1])
                        s1 = (row-1) % params.Nt[1]
                        labstr = string( s2, s1 )
                    elseif params.Nosc == 3
                        s3 = div((row-1), params.Nt[1]*params.Nt[2])
                        s12 = row -1 - s3 * params.Nt[1]*params.Nt[2]
                        s2 = div(s12, params.Nt[1])
                        s1 = s12 - s2 * params.Nt[1]
                        labstr = string( s3, s2, s1 )
                    end
                    # Not enough room for too many labels
                    if nForb > 16
                        labstr = ""
                    end
                    plot!(t[rg], abs.(us[row,iplot,rg]).^2, lab = labstr * " from " * statestr * " state", xlabel = "Time [ns]", ylabel="Population")
                end
            end # for row
#            plotarray[iplot] = h
        end
    end
    plt3 = h
    # if N <= 2
    #     plt3 = plot(plotarray..., layout = (N,1))
    # else
    #     plt3 = plot(plotarray..., layout = N)
    # end
    
    return plt3
end

function plot_forward(us, T)
    nsteps = length(us[1,1,:])  
    Ntot = length(us[:,1,1])
    N =  length(us[1,:,1])

    U0 = us[:,1,1]
    dum, ii = findmax(abs.(U0)) # Which basis did we start from?
    # test
    println("plot_forward: initial condition corresponds to element # ", ii-1, " (vectorized representation)")

    t = range(0, stop = T, length = nsteps)
    rg = 1:nsteps # range object  # plot all data points

#    ii = 1;
    titlestr = raw"Evolution from state $|" * string(ii-1) * raw"\rangle$"
    plt = plot(xlabel = "Time [ns]", ylabel = "Population", title = titlestr, size=(750, 400), legend=:outerright) # abs^2
    for jj = 1:Ntot
#        if jj != ii # don't plot the evolution of the initial state
            labstr = string("State ", jj-1)
            plot!(t[rg], abs.(us[jj,1,:]).^2, lab=labstr) # abs^2
#        end
    end
    return plt
end

# Evaluate the control functions on a grid in time in units of GHz
"""
    pj [, qj] = evalctrl(params, pcof0, td, func) 

Evaluate the control function with index `func` at an array of time levels `td`.  

NOTE: the control function index `func` is 1-based. 

NOTE: The return value(s) depend on `func`. For `func∈[1,Ncoupled]`, `pj, qj` are returned. Otherwise, 
only `pj` is returned, corresponding to control number `func`.

# Arguments
- `params:: objparams`: Struct with problem definition
- `pcof0:: Array{Float64,1})`: Vector of parameter values
- `td:: Array{Float64,1})`: Time values control is to be evaluated
- `jFunc:: Int64`: Index of the control signal desired
"""
function evalctrl(params::objparams, pcof0:: Array{Float64, 1}, td:: Array{Float64, 1}, jFunc:: Int64) 
    if params.pFidType == 3
        nCoeff = length(pcof0)-1
    else
        nCoeff = length(pcof0)
    end
    pcof = pcof0[1:nCoeff]
    D1 = div(nCoeff, 2*(params.Ncoupled + params.Nunc)*params.Nfreq)  # number of B-spline coeff per control function

    if (params.use_bcarrier)
        # B-splines with carrier waves
        splinepar = bcparams(params.T, D1, params.Ncoupled, params.Nunc, params.Cfreq, pcof)
    else
        # regular B-splines
        splinepar = splineparams(params.T, D1, 2*(params.Ncoupled + params.Nunc), pcof)
    end

    # define inline function to enable vectorization over t
    controlplot(t, splinefunc) = controlfunc(t, splinepar, splinefunc)

    fact = 1.0/(2*pi) # conversion factor to GHz
    fact = 1.0 # conversion factor to rad/ns

    # coupled & uncoupled controls are treated the same way
    qs = (jFunc-1)*2
    qa = qs+1
        
    pj = fact.*controlplot.(td, qs)
    qj = fact.*controlplot.(td, qa)
    return pj, qj
    
end

"""
    guardlev = identify_guard_levels(params[, custom = 0])

Build a Bool array indicating if a given energy level is a guard
level in the simulation.
 
# Arguments
- `params:: objparams`: Struct with problem definition
- `custom:: Int64`: A nonzero value gives a special stirap pulses case
"""
function identify_guard_levels(params::Juqbox.objparams, custom:: Int64 = 0)
    # identify all guard levels
    Ntot = params.N+params.Nguard
    guardlev = fill(false, Ntot)

    if params.Nosc == 1
        if custom == 0
            guardlev[params.N+1:Ntot] .= true
        else # special case for stirap pulses
            guardlev[2] = true
            guardlev[4] = true
        end
    elseif params.Nosc == 2
        for q2 in 1:params.Nt[2]
            for q1 in 1:params.Nt[1]
                if q1 > params.Ne[1] || q2 > params.Ne[2]
                    guardlev[(q2-1)*params.Nt[1] + q1] = true
                end
            end
        end
    elseif params.Nosc == 3
        for q3 in 1:params.Nt[3]
            for q2 in 1:params.Nt[2]
                for q1 in 1:params.Nt[1]
                    if q1 > params.Ne[1] || q2 > params.Ne[2] || q3 > params.Ne[3]
                        guardlev[(q3-1)*params.Nt[1]*params.Nt[2] + (q2-1)*params.Nt[1] + q1] = true
                    end
                end
            end
        end
    end
    return guardlev
end #identify_guard_levels

"""
    forbiddenlev = identify_guard_levels(params[, custom = 0])

Build a Bool array indicating which energy levels are forbidden levels in the state vector. The
forbidden levels in a state vector are defined as thos corresponding to the highest energy level in
at least one of its subsystems.
 
# Arguments
- `params:: objparams`: Struct with problem definition
- `custom:: Int64`: For nonzero value special stirap pulses case
"""
function identify_forbidden_levels(params:: Juqbox.objparams, custom::Int64 = 0)
    # identify all forbidden levels
    Ntot = params.N+params.Nguard
    forbiddenlev = fill(false, Ntot)

    if params.Nosc == 1
        if custom == 0
            forbiddenlev[Ntot] = true
        else # Special case for stirap pulses
            forbiddenlev[2] = true
            forbiddenlev[4] = true
        end
    elseif params.Nosc == 2
        k=0
        for q2 in 1:params.Nt[2]
            for q1 in 1:params.Nt[1]
                k += 1
                if q1 == params.Nt[1] || q2 == params.Nt[2]
                    forbiddenlev[k] = true
                end
            end
        end
    elseif params.Nosc == 3
        k=0
        for q3 in 1:params.Nt[3]
            for q2 in 1:params.Nt[2]
                for q1 in 1:params.Nt[1]
                    k += 1
                    if q1 == params.Nt[1] || q2 == params.Nt[2] || q3 == params.Nt[3]
                        forbiddenlev[k] = true
                    end
                end
            end
        end
    end
    return forbiddenlev
end #identify_forbidden_levels


function specify_level3(params:: Juqbox.objparams, Nl3:: Int64) # Nl3 is 0-based
    # identify all  levels
    Ntot = params.N+params.Nguard
    specifiedlev = fill(false, Ntot)

    if params.Nosc == 3
        k=0
        for q3 in 1:params.Nt[3]
            for q2 in 1:params.Nt[2]
                for q1 in 1:params.Nt[1]
                    k += 1
                    if q3 == Nl3+1
                        specifiedlev[k] = true
                    end
                end
            end
        end
    end
    return specifiedlev
end #specify_level3


"""
    marg_prob = marginalize3(params, unitaryhist)

Evaluate marginalized probabilities for the case of 3 subsystems.
 
# Arguments
- `param:: objparams`: Struct with problem definition
- `unitaryhist:: Array{Complex{Float64},3})`: State vector history for each timestep
"""
function marginalize3(params:: Juqbox.objparams, unitaryhist:: Array{Complex{Float64},3})
    nsteps1 = size(unitaryhist,3)
    if params.Nosc == 3
        marg_prob = zeros(params.Nt[3], params.N, nsteps1)

        k=0
        for q3 in 1:params.Nt[3]
            for q2 in 1:params.Nt[2]
                for q1 in 1:params.Nt[1]
                    #                    offset = (q3-1)*params.Nt[1]*params.Nt[2]
                    k += 1
                    marg_prob[q3, :, :] += abs.(unitaryhist[k, :, :]).^2
                end # for q1
            end # for q2
        end # for q3

        return marg_prob
    end # if Nosc = 3
end #marginalize3

"""
    pconv = plot_conv_hist(params [, convname:: String=""])

Plot the optimization convergence history, including history of 
the different terms in the objective function and the norm of the gradient.

# Arguments
- `param:: objparams`: Struct with problem definition
- `convname:: String`: Name of plot file to be generated
"""
function plot_conv_hist(params:: Juqbox.objparams, convname:: String="")
    pconv = Plots.plot(xlabel="Iteration", title="Convergence history", size=(400, 300))
    if params.saveConvHist && length(params.objHist)>0
        nIter = length(params.objHist)
        # note that primaryHist and secondaryHist have one additional elements corresponding to iter=0

        if minimum(params.objHist) > 0 && minimum(params.primaryHist) > 0 &&
            (params.Nguard == 0 || minimum(params.secondaryHist) > 0)
            Plots.plot!(pconv, yscale=:log10)
        end
        Plots.plot!(pconv, 1:nIter, params.objHist, lab=L"{\cal G}") #, markershape=:utriangle, markersize=5)
        Plots.plot!(pconv, 1:nIter, params.primaryHist, lab=L"{\cal J}_1", style=:dash)
        if (params.Nguard > 0)
            Plots.plot!(pconv, 1:nIter, params.secondaryHist, lab=L"{\cal J}_2")
        end
        Plots.plot!(pconv, 1:nIter, params.dualInfidelityHist, lab=L"\|\nabla{\cal G} - z\|_\infty") # dual infeasibility
        Plots.xlims!(pconv, (0, nIter+1))

        if length(convname)>0
            Plots.savefig(pconv, convname)
            println("Saved convergence history plot on file '", convname, "'")
        end
    else
        println("Warning: plot_conv_hist: no convergence history to plot")
    end
    return pconv
end #plot_conv_hist


"""
    pl = plot_results(params, pcof; [casename = "test", save_files = false, samplerate = 32])

Create array of plot objects that can be visualized by, e.g., `display(pl[1])`.

# Arguments
- `params::objparams`: Object holding problem definition
- `pcof::Array{Float64,1}`: Parameter vector
- `casename::String`: Default: `"test"`. String used in plot titles and in file names
- `save_files::Bool`: Default: `false`.Set to `true` to save plots on files with automatically generated filenames
- `samplerate:: Int64`: Default: `32` samples per unit time (ns). Sample rate for generating plots.
"""
function plot_results(params::objparams, pcof::Array{Float64,1}; casename::String = "test", save_files::Bool = false, samplerate:: Int64 = 32)
    # Set default font sizes
    fnt = Plots.font("Helvetica", 12)
    lfnt = Plots.font("Helvetica", 10)
    Plots.default(titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=lfnt, linewidth=1, size=(650, 350))

    nCoeff = length(pcof)
    # Is there a better approach that avoids re-allocating the working_arrays object?
    wa = Juqbox.Working_Arrays(params, nCoeff)

    custom = 0

    # filenames
    cfname = casename * "-ctrl-rot" * ".png"
    probname = casename * "-prob" * ".png"
    forbname = casename * "-forb" * ".png"
    fftname = casename * "-fft-lab-ctrl-lin" * ".png"
    fftname2 = casename * "-fft-lab-ctrl-log" * ".png"
    # data file names
    labbasename = casename * "-ctrl-lab"
    ctrlbasename = casename * "-ctrl-rot"

    println("Tikhonov coefficient: tik0 = ", params.tik0)
    println("B-carrier basis: ", params.use_bcarrier)
    println("Number of time steps: ", params.nsteps)

    # evaluate fidelity
    objv, grad, unitaryhistory, fidelity = Juqbox.traceobjgrad(pcof, params, wa, true, true);

    # save convergence history
    convname = ""
    if save_files
        convname = casename * "-conv" * ".png"
    end
    pconv = Juqbox.plot_conv_hist(params, convname)

    # scatter plot of control parameters
    tstring = casename * "-control-vector"
    plcof = scatter(pcof, lab="", title=tstring, xlabel="Index", ylabel="rad/ns")

    guardlev = Juqbox.identify_guard_levels(params, custom)
    forbiddenlev = Juqbox.identify_forbidden_levels(params, custom)

    # make plots of the evolution of probabilities
    pl1 = Juqbox.plotunitary(unitaryhistory, params, guardlev)
    pl3 = Juqbox.plotspecified(unitaryhistory, params, guardlev, forbiddenlev)

    if save_files
        # save the figure with the state probabilities
        Plots.savefig(pl1,probname)
        println("Saved state population plot on file '", probname, "'");

        # save the figure with the forbidden state
        Plots.savefig(pl3,forbname)
        println("Saved forbidden state population plot on file '", forbname, "'");
    end

    if params.Nosc == 3 # Generalize to Nosc = 2
        # evaluate marginalized probabilities
        mp = Juqbox.marginalize3(params, unitaryhistory);
        # plot them
        nsteps1=size(unitaryhistory,3)
        tm = range(0, stop = params.T, length = nsteps1);
        # one subfigure for each initial condition
        plotarray = Array{Plots.Plot}(undef, params.N) #empty array for separate plots

        for col in 1:params.N # One plot for each initial condition
            #  params.Nosc == 3
            # Example: Ne[3] = 2, Ne[2] = 3, Ne[1]=4 
            #     0,    1,     2,    3,    4,     5,     6,    7,     8,    9,   10,   11,   12,   13,   14,  15,   16,   17,   18,  19,   20,   21,   22,   23  (col-1)
            # 000, 001, 002, 003, 010, 011, 012, 013, 020, 021, 022, 023,  100, 101, 102, 103, 110, 111, 112, 113, 120, 121, 122, 123
            s3 = div((col-1), params.Ne[1]*params.Ne[2])
            s12 = (col-1) % (params.Ne[1]*params.Ne[2])
            s2 = div(s12, params.Ne[1])
            s1 = s12 %  params.Ne[1]
            statestr = string( s3, s2, s1 )

            local titlestr = raw"Evolution from state $|" * statestr * raw"\rangle$"
            h = Plots.plot(title = titlestr)

            for row in 1:params.Nt[3]
                labstr = "S-state " * string(row-1)
                Plots.plot!(tm, mp[row,col,:], lab = labstr, xlabel = "Time [ns]", ylabel="Marg. prob.", legend= :outerright)
            end # for row
            plotarray[col] = h
        end # for col
        if params.N <= 2
            plm = Plots.plot(plotarray..., layout = (params.N,1))
        else
            plm = Plots.plot(plotarray..., layout = params.N)
        end
    end


    # Evaluate the ctrl functions on this grid in time
    nplot = round(Int64, params.T*samplerate)
    # is this resolution sufficient for the lab frame ctrl functions so we can get meaningful FFTs?
    td = collect(range(0, stop = params.T, length = nplot+1))

    # Initialize storing of the lab drive
    labdrive = zeros(nplot+1)

    useMHz = true
    if useMHz
        scalefactor = 1000/(2*pi)
        unitStr = "MHz"
    else
        scalefactor = 1.0
        unitStr = "rad/ns"
    end

    # one subfigure for each control function
    plotarray = Array{Plots.Plot}(undef, params.Ncoupled) #empty array for separate plots
    plotarray_lab = Array{Plots.Plot}(undef, params.Ncoupled + params.Nunc) #empty array for separate plots

    println("Rotational frequencies: ", params.Rfreq)

    for q=1:params.Ncoupled
        # evaluate ctrl functions for the q'th Hamiltonian
        pfunc, qfunc = Juqbox.evalctrl(params, pcof, td, q)

        pfunc = scalefactor .* pfunc
        qfunc = scalefactor .* qfunc

        pmax = maximum(abs.(pfunc))
        qmax = maximum(abs.(qfunc))
        # first plot for control function for the symmetric Hamiltonian
        local titlestr = "Rotating frame coupled ctrl - " * string(q) * " Max-p=" *@sprintf("%.3e", pmax) * " Max-q=" *
            @sprintf("%.3e", qmax) * " " * unitStr
        plotarray[q] = Plots.plot(td, pfunc, lab=L"p(t)", title = titlestr, xlabel="Time [ns]",
                                  ylabel=unitStr, legend= :outerright)
        # add in the control function for the anti-symmetric Hamiltonian
        Plots.plot!(td, qfunc, lab=L"q(t)")

        println("Rot. frame ctrl-", q, ": Max-p(t) = ", pmax, " Max-q(t) = ", qmax, " ", unitStr)

        # Corresponding lab frame control
        omq = 2*pi*params.Rfreq[q]
        labdrive .= 2*pfunc .* cos.(omq*td) .- 2*qfunc .* sin.(omq*td)

        lmax = maximum(abs.(labdrive))
        local titlestr = "Lab frame coupled ctrl - " * string(q) * " Max=" *@sprintf("%.3e", lmax) * " " * unitStr
        plotarray_lab[q]= Plots.plot(td, labdrive, lab="", title = titlestr, size = (650, 250), xlabel="Time [ns]", ylabel=unitStr)

        println("Lab frame ctrl-", q, " Max amplitude = ", lmax, " ", unitStr)
        
        if save_files
            # Save ctrl functions on file
            pqname = ctrlbasename * "-" * string(q) * ".dat"
            writedlm(pqname, pfunc, qfunc)
            println("Saved ctrl functions for Hamiltonian #", q, " on file '", pqname, "', samplerate = ", samplerate);

            # save the lab frame ctrl func
            labname = labbasename * "-" * string(q) * ".dat"
            writedlm(labname, labdrive)
            println("Saved lab frame ctrl function on file '", labname, "'", " samplerate = ", samplerate);
        end
    end

    # Add in uncoupled controls
    if(params.Nunc >  0)
        max_uncoupled = zeros(length(params.Hunc_ops))
        for q =1:params.Nunc
            qu = 2*params.Ncoupled + q - 1
            pfunc = scalefactor .* Juqbox.evalctrl(params, pcof, td, qu)
            max_uncoupled[q] = maximum(abs.(pfunc))
            plotarray_lab[params.Ncoupled + q] =  Plots.plot(td, pfunc, lab="", linewidth = 2, title = "Uncoupled Ctrl Function",
                                                             size = (650, 250), xlabel="Time [ns]", ylabel=unitStr)
        end
        println("Max amplitude uncoupled ctrl = ", maximum(max_uncoupled), unitStr)

        # TODO: save uncoupled controls on file
    end

    # Accumulate all ctrl function sub-plots
    pl2  = Plots.plot(plotarray..., layout = (params.Ncoupled, 1))
    pl4  = Plots.plot(plotarray_lab..., layout = (params.Ncoupled, 1))

    if save_files
        rotplotname = ctrlbasename * ".png"
        Plots.savefig(pl2, rotplotname)
        println("Saved rotating frame ctrl plot on file '", rotplotname);

        local labplotname = labbasename * ".png"
        Plots.savefig(pl4, labplotname)
        println("Saved lab frame ctrl plot on file '", labplotname);
    end

    # plot the Fourier transform of the control function in the lab frame
    # Fourier transform
    nplot = length(labdrive)
    Fdr_lab = fftshift( fft(labdrive) ) / nplot
    dt = td[2] - td[1]
    freq = fftshift( fftfreq(nplot, 1.0/dt) )
    #pl5 = Plots.scatter(freq, abs.(Fdr_lab), lab="", title = "abs(FFT Lab frame)", size = (650, 250), xlabel="Frequency [GHz]", markersize=7)
    pl5 = Plots.plot(freq, abs.(Fdr_lab), lab="", title = "Spectrum, lab frame ctrl", size = (650, 350), xlabel="Frequency [GHz]",
                 ylabel="Amplitude " * unitStr, framestyle = :box, grid = :hide) #, yaxis=:log10)

    # messing with yticks
    # lb=0.0
    # ub = 0.01
    # ticks = range(lb,stop=ub,length=6)
    # Plots.plot!(yticks=ticks)

    fmin = 0.5*minimum(params.Rfreq) 
    fmax = maximum(params.Rfreq) + 0.5
    xlims!((fmin, fmax))
    # tmp
    # xlims!((3.0, 6.5))

    if save_files
        Plots.savefig(pl5, fftname)
    end

    # log-scale spectrum
    mag_Fdr_lab = abs.(Fdr_lab)
    if minimum(mag_Fdr_lab) > 0.0
        pl6 = Plots.plot(freq, mag_Fdr_lab, lab="", title = "Spectrum, lab frame ctrl", xlabel="Frequency [GHz]",
                         ylabel="Amplitude " * unitStr, yaxis=:log10)
        # tmp
        xlims!((3.0, 6.5))
        if save_files
            Plots.savefig(pl6, fftname2)
            println("Saved FFT of lab ctrl function on files '", fftname, "' and '", fftname2, "'");
        end
    end

    # Return an array of plot objects
    return [pl1, pl2, pl3, pl4, pl5, pl6, plcof, pconv]

end



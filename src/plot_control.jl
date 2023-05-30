"""
    pl = plot_control(params, pcof; samplerate = 32])

Create array of plot objects that can be visualized by, e.g., `display(pl[1])`.

# Arguments
- `params::objparams`: Object holding problem definition
- `pcof::Array{Float64,1}`: Parameter vector
- `samplerate:: Int64`: Default: `32` samples per unit time (ns). Sample rate for generating plots.
# Return arguments 
- `plotCtrl::Vector{Plots.plot}`: Plot of the rotating frame control functions
- `plCtrlFFT::Vector{Plots.plot}`: Plot of the Fourier amplitude of the laboratory frame control functions
"""
function plot_control(params::objparams, pcof::Vector{Float64}; samplerate::Int64=32)
   # Evaluate the ctrl functions on this grid in time
   nplot = round(Int64, params.T*samplerate)
   # is this resolution sufficient for the lab frame ctrl functions so we can get meaningful FFTs?
   td = collect(range(0, stop = params.T, length = nplot+1))

   # Initialize storing of the lab drive
   labdrive = zeros(nplot+1)

   nFFT = length(labdrive)
   dt = td[2] - td[1]
   freq = fftshift( fftfreq(nFFT, 1.0/dt) )
   
   useMHz = true
   if useMHz
       scalefactor = 1000/(2*pi)
       unitStr = "MHz"
   else
       scalefactor = 1.0
       unitStr = "rad/ns"
   end

   # one subfigure for each control function
   plotarray_ctrl = Array{Plots.Plot}(undef, params.Ncoupled+ params.Nunc) #empty array for separate plots
   plotarray_fft = Array{Plots.Plot}(undef, params.Ncoupled+ params.Nunc) #empty array for separate plots
   
   println("Rotational frequencies: ", params.Rfreq)

   for q=1:params.Ncoupled+params.Nunc
       # evaluate ctrl functions for the q'th Hamiltonian
       pfunc, qfunc = Juqbox.evalctrl(params, pcof, td, q)

       pfunc = scalefactor .* pfunc
       qfunc = scalefactor .* qfunc

       pmax = maximum(abs.(pfunc))
       qmax = maximum(abs.(qfunc))
       # first plot for control function for the symmetric Hamiltonian
       local titlestr = "Rotating frame ctrl - " * string(q) # * " Max-p=" *@sprintf("%.3e", pmax) * " Max-q=" *@sprintf("%.3e", qmax) * " " * unitStr
       plotarray_ctrl[q] = Plots.plot(td, pfunc, lab=L"p(t)", title = titlestr, xlabel="Time [ns]",
                                 ylabel=unitStr, legend= :outerright)
       # add in the control function for the anti-symmetric Hamiltonian
       Plots.plot!(td, qfunc, lab=L"q(t)")

       println("Rot. frame ctrl-", q, ": Max-p(t) = ", pmax, " Max-q(t) = ", qmax, " ", unitStr)

       # Corresponding lab frame control
       omq = 2*pi*params.Rfreq[q] # FIX index of Rfreq
       labdrive .= 2*pfunc .* cos.(omq*td) .- 2*qfunc .* sin.(omq*td)

       #lmax = maximum(abs.(labdrive))
       #local titlestr = "Lab frame ctrl - " * string(q) * " Max=" *@sprintf("%.3e", lmax) * " " * unitStr
       #plotarray_lab[q]= Plots.plot(td, labdrive, lab="", title = titlestr, size = (650, 250), xlabel="Time [ns]", ylabel=unitStr)

       #println("Lab frame ctrl-", q, " Max amplitude = ", lmax, " ", unitStr)
       
       # plot the Fourier transform of the control function in the lab frame
       # Fourier transform
       Fdr_lab = fftshift( fft(labdrive) ) / nFFT

       local titlestr = "Spectrum, lab frame ctrl - " * string(q)
       plotarray_fft[q] = Plots.plot(freq, abs.(Fdr_lab), lab="", title = titlestr, size = (650, 350), xlabel="Freq. [GHz]",
                                     ylabel="Amp.", framestyle = :box) #, grid = :hide

       fmin = 0.5*minimum(params.Rfreq) 
       fmax = maximum(params.Rfreq) + 0.5 # these limits are kind of arbitrary
       xlims!((fmin, fmax))

   end

    # Accumulate all ctrl function sub-plots
    #   pl2  = Plots.plot(plotarray_ctrl..., layout = (params.Ncoupled + params.Nunc, 1))
    #   pl5  = Plots.plot(plotarray_fft..., layout = (params.Ncoupled + params.Nunc, 1))
   
  return plotarray_ctrl, plotarray_fft
end
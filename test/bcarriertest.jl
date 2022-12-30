# bcarriertest(Nspline, kpar)
#
# INPUT:
# D1: number of spline basis functions per segment and frequency
# kpar: component of the gradient to plot

using Printf
using Plots
#pyplot()
using FFTW
using Random

using Juqbox

function plotCoupled()
  T= 25.0
  D1= 5
  om = [0.0 1.0; 0.0 2.0]
  Nctrl = size(om,1)
  Nfreq = size(om,2)
  Ncoeff = 2*D1*Nctrl*Nfreq

  Random.seed!(12345)
  pcof = 2*(rand(Ncoeff) .- 0.5)
  
  pl1, pl2 = bcarrierplot(T, D1, om, pcof)
  return pl1, pl2
end

function  bcarriertest(D1::Int64, kpar::Int64 = 1, num0::Int64 = 0 ) #num0 should be 0 or 2
    @assert(D1 >= 3)
    
    #Final time
    T   = 120.0
    dt  = 0.25

    Nsteps = ceil(Int64,T/dt)
    dt = T/Nsteps # correct dt

    println("Final time = ", T, ", number of time steps = ", Nsteps, ", time step =", dt)

    Ncoupled  = 1 # cntrl functions for coupled Hamiltonians
    Nunc  = 1   # cntrl functions for UNcoupled Hamiltonians
    Nfreq = 1   # 1 frequencies

    @assert(num0 < 2*(Ncoupled+Nunc)-1)
    num1 = num0 + 1
    
    Ncof = 2*(Ncoupled+Nunc) *D1*Nfreq # total # coefficients

    omega = zeros(Ncoupled+Nunc,Nfreq);
    waveNum = 10  # carrier wave frequency
    #  omega[1:Nseg,1] .= 0.0
    omega[1,1] = waveNum * 2*pi/T # Coupled
    omega[2,1] = waveNum * 2*pi/T # Un-coupled
    
    # assign values to the coefficients
    pcof = zeros(Ncof);
    g    = copy(pcof)

    # p1(t)
    pcof[1:D1] .= 1.0 # first freq, alpha_1
    # q1(t) part
    pcof[D1+1:2*D1] .= -0.5 #  first freq, alpha_2

    # p2(t)
    pcof[2*D1+1:3*D1] .= -2.0 # first freq, alpha_1
    # q2(t)
    pcof[3*D1+1:4*D1] .= 3.0 #  first freq, alpha_2

    bsparam = Juqbox.bcparams(T, D1, Ncoupled, Nunc, omega, pcof)
    @show(typeof(bsparam))
    
    td = collect(range(0,length = Nsteps, stop = T )) # collect turns the range object into a float64 array

    splinefcn(t, num) = Juqbox.bcarrier2(t, bsparam, num) # define shortcut to enable broadcasting

    # one figure for both cntrl function
    pl1 = plot(title="Control functions")

    ctrl = splinefcn.(td,num0)
    plot!(td,ctrl,lab="Re",linewidth=2)

    ctrl = splinefcn.(td,num1)
    plot!(td,ctrl,lab="Im",linewidth=2)

    # test gradient
    eps = 0.1
    pcof1 = copy(pcof) # needed for deep copy
    if num0 == 0
        pcof1[kpar] += eps
    elseif num0 == 2
        kpar +=  2*D1*Nfreq
        pcof1[kpar] += eps
    end
    @printf("kpar = %d, pcof0[kpar] = %e, pcof1[kpar] = %e, Ncof = %d\n", kpar, pcof[kpar], pcof1[kpar], Ncof)
    
    bsp1 = Juqbox.bcparams(T, D1, Ncoupled, Nunc, omega, pcof1)
    splfcn0(t, num) = Juqbox.bcarrier2(t, bsparam, num) # define shortcut to enable broadcasting
    splfcn1(t, num) = Juqbox.bcarrier2(t, bsp1, num) # define shortcut to enable broadcasting

    # one figure for both cntrl function
    pl2 = plot(title="Gradient of Control function")

    ctrl0 = splfcn0.(td,num0)
    ctrl1 = splfcn1.(td,num0)

    # gradient
    grad = zeros(Nsteps)
    for q in 1:Nsteps
        t = td[q]
        Juqbox.gradbcarrier2!(t, bsparam, num0, g)
        grad[q] += g[kpar]
    end

    plot!(td,grad, lab="Grad-Re")
    gradFD = (ctrl1 - ctrl0)./eps
    plot!(td,gradFD, lab="FD-Re", linestyle=:dash,linewidth=2)

    println("spline #", num0, " max(grad-FD) = ", maximum(grad - gradFD))

    ctrl0 = splfcn0.(td,num1)
    ctrl1 = splfcn1.(td,num1)

    # gradient
    grad = zeros(Nsteps)
    for q in 1:Nsteps
        t = td[q]
        Juqbox.gradbcarrier2!(t, bsparam, num1, g)
        grad[q] += g[kpar]
    end

    plot!(td,grad, lab="Grad-Im",linewidth=2)
    gradFD = (ctrl1 - ctrl0)./eps
    plot!(td, gradFD, lab="FD-Im", linestyle=:dash,linewidth=2)

    println("spline #", num1, " max(grad-FD) = ", maximum(grad - gradFD))

    return pl1, pl2
end

#-----------------------------------------------
# plot the control functions 
#-----------------------------------------------
function  bcarrierplot(T::Float64, D1::Int64, omega::Array{Float64,2}, pcof::Array{Float64,1}, basestr="Ctrl")

  bsparam = Juqbox.bcparams(T, D1, omega, pcof)

  #Final time
  samplerate = 16

  nplot = round(Int64, T*samplerate)
  dt    = 1.0/samplerate

  println("Final time = ", T, ", number of time steps = ", nplot, ", time step =", dt)

  @show(typeof(bsparam))
	
  td = collect(range(0, stop = T-dt, length = nplot)) # FFT

  # define shortcut to enable broadcasting
  splinefcn(t, num) = Juqbox.bcarrier2(t, bsparam, num) 

  Nsubpl = bsparam.Ncoupled
  plotarray = Array{Plots.Plot}(undef, Nsubpl) #empty array for separate plots

  for s in 1:Nsubpl
    # real part
    ctrlr = splinefcn.(td,s-1)
    labstr = string("Re");
    titlestr = string(basestr, ", osc-", s);
    plotarray[s] = plot(td, ctrlr, lab=labstr, linewidth=2, title=titlestr)
    # imag part
    ctrli = splinefcn.(td,s+Nsubpl-1)
    labstr = string("Im");
    plot!(td, ctrli, lab=labstr, linewidth=2)
  end

  # one figure for all cntrl functions
  pl1 = plot(plotarray..., layout = (Nsubpl,1))

  # Fourier transforms
  frequency = fftshift( AbstractFFTs.fftfreq(length(td), samplerate) )
  fmax = 0.75

  plotarray2 = Array{Plots.Plot}(undef, Nsubpl) #empty array for separate plots

  for s in 1:Nsubpl
    # real part
    ctrlr = splinefcn.(td,s-1)
    Fctrlr = fftshift( fft(ctrlr) ) / nplot
    labstr = string("Re");
    titlestr = string(basestr, ", osc-", s, ", Abs Fourier");
    plotarray2[s] = plot(frequency, abs.(Fctrlr), xlim=(-fmax,fmax), yscale= :log10, lab= labstr, title=titlestr, size = (1000, 500))
    # imag part
    ctrli = splinefcn.(td,s+Nsubpl-1)
    Fctrli = fftshift( fft(ctrli) ) / nplot
    labstr = string("Im");
    plot!(frequency, abs.(Fctrli), xlim=(-fmax,fmax), yscale= :log10, lab= labstr, xaxis="Freq [GHz]")
  end

  # one figure for all cntrl functions
  pl2 = plot(plotarray2..., layout = (Nsubpl,1))

  return pl1,pl2
end

#------------------------------------------------------------
function  testbcarrier(D1::Int64 = 5, kpar::Int64 = 5)
    @assert(D1 >= 1)
    
    T = 10.0
    samplerate = 20

    Nfreq    = 2;
    Ncoupled = 2; # Corresponds to the functions p1, q1, p2, q2
    Nunc     = 1  # number of uncoupled controls per oscillator
    nCoeff   = 2*(Ncoupled + Nunc)*Nfreq*D1 # Always an even number?
    nCoupled = 2*Ncoupled*Nfreq*D1 # Number of coupled coefficients

    @assert(kpar >= 1 && kpar <=nCoeff)
    
    # random coefficients in [-1,1]
    Random.seed!(12345)
    pcof = 2*(rand(nCoeff) .- 0.5);

    omega = zeros(Ncoupled, Nfreq);
    omega[1, 1] = 0
    omega[1, 2] = pi;
    
    bcpar = Juqbox.bcparams(T, D1, Ncoupled, Nunc, omega, pcof)

    # Evaluate the control functions at the discrete time levels
    dt    = 1.0/samplerate
    nplot = round(Int64, T*samplerate)
    td    = range(0, stop = T, length = nplot+1)

    println("Final time = ", T, ", nplot = ", nplot, ", plotting time step = ", dt)

    osc_ind = 2*div(kpar-1, Nfreq*D1*2) # kpar-1 because osc_ind is zero-based; 2* because osc_ind is even
    println("kpar: ", kpar, " Ncoupled: ", Ncoupled, " Nfreq: ", Nfreq, " D1: ", D1, " osc_ind: ", osc_ind)
    if(osc_ind > 2*Ncoupled-1)
      osc_ind = 0
    end
    f_ind   = max(div(kpar-1, Nfreq*D1),2*Ncoupled)
    println("f_ind: ", f_ind)

    pfunc(t) = Juqbox.bcarrier2(t, bcpar, osc_ind)   # pj(t)
    qfunc(t) = Juqbox.bcarrier2(t, bcpar, osc_ind+1) # qj(t)
    ffunc(t) = Juqbox.bcarrier2(t, bcpar, f_ind)     # fj(t) (uncoupled term)

    pfunc_grad!(t,g) = Juqbox.gradbcarrier2!(t, bcpar, osc_ind,g)
    qfunc_grad!(t,g) = Juqbox.gradbcarrier2!(t, bcpar, osc_ind+1,g)
    ffunc_grad!(t,g) = Juqbox.gradbcarrier2!(t, bcpar, f_ind,g)

    # FD perturbation
    eps = 1e-5

    # new object with new coefficient array
    pcof_p     = copy(pcof)
    bcpar_p    = Juqbox.bcparams(T, D1, Ncoupled, Nunc, omega, pcof)
    pfunc_p(t) = Juqbox.bcarrier2(t, bcpar_p, osc_ind)
    qfunc_p(t) = Juqbox.bcarrier2(t, bcpar_p, osc_ind+1)
    ffunc_p(t) = Juqbox.bcarrier2(t, bcpar_p, f_ind) 

    # forwards
    bcpar_p.pcof[kpar] += eps
    Bp = pfunc_p.(td) 
    Qp = qfunc_p.(td) 
    Fp = ffunc_p.(td) 

    # backwards
    bcpar_p.pcof[kpar] -= 2*eps
    Bm = pfunc_p.(td)
    Qm = qfunc_p.(td)
    Fm = ffunc_p.(td)

    pgrad_fd = 0.5.*(Bp - Bm)./eps
    qgrad_fd = 0.5.*(Qp - Qm)./eps
    fgrad_fd = 0.5.*(Fp - Fm)./eps

    # analytical gradient
    allgradp = zeros(length(td), nCoeff)
    allgradq = zeros(length(td), nCoeff)
    allgradf = zeros(length(td), nCoeff)

    grad1 = zeros(nCoeff)
    I1 = 1:length(td)
    for q=I1
        pfunc_grad!(td[q],grad1)
        allgradp[q,1:nCoeff] = grad1
        qfunc_grad!(td[q],grad1)
        allgradq[q,1:nCoeff] = grad1
        ffunc_grad!(td[q],grad1)
        allgradf[q,1:nCoeff] = grad1
    end


    # FD gradient
    pl1 = plot(td, pgrad_fd, lab = "FD-grad-p", size=(650,350), linewidth=2)
    plot!(td, qgrad_fd, lab = "FD-grad-q")
    plot!(td, fgrad_fd, lab = "FD-grad-f")

    # analytical gradient
    labstr = "grad-p-" * string(kpar)
    scatter!(td, allgradp[I1,kpar], lab = labstr, markersize=4, marker= :square)
    labstr = "grad-q-" * string(kpar)
    scatter!(td, allgradq[I1,kpar], lab = labstr, markersize=4, marker= :utriangle)
    labstr = "grad-f-" * string(kpar)
    scatter!(td, allgradf[I1,kpar], lab = labstr, markersize=4, marker= :star)

    return pl1

end

using LinearAlgebra
using Plots
using DelimitedFiles

function getMat(interval::Int64, N::Int64, type::Int64)
    if interval == 1
        return ones(N,N) .+ type
    else
        v = collect(range(1.0, 10.0, length=N*N)) .- type
        return reshape(v,N,N)
    end
end

function cons_test(pcof0::Vector{Float64}, e_con::Vector{Float64}, N::Int64, nTimeIntervals::Int64, nAlpha::Int64)
    nWinit = 2*N^2
    nMat = N^2

    #cons_idx = 0 # this variable is shared by all threads
    Threads.@threads for interval = 1:nTimeIntervals-1 # constraints only at interior time intervals

        #println("Interval # ", interval, " threadID = ", Threads.threadid())
        
        if interval == 1
            # initial conditions 
            Winit_r = ones(N,N)
            Winit_i = Matrix{Float64}(I,N,N)
        else
            # initial conditions from pcof0 (determined by optimization)
            
            rg1 = (nAlpha+(interval-2)*nWinit+1:nAlpha+(interval-2)*nWinit+nMat)
            Winit_r = reshape(pcof0[rg1], N, N)

            rg2 = nMat .+ rg1 
            Winit_i = reshape(pcof0[rg2], N, N)
        end

        # Evolve the state under Schroedinger's equation
        # NOTE: the S-V scheme treats the real and imaginary parts with different time integrators
        # First compute the solution operator for a basis of real initial conditions: I
        reInitOp = getMat(interval, N, 1)
        
        # Then a basis for purely imaginary initial conditions: iI
        imInitOp = getMat(interval, N, 2)
        
        # Now we can  account for the initial conditions for this time interval and easily calculate the gradient wrt Winit
        Uend_r = (reInitOp * Winit_r) # real part of above expression
        Uend_i = (imInitOp * Winit_i) # imaginary part

        # 1st the real part of all constraints for this interval
        ur_vec = vec(Uend_r)
        rg3 = nAlpha .+ (interval-1)*nWinit .+ (1:nMat)
        wr_vec = pcof0[rg3]
        # println("Size(Uend_r) = ", size(Uend_r), " size(ur_vec) = ", size(ur_vec), " size(wr_vec) = ", size(wr_vec) )
        rg4 = rg3 .- nAlpha # (interval-1)*nWinit+1:(interval-1)*nWinit+nMat
        e_con[rg4] = ur_vec - wr_vec # Cjump_r = Uend_r - Wend_r

        # 2nd the imaginary part of all constraints for this interval
        ui_vec = vec(Uend_i)
        rg5 = nAlpha .+ (interval-1)*nWinit+nMat .+ (1:nMat)
        wi_vec = pcof0[rg5]
        rg6 = rg5 .- nAlpha # ((interval-1)*nWinit+nMat+1:(interval-1)*nWinit+2*nMat)
        e_con[rg6] = ui_vec - wi_vec # Cjump_i = Uend_i - Wend_i

    end # end for interval
end

N = 3
nWinit = 2*N^2
nTimeIntervals = 4
nAlpha = 5
nCoeff = nAlpha + nWinit*(nTimeIntervals - 1)
pcof0 = collect(range(-5.0, 7.0, length=nCoeff))

e_con = zeros(nWinit*(nTimeIntervals - 1))

println("Number of threads: ", Threads.nthreads(), " #time-intervals: ", nTimeIntervals)
@time cons_test(pcof0, e_con, N, nTimeIntervals, nAlpha)

econ_ref = readdlm("econ-ref-4.dat") # for nTimeInt = 4
#println("e_con: ", e_con)
println("norm(e_con - econ_ref) = ", norm(e_con - econ_ref))
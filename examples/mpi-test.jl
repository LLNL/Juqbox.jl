using LinearAlgebra
using Plots
using DelimitedFiles
using MPI

"""
    split_count(N::Integer, n::Integer)

Return a vector of `n` integers which are approximately equally sized and sum to `N`.
"""
function split_count(N::Integer, n::Integer)
    q,r = divrem(N, n)
    return [i <= r ? q+1 : q for i = 1:n]
end

function getMat(interval::Int64, N::Int64, type::Int64)
    if interval == 1
        return ones(N,N) .+ type
    else
        v = collect(range(1.0, 10.0, length=N*N)) .- type
        return reshape(v,N,N)
    end
end

function cons_test(pcof0::Vector{Float64}, e_con::Vector{Float64}, myStartInt::Int64, myEndInt::Int64, N::Int64, nAlpha::Int64, nTimeIntervals::Int64)
    nWinit = 2*N^2
    nMat = N^2

    #cons_idx = 0 # this variable is shared by all threads
    for interval = myStartInt:myEndInt # constraints only at interior time intervals

        println("Interval # ", interval)
        
        if interval == 1
            # initial conditions 
            Winit_r = ones(N,N)
            Winit_i = Matrix{Float64}(I,N,N)
        else
            # initial conditions from pcof0 (determined by optimization)
            rg1 = nAlpha .+ (interval-2)*nWinit .+ (1:nMat)
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
        ur_vec = vec(reInitOp * Winit_r) # real part of above expression
        ui_vec = vec(imInitOp * Winit_i) # imaginary part
        
        if interval < nTimeIntervals
            # return equality constrains for the interior intervals
            # 1st the real part of all constraints for this interval
            rg3 = nAlpha .+ (interval-1)*nWinit .+ (1:nMat) # global index range
            wr_vec = pcof0[rg3] # pcof0 is a global vector
            
            rg4 = (interval - myStartInt)*nWinit .+ (1:nMat) # local index range
            e_con[rg4] = ur_vec - wr_vec # e_con is a local vector

            # 2nd the imaginary part of all constraints for this interval
            rg5 = rg3 .+ nMat
            wi_vec = pcof0[rg5] # pcof0 is a global vector
            
            rg6 = rg4 .+ nMat
            e_con[rg6] = ui_vec - wi_vec # e_con is a local vector
        else interval == nProcs 
            # just return the evolved state
            rg4 = (interval - myStartInt)*nWinit .+ (1:nMat)
            e_con[rg4] = ur_vec # e_con is a local vector
            rg6 = rg4 .+ nMat
            e_con[rg6] = ui_vec # e_con is a local vector
        end

    end # end for interval
end

function setup_mpi(nTimeIntervals::Int64, nWinit::Int64) # TODO: put this fcn in a struct
    MPI.Init()

    comm = MPI.COMM_WORLD # Global communicator
    myRank = MPI.Comm_rank(comm) # myRank
    nProcs = MPI.Comm_size(comm) # nProcs
    root = 0

    if nProcs > nTimeIntervals
        if myRank == root
            println("Error: nProc=$nProcs, > nTimeIntervals=$nTimeIntervals")
        end
        MPI.Abort(comm,-1)
    end

    if myRank == root
        println("Number of procs: ", nProcs, " #time-intervals: ", nTimeIntervals)
    end

    # local number of time intervals for evaluating constraints
    nIntervalsInRank = split_count(nTimeIntervals, nProcs) # split_count(nTimeIntervals-1, nProcs) 
    cum_counts = cumsum(nIntervalsInRank)

    if myRank == root
        println("# local time intervals: ", nIntervalsInRank, " cumsum: ", cum_counts)
    end

    if myRank == 0
        myStartInt = 1
    else
        myStartInt = cum_counts[myRank]+1
    end
    myEndInt = cum_counts[myRank+1] # arrays are 1-bound, myRank starts from 0

    for i = 0:nProcs-1
        if i == myRank
            println("rank ", myRank, " startInt ", myStartInt, " endInt ", myEndInt)
        end
        MPI.Barrier(comm)
    end

    # local number of elements in e_con_local for each rank
    nValuesProc = nWinit * nIntervalsInRank # this is a vector
    if myRank == root
        println("nValuesProc: ", nValuesProc, " sum: ", sum(nValuesProc))
    end

    return comm, myRank, nProcs, root, nIntervalsInRank, myStartInt, myEndInt, nValuesProc
end # setup_mpi()


nTimeIntervals = 4
N = 3 # Matrices are NxN
nWinit = 2*N^2

comm, myRank, nProcs, root, nIntervalsInRank, myStartInt, myEndInt, nValuesProc = setup_mpi(nTimeIntervals, nWinit)

# setup test parameters

nAlpha = 5
nCoeff = nAlpha + nWinit*(nTimeIntervals - 1) # pcof holds B-spline coefficients and initial cond's for the interior intervals
pcof0 = collect(range(-5.0, 7.0, length=nCoeff)) # make up a pcof array for testing

e_con_local  = zeros(nWinit*(myEndInt - myStartInt + 1))

# compute "my" part of econ_global
cons_test(pcof0, e_con_local, myStartInt, myEndInt, N, nAlpha, nTimeIntervals)

# println("Local e_con:")
# for i = 0:nProcs-1
#     if i == myRank
#         @show myRank, e_con_local
#         println()
#     end
#     MPI.Barrier(comm)
# end
# println()

# use Allgatherv! to communicate the local results to all ranks
e_con_global = zeros(nWinit*nTimeIntervals)

output_vbuf = VBuffer(e_con_global, nValuesProc)
MPI.Allgatherv!(e_con_local, output_vbuf, comm)

if myRank == root
    println("root e_con: ", e_con_global)

    fname = "econ-ref-" * string(nTimeIntervals) * ".dat"
    if isfile(fname)
        econ_ref = readdlm(fname) # read reference file
    else
        econ_ref = zeros(0)
    end

    if length(e_con_global) == length(econ_ref)
        println("Comparing to reference solution: norm(e_con - econ_ref) = ", norm(e_con_global - econ_ref))
    elseif nProcs == 1
        writedlm(fname, e_con_global)
        println("Saved reference solution on file: ", fname)
    end
end

MPI.Finalize()

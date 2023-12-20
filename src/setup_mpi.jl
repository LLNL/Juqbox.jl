"""
    split_count(N::Integer, n::Integer)

Return a vector of `n` integers which are approximately equally sized and sum to `N`.
"""
function split_count(N::Int64, n::Int64)
    q,r = divrem(N, n)
    return [i <= r ? q+1 : q for i = 1:n]
end

struct setup_mpi
    comm::MPI.Comm
    myRank::Int64
    nProcs::Int64
    root::Int64
    nTimeIntervals::Int64
    nIntervalsInRank::Vector{Int64}
    myStartInt::Int64 
    myEndInt::Int64
    output_vbuf:: MPI.VBuffer{Vector{Float64}}

    function setup_mpi(nTimeIntervals::Int64, nWinit::Int64, debug::Bool = false)

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

        if debug && myRank == root
            println("Number of procs: ", nProcs, " #time-intervals: ", nTimeIntervals)
        end

        # local number of time intervals for evaluating constraints
        nIntervalsInRank = split_count(nTimeIntervals, nProcs) # split_count(nTimeIntervals-1, nProcs) 
        cum_counts = cumsum(nIntervalsInRank)

        if debug && myRank == root
            println("# local time intervals: ", nIntervalsInRank, " cumsum: ", cum_counts)
        end

        if myRank == 0
            myStartInt = 1
        else
            myStartInt = cum_counts[myRank]+1
        end
        myEndInt = cum_counts[myRank+1] # arrays are 1-bound, myRank starts from 0

        if debug
            for i = 0:nProcs-1
                if i == myRank
                    println("rank ", myRank, " startInt ", myStartInt, " endInt ", myEndInt)
                end
                MPI.Barrier(comm)
            end
        end

            # allocate storage for the constraints
        nStateVecGlobal = nTimeIntervals*nWinit # 2*N^2 constraint per time interval
        state_vec_global = zeros(nStateVecGlobal)

        # local number of elements in state_vec_local for each rank
        nValuesProc = nWinit * nIntervalsInRank # nValuesProc is a vector

        if debug && myRank == root
            println("nValuesProc: ", nValuesProc, " sum: ", sum(nValuesProc))
            println("# global elements: ", nStateVecGlobal)
        end

        output_vbuf = MPI.VBuffer(state_vec_global, nValuesProc) # Buffer for Allgatherv!

        new(comm, myRank, nProcs, root, nTimeIntervals, nIntervalsInRank, myStartInt, myEndInt, output_vbuf)
    end 
end # struct setup_mpi
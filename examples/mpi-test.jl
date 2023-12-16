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

function cons_test(pcof0::Vector{Float64}, e_con::Vector{Float64}, my_startInt::Int64, my_endInt::Int64, N::Int64, nAlpha::Int64)
    nWinit = 2*N^2
    nMat = N^2

    #cons_idx = 0 # this variable is shared by all threads
    for interval = my_startInt:my_endInt # constraints only at interior time intervals

        println("Interval # ", interval)
        
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
        rg3 = nAlpha .+ (interval-1)*nWinit .+ (1:nMat) # global index range
        wr_vec = pcof0[rg3] # pcof0 is a global vector
        
        rg4 = (interval - my_startInt)*nWinit+1:(interval - my_startInt)*nWinit+nMat
        e_con[rg4] = ur_vec - wr_vec # e_con is a local vector

        # 2nd the imaginary part of all constraints for this interval
        ui_vec = vec(Uend_i)
        rg5 = rg3 .+ nMat
        wi_vec = pcof0[rg5] # pcof0 is a global vector
        
        rg6 = rg4 .+ nMat
        e_con[rg6] = ui_vec - wi_vec # e_con is a local vector

    end # end for interval
end

MPI.Init()

comm = MPI.COMM_WORLD # Global communicator
myRank = MPI.Comm_rank(comm) # myRank
nProcs = MPI.Comm_size(comm) # nProcs

root = 0

nTimeIntervals = 4
nInternalIntervals = nTimeIntervals - 1

if myRank == root
    println("Number of procs: ", nProcs, " #internal time-intervals: ", nInternalIntervals)
end

# local number of time intervals for evaluating constraints
nCounts = split_count(nTimeIntervals-1, nProcs) 
cum_counts = cumsum(nCounts)

if myRank == root
    println("# local time intervals: ", nCounts, " cumsum: ", cum_counts)
end

if myRank == 0
    my_startInt = 1
else
    my_startInt = cum_counts[myRank]+1
end
my_endInt = cum_counts[myRank+1] # arrays are 1-bound

println("rank ", myRank, " startInt ", my_startInt, " endInt ", my_endInt)

# setup test parameters

N = 3 # Matrices are NxN
nWinit = 2*N^2
nAlpha = 5
nCoeff = nAlpha + nWinit*(nTimeIntervals - 1)
pcof0 = collect(range(-5.0, 7.0, length=nCoeff))

e_con_local  = zeros(nWinit*(my_endInt - my_startInt + 1))

# compute "my" part of econ_global
cons_test(pcof0, e_con_local, my_startInt, my_endInt, N, nAlpha)

# assemble e_con_global
nValuesProc = nWinit * nCounts # this is a vector
if myRank == root
    println("nValuesProc: ", nValuesProc, " sum: ", sum(nValuesProc))
end

# use Allgatherv! to communicate the local results to all ranks
e_con_global = zeros(nWinit*nInternalIntervals)
# doesn't work!
# MPI.Allgatherv!(e_con_local, e_con_global, comm)

output_vbuf = VBuffer(e_con_global, nValuesProc)
MPI.Allgatherv!(e_con_local, output_vbuf, comm)

if myRank == root
    econ_ref = readdlm("econ-ref-4.dat") # for nTimeInt = 4
    println("root e_con: ", e_con_global)
    println("norm(e_con - econ_ref) = ", norm(e_con_global - econ_ref))
end

MPI.Finalize()

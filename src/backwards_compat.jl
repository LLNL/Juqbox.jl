# This version assumes full matrices. Compute C-> α*A*B + β*C
@inline function mul!(C::Array{Float64,M}, A::Array{Float64,N}, B::Array{Float64,M}, α::Float64, β::Float64) where {M,N}
    C .*= β
    szB1 = size(B,1)
    szB2 = size(B,2)
    szC2 = size(C,2)
    szC1 = size(C,1)
    for j=1:szC2
        offset = (j-1)*szB1
        offset2 = (j-1)*szC1
        for i=1:szC1
            tmp = 0.0
            for k=1:szB1
                ind2 = offset + k
                tmp += A[i,k]*B[ind2]
            end
            ind = offset2 + i
            C[ind] += α*tmp
        end
    end
end

# This version assumes A is a sparse matrix. Compute C-> α*A*B + β*C
@inline function mul!(C::Array{Float64,M}, A::SparseMatrixCSC{Float64,Int64}, B::Array{Float64,M}, α::Float64, β::Float64) where {M}
    C .*= β
    szB1 = size(B,1)
    szB2 = size(B,2)
    szA2 = size(A,2)
    szC1 = size(C,1)
    csc_ind = 1
    for j=1:szA2
        offset = (j-1)*szB1
        # offset2 = (j-1)*szC1
        start = A.colptr[j]
        stop = A.colptr[j+1]
        len = stop-start
        for i=1:len
            row = A.rowval[csc_ind]
            val = A.nzval[csc_ind]
            for k=1:szB2
                ind = (k-1)*szC1 + row
                ind2 = (k-1)*szB1 + j
                C[ind] += α*val*B[ind2]
            end
            csc_ind += 1
        end
    end
end

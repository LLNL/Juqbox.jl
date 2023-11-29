### Set up a test problem using one of the standard Hamiltonian models
# using Juqbox
using Printf
# using Plots
# using LinearAlgebra
# using Random
# using Dates

function check_unitarity(U_r::Matrix{Float64}, U_i::Matrix{Float64}; threshold::Float64=1.0e-2)
    Uc = U_r + 1im * U_i
    constraint = (Uc * Uc')
    for i = 1:size(Uc, 1)
        for j = 1:size(Uc, 2)
            if (i == j)
                test = abs(constraint[i, j] - 1.0)
            else
                test = abs(constraint[i, j])
            end
            if (test > threshold)
                @printf("(%d, %d): %.3E + %.3E im\n", i, j, real(constraint[i, j]), imag(constraint[i, j]))
            end
        end
    end
end

## sum(u .* conj(v)) = sum(ur .* vr + ui .* vi) + 1im * sum(ui .* vr - ur .* vi)
function inner_product(u_r::Vector{Float64}, u_i::Vector{Float64}, v_r::Vector{Float64}, v_i::Vector{Float64})
    uv_r = dot(u_r, v_r) + dot(u_i, v_i)
    uv_i = dot(u_i, v_r) - dot(u_r, v_i)

    return uv_r, uv_i
end

function norm_vec(u_r::Vector{Float64}, u_i::Vector{Float64})
    return sqrt(dot(u_r, u_r) + dot(u_i, u_i))
end

function unitarize(W_r::Matrix{Float64}, W_i::Matrix{Float64}, compute_grad::Bool)

    U_r = zeros(size(W_r))
    U_i = zeros(size(W_i))

    if (compute_grad)
        V_r = zeros(size(W_r))
        V_i = zeros(size(W_i))
    end

    # forward gram-schmidt
    for k = 1:size(U_r,1)
        vr = deepcopy(W_r[:, k])
        vi = deepcopy(W_i[:, k])
        for c = 1:k-1
            ur = U_r[:, c]
            ui = U_i[:, c]
            tmp_r, tmp_i = inner_product(vr, vi, ur, ui)
            vr -= tmp_r * ur - tmp_i * ui
            vi -= tmp_r * ui + tmp_i * ur
        end
        if (compute_grad)
            V_r[:, k] = vr
            V_i[:, k] = vi
        end
        vnorm = norm_vec(vr, vi)
        U_r[:, k] = vr / vnorm;
        U_i[:, k] = vi / vnorm;
    end

    if (compute_grad)
        return U_r, U_i, V_r, V_i
    else
        return U_r, U_i
    end

    # if (compute_grad)
    #     return W_r .* W_r, W_i, V_r, V_i
    # else
    #     return W_r .* W_r, W_i
    # end
end

function unitarize_adjoint(W_r::Matrix{Float64}, W_i::Matrix{Float64},
    V_r::Matrix{Float64}, V_i::Matrix{Float64}, U_r::Matrix{Float64}, U_i::Matrix{Float64}, 
    gradU_r::Matrix{Float64}, gradU_i::Matrix{Float64})

    # adjoint gram-schmidt
    Ws_r = zeros(size(U_r))
    Ws_i = zeros(size(U_r))
    Vs_r = zeros(size(U_r))
    Vs_i = zeros(size(U_r))
    
    for k = size(U_r,1):-1:1
        usr = deepcopy(gradU_r[:, k])
        usi = deepcopy(gradU_i[:, k])
        for c = k+1:size(U_r,1)
            wu_r, wu_i = inner_product(W_r[:,c], W_i[:,c], U_r[:,k], U_i[:,k])
            vsu_r, vsu_i = inner_product(Vs_r[:,c], Vs_i[:,c], U_r[:,k], U_i[:,k])

            usr -= wu_r * Vs_r[:,c] + wu_i * Vs_i[:,c] + vsu_r * W_r[:,c] + vsu_i * W_i[:,c]
            usi -= -wu_i * Vs_r[:,c] + wu_r * Vs_i[:,c] - vsu_i * W_r[:,c] + vsu_r * W_i[:,c]
        end

        vnorm = norm_vec(V_r[:, k], V_i[:, k])
        usu = dot(usr, U_r[:, k]) + dot(usi, U_i[:, k])
        Vs_r[:, k] = 1. / vnorm * (usr - usu * U_r[:, k])
        Vs_i[:, k] = 1. / vnorm * (usi - usu * U_i[:, k])

        wsr = deepcopy(Vs_r[:, k])
        wsi = deepcopy(Vs_i[:, k])
        vsr = Vs_r[:, k]
        vsi = Vs_i[:, k]
        for c = 1:k-1
            ur = U_r[:, c]
            ui = U_i[:, c]
            vsu_r, vsu_i = inner_product(vsr, vsi, ur, ui)
            wsr -= vsu_r * ur - vsu_i * ui
            wsi -= vsu_i * ur + vsu_r * ui
        end
        Ws_r[:, k] = wsr
        Ws_i[:, k] = wsi
    end

    return Ws_r, Ws_i
    # return 2.0 * W_r .* gradU_r, gradU_i
end
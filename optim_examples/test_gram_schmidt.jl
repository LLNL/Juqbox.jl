### Set up a test problem using one of the standard Hamiltonian models
# using Juqbox
using Printf
using Plots
using LinearAlgebra
using Random
using Dates
using Juqbox
# include("gram_schmidt.jl")

N = 4

W_r = 2.0 * rand(N, N) .- 1.0
W_i = 2.0 * rand(N, N) .- 1.0

Wc = W_r + 1im * W_i
Wc_adj = conj(Wc')
println(typeof(Wc))
# println(Wc)
# println(Wc_adj)

# println(Wc[:,1])

U_r, U_i, V_r, V_i = unitarize(W_r, W_i, true)
Uc = U_r + 1im * U_i
Uc_adj = Uc'
println(Uc * Uc_adj)
println(Uc_adj * Uc)

G_r = rand(N, N)
G_i = rand(N, N)
J0 = sum(G_r .* U_r) + sum(G_i .* U_i)
gradW_r, gradW_i = unitarize_adjoint(W_r, W_i, V_r, V_i, U_r, U_i, G_r, G_i)
gg = sum(gradW_r .* gradW_r) + sum(gradW_i .* gradW_i)
println("J0: ", J0)
println("|g|: ", sqrt(gg))

gnorm = sqrt(gg)
grad_r = gradW_r / gnorm;
grad_i = gradW_i / gnorm;
sc = 1.0 / gnorm;

Nk = 35
@printf("dx\tf1\tdfdx\terror\n");
for k = 1:Nk
    dx = 10^(-0.25 * k);
    W_r1 = W_r + dx * sc * grad_r;
    W_i1 = W_i + dx * sc * grad_i;

    U_r1, U_i1 = unitarize(W_r1, W_i1, false)
    J1 = sum(G_r .* U_r1) + sum(G_i .* U_i1)

    dJdx = (J1 - J0) / dx / sc;
    error = abs((dJdx - gnorm) / gnorm)
    @printf("%.5E\t%.5E\t%.5E\t%.5E\n", dx, J1, dJdx, error);
end

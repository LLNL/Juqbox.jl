Examples of the setup procedure can be found in the scripts in the `Juqbox.jl/examples` directory.
The examples are invoked by, e.g.
- `include("cnot1-setup.jl")`
The following cases are included:
- `rabi-setup.jl` Pi-pulse (X-gate) for a qubit, i.e. a Rabi oscillator.
- `cnot1-setup.jl` CNOT gate for a single qudit with 4 essential and 2 guard levels. 
- `flux-setup.jl` CNOT gate for single qubit with a flux-tuning control Hamiltonian.
- `cnot2-setup.jl` CNOT gate for a pair of coupled qubits with guard levels.
- `cnot3-setup.jl` Cross-resonance CNOT gate for a pair of qubits that are coupled by a cavity resonator.
**Note:** The last case reads an optimized solution from file.

## Risk-Neutral Optimization
In practice the entries of the Hamiltonian may have some uncertainty, especially for higher energy levels, and it is desirable to design control pulses that are more robust to noise. 
For both of the following examples, we consider a risk-neutral strategy to design a ``|0\rangle \leftrightarrow |2\rangle`` SWAP gate on a single qubit, with three essential levels and one guard level. Let ``\epsilon`` be a random variable and consider the uncertain system Hamiltonian ``H^{u}_s(\epsilon) = H^{rw}_s + H'(\epsilon)`` where ``H^{rw}_s`` is the system Hamiltonian in the rotating frame, and ``H'(\epsilon)`` is the uncertain diagonal perturbation:
```math
    \frac{H'(\epsilon)}{2\pi} = \begin{pmatrix}
      0 &  &  &  \\
       & \epsilon/100 &  &  \\
       &  & \epsilon/10 &  \\
       &  &  & \epsilon
    \end{pmatrix}.
```
We note that the uncertain system Hamiltonian has expectation
``\mathbb{E}[H^{u}_s(\epsilon)] = H^{rw}_s``. We may correspondingly
update the original objective function, ``\mathcal{G}(\bm{\alpha},H^{rw}_s)``, to the risk-neutral utility function ``\widetilde{\mathcal{G}}(\bm{\alpha}) = \mathbb{E}[\mathcal{G}(\bm{\alpha}, H^u_s(\epsilon))]``. For simple forms of the random variable ``\epsilon``, we may compute ``\widetilde{\mathcal{G}}`` by quadrature:
```math
    \mathbb{E}[\mathcal{G}(\alpha,H^u_s (\epsilon))] 
    = \int_{-\epsilon_\text{max}}^{ \epsilon_\text{max}}
       \mathcal{G}(\alpha, H^u_s(\epsilon)) \, d\epsilon
       \approx
       \sum_{k = 1}^M w_k \mathcal{G}(\alpha, H^u_s(\epsilon_k) ),
```
where ``w_k`` and ``\epsilon_k`` are the weights and collocation points
of a quadrature rule. 

Examples of the setup procedure can be found in the scripts in the `Juqbox.jl/examples/Risk_Neutral` directory.
The `run_all.jl` routine for both examples performs both a deterministic optimization, and a risk-neutral optimization
where the system Hamiltonian is perturbed by additive noise. Full details of the first example can be found in Section 6.2 of the manuscript found [here](https://arxiv.org/abs/2106.14310).

### Example 1 : Uniform Noise
For a first example, we let ``\epsilon \sim \text{Unif}(- \epsilon_\text{max}, \epsilon_\text{max})`` be a uniform random variable for some ``\epsilon_\text{max} > 0``. We may compute the risk-neutral objective function via a simple
Gauss-Legendre quadrature. To do this, Juqbox calls the `FastGaussianQuadrature.jl` package to generate 
appropriate nodes and weights via the call
```julia
nodes, weights = gausslegendre(nquad)
```
for `nquad` quadrature points. We note, however, that the usual Gauss-Legendre quadrature integrates over
the interval ``[-1,1]``. A simple modification of the nodes and weights then creates a quadrature rule 
in the interval ``(- \epsilon_\text{max}, \epsilon_\text{max})``:
```julia
# Map nodes to [-ϵ/2,ϵ/2]
nodes .*= 0.5*ep_max
weights .*= 0.5
```
Passing `nodes` and `weights` as keyword arguments to the `Juqbox.setup_ipopt_problem` routine then allows Juqbox to 
perform a risk-neutral optimization:
```julia
prob = Juqbox.setup_ipopt_problem(params,..., nodes=nodes, weights=weights)
```

### Example 2 : Bimodal Gaussian Noise
From the previous example, we see that performing a risk-neutral optimization
requires the user to provide an appropriate quadrature rule for a given noise model.
If one is interested in considered models with normally-distributed (Gaussian) noise,
we may instead use a Gauss-Hermite rule to generate quadrature for a risk-neutral
optimization. For a bimodal Gaussian distribution, for instance, we may specify an
array of means and standard deviations for each:
```julia
mean_vec = 2*pi.*[-1e-2 1e-2]
sig_vec = 2*pi*[1e-3, 1e-3]
```
With the mean and standard deviation for each Gaussian mode defined, we may then
generate a set of reference Gauss-Hermite quadrature weights and nodes (chosen specifically
for the ``e^{-x^2}`` weight function):
```julia
nodes_tmp, weights_tmp = gausshermite(nquad)
```
We note that this quadrature rule essentially integrates a function against a 
Gaussian with a mean of zero and unit variance. The simple transformation below 
obtains a rule for the above user-specified probability distribution:
```julia
# Make a larger node list for full distribution
n_modes = length(mean_vec)
nodes = zeros(n_modes*nquad)
weights = copy(nodes)
inv_n = 1.0/(n_modes*sqrt(pi))

# Mapping to reference weighting function
for i = 1:n_modes
    μ = mean_vec[i]
    σ = sig_vec[i]

	offset = (i-1)*nquad
	for j = 1:length(nodes_tmp)
		nodes[j + offset] = sqrt(2)*σ*nodes_tmp[j] + μ
        weights[j+offset] = weights_tmp[j]*inv_n
    end
end
```

### Other Noise Models
We note that by modifying the `eval_f_par` and `eval_grad_f_par` routines in `src/ipopt_interface.jl` file 
allows the use of other noise models for a risk-neutral optimization. Juqbox currently does not support 
time-dependent noise processes.




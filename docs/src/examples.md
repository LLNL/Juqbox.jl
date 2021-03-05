## Examples

Examples of the setup procedure can be found in the scripts in the `Juqbox.jl/examples` directory.
The examples are invoked by, e.g.
- `include("cnot1-setup.jl")`
The following cases are included:
- `rabi-setup.jl` Pi-pulse (X-gate) for a qubit, i.e. a Rabi oscillator.
- `cnot1-setup.jl` CNOT gate for a single qudit with 4 essential and 2 guard levels. 
- `flux-setup.jl` CNOT gate for single qubit with a flux-tuning control Hamiltonian.
- `cnot2-setup.jl` CNOT gate for a pair of coupled qubits with guard levels.
- `cnot3-setup.jl` Cross-resonance CNOT gate for a pair of qubits that are coupled by a cavity resonator.
**Note:** This case reads an optimized solution from file.

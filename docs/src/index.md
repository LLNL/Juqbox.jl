![Control functions](cnot2-pop.png)

# Juqbox.jl

Juqbox.jl is a package for solving quantum optimal control problems in closed quantum systems, where the evolution of the state vector is governed by Schroedinger's equation.

The main features of Juqbox include
- Symplectic time integration of Schroedinger's equation using the Stormer-Verlet scheme.
- Efficient parameterization of the control functions using B-splines with carrier waves.
- Objective function includes target gate infidelity and occupation of guarded (forbidden) states.
- Exact computation of the gradient of the objective function by solving the discrete adjoint equation.

The numerical methods in Juqbox.jl are documented in this report:
1. N. A. Petersson, F. M. Garcia, A. E. Copeland, Y. L. Rydin and J. L. DuBois, “Discrete Adjoints for Accurate Numerical Optimization with Application to Quantum Control”, LLNL-JRNL-800457, arXiv:2001.01013.


![Logo](JuQbox_logo-inline-color.png)

#

Juqbox.jl is a package for solving quantum optimal control problems in closed quantum systems, where the evolution of the state vector is governed by Schroedinger's equation.

The main features of Juqbox include
- Symplectic time integration of Schroedinger's equation using the Stormer-Verlet scheme.
- Efficient parameterization of the control functions via B-splines with carrier waves.
- Objective function includes target gate infidelity and occupation of guarded (forbidden) states.
- Exact computation of the gradient of the objective function by solving the discrete adjoint equation.

The numerical methods in Juqbox.jl are documented in these papers:
1. N. A. Petersson and F. M. Garcia, "Optimal Control of Closed Quantum Systems via B-Splines with Carrier Waves", SIAM J. Sci. Comput. (2022) 44(6): A3592-A3616, LLNL-JRNL-823853, [arXiv:2106.14310](https://arxiv.org/abs/2106.14310).
2. N. A. Petersson, F. M. Garcia, A. E. Copeland, Y. L. Rydin and J. L. DuBois, “Discrete Adjoints for Accurate Numerical Optimization with Application to Quantum Control”, LLNL-JRNL-800457, [arXiv:2001.01013](https://arxiv.org/abs/2001.01013).


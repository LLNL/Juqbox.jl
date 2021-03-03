# Juqbox.jl

## The **Juqbox.jl** package contains software for solving quantum optimal control problems in closed quantum systems, where the evolution of the state vector is governed by Schroedinger's equation.

## Installation
The following instructions assume that you have already installed Julia on your system, currently version 1.5.3. Before you do anything else, make sure you add the following line to your .bash_profile (or corresponding file):<br>
**export JULIA_PROJECT="@."**<br>
This environment variable tells Julia to look for Project.toml files in your current or parent directory.

### Building and testing **Juqbox**
shell> julia<br>
julia> ]<br>
(@v1.5) pkg> add  https://github.com/LLNL/Juqbox.jl.git<br>
(@v1.5) pkg> precompile
(@v1.5) pkg> test Juqbox
... all tests should pass ...<br>

To exit the package manager and Julia you do<br>
(@v1.5) pkg> (DEL) <br>
 
## Usage
To solve a quantum optimal control problem with the **Juqbox** package, the work flow consists of the following general steps:
1. Specify the problem
2. Optimize
3. Visualize the results


### 1. Specifying the problem
The setup phase includes specifying:
- The size of the state vector
- The system and control Hamiltonians
- The target gate transformation
- Duration of the gate and number of time steps for integrating Schroedinger's equation.
- Specifying carrier wave frequencies and the number of B-spline coefficients for parameterizing the control functions.

These properties are stored in a **mutable struct** that is populated by calling **params = Juqbox.objparams()**.<br>

The next steps are
- Assign the initial parameter vector (called **pcof0** in the examples below)
- Set bounds for the parameter vector to be imposed during the optimization
- Allocate working arrays by calling **wa = Juqbox.Working_arrays()**
- Assign convergence criteria and other parameters for the optimizer
- Build the optimization structure by calling **prob = Juqbox.setup_ipopt_problem()**

Examples of the setup procedure can be found in the following script in the **Juqbox.jl/examples** directory (invoke by, e.g. **include("cnot1-setup.jl")**) 
- **rabi-setup.jl** Pi-pulse (X-gate) for a qubit, i.e. a Rabi oscillator.
- **cnot1-setup.jl** CNOT gate for a single qudit with 4 essential and 2 guard levels. 
- **flux-setup.jl** CNOT gate for single qubit with a flux-tuning control Hamiltonian.
- **cnot2-setup.jl** CNOT gate for a pair of coupled qubits with guard levels.
- **cnot3-setup.jl** Cross-resonance CNOT gate for a pair of qubits that are coupled by a cavity resonator. **Note:** This case reads an optimized solution from file."

### 2. Optimization
Once you have been assigned the **params** and **prob** objects, as well as the initial parameter vector **pcof0**, the optimizer is invoked by
- **pcof = Juqbox.run_optimizer(prob, pcof0 [, jld2_filename])**

### 3. Visualizing the results
General properties of the optimized solution such as trace infidelity and unitary accuracy can be evaluated, and a number of figures can generated by invoking<br>
**pla = plot_results(params, pcof)**<br>
The figures are shown using, e.g. **display(pla[1])**, where **pla** is an array of Julia plot object. The following plot objects are populated:
- **pla[1]** Evolution of the state vector population
- **pla[2]** Control functions in the rotating frame of reference
- **pla[3]** Population of the forbidden energy levels
- **pla[4]** Lab frame control function(s)
- **pla[5]** Fourier transform of the lab-frame control functions (linear scale)
- **pla[6]** Fourier transform of the lab-frame control functions (log scale)
- **pla[7]** Coefficients of the optimized parameter vector
- **pla[8]** Convergence of the optimization

## Contributing to Juqbox.jl
Juqbox.jl is currently under significant development. The prefered method of contributing is through a pull request (PR). If you are interested in contributing, please contact Anders Petersson (petersson1@llnl.gov).

## Credits
Most of the Julia code was written by Anders Petersson and Fortino Garcia. Important contributions were also provided by Ylva Rydin and Austin Copeland. 

## License
Juqbox.jl is relased under the MIT license.

### Note: FFTW.jl license 
Juqbox.jl uses the Julia package FFTW.jl for post processing of the
results. That package is released under the MIT Expat license and provides Julia bindings to the
FFTW library for fast Fourier transforms (FFTs), as well as functionality useful for signal
processing. Note that the FFTW library is licensed under GPLv2 or higher (see its license file), but
the bindings to the FFTW library in the FFTW.jl package are licensed under MIT. As an lternative to
using the FFTW libary, the FFTs in Intel's Math Kernel Library (MKL) can be used by setting an
environment variable JULIA_FFTW_PROVIDER to MKL and running Pkg.build("FFTW"). MKL will be provided
through MKL_jll. Setting this environment variable only needs to be done for the first build of the
package; after that, the package will remember to use MKL when building and updating.

### Note: Ipopt.jl license 
Juqbox.jl uses the Julia package Ipopt.jl for optimizing control
functions. That package is released under the MIT Expat License and provides Julia bindings to the
Ipopt library, which is released under the Eclipse Public License.






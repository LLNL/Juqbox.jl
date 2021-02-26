### The **Juqbox** package contains software for solving quantum optimal control problems in closed quantum systems, where the evolution of the state vector is governed by Schroedinger's equation.

# Installation
The following instructions assume that you have already installed Julia on your system. Before you do anything else, make sure you add the following line to your .bash_profile (or corresponding file):<br>
**export JULIA_PROJECT="@."**<br>
This environment variable tells Julia to look for Project.toml files in your current or parent directory.

**Note:** If you are upgrading Julia to a new version, e.g. 1.4 -> 1.5, it may be necessary to first remove your ~/.julia directory: rm -rf ~/.julia. However, removing that directory should be considered as a last resort because it implies that you will have to re-install all of your packages.

## Downloading the **Juqbox** package
Clone the Juqbox package into a directory that is **NOT** a subdirectory of any other git repository (e.g. QCC). In this example we put it in the ~/src/Juqbox.jl directory:<br>
sh> cd ~/src<br>
sh> git clone https://lc.llnl.gov/bitbucket/scm/wave/Juqbox.jl.git<br>
<br>

## Note: FFTW.jl license
Juqbox.jl uses the Julia package FFTW.jl for post processing of the results. That package provides Julia
bindings to the FFTW library for fast Fourier transforms (FFTs), as well as functionality useful for
signal processing. Note that the FFTW library is licensed under GPLv2 or higher (see its license
file), but the bindings to the FFTW library in the FFTW.jl package are licensed under
MIT. As an lternative to using the FFTW libary, the FFTs in Intel's Math Kernel Library (MKL) can be used by setting an
environment variable JULIA_FFTW_PROVIDER to MKL and running Pkg.build("FFTW"). MKL will be provided
through MKL_jll. Setting this environment variable only needs to be done for the first build of the
package; after that, the package will remember to use MKL when building and updating.

## Installing Julia packages
Our software relies on the following Julia packages:
- IJulia (for Jupiter notebooks)
- Ipopt (optimizer)
- FFTW (Fourier transforms)
- Plots (for plotting)
- PyPlot (graphics driver)
- FileIO (reading and writing HDF5, JLD2, etc)
- JLD2 (HDF5 compatible files)
- LaTeXStrings (Using latex symbols in plot strings, etc.)
- Random (Random number generators)
- SparseArrays (Sparse matrix support)
- Printf (C-style printf macro)
- Test (for testing)
- Documenter (for stand-alone documentation)

You can check which packages you have installed by starting Julia and entering the package manager:<br>
sh> cd ~/src<br>
sh> julia <br>
julia> ] <br>
(@v1.5) pkg> status<br>

If any package is missing, you can add it using, e.g.,<br>
(@v1.5) pkg> add IJulia<br>

Since the **Juqbox** package lives in a separate git repository, it is installed in a slightly different way (assuming it is in the ~/src/Juqbox directory):<br>
(@v1.5) pkg> add ~/src/Juqbox.jl <br>

To exit the package manager and Julia you do<br>
(@v1.5) pkg> (DEL) <br>
julia> exit() <br>
sh> <br>
 
## Building and testing **Juqbox**
sh> cd ~/src/Juqbox<br>
sh> julia<br>
julia> ]<br>
(Juqbox) pkg> precompile<br>
(Juqbox) pkg> test<br>
... all tests should pass ...<br>
(Juqbox) pkg> (DEL)<br>
julia> exit()<br>

# Using the **Juqbox** package
To solve a quantum optimal control problem with the **Juqbox** package, the work flow consists of the following general steps:
1. Specify the problem
2. Optimize
3. Visualize the results


## 1. Specifying the problem
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

Examples of the setup procedure can be found in the following script in the **examples** directory (invoke by, e.g. **include("cnot1-setup.jl")**) 
- **rabi-setup.jl** Pi-pulse (X-gate) for a qubit, i.e. a Rabi oscillator.
- **cnot1-setup.jl** CNOT gate for a single qudit with 4 essential and 2 guard levels. 
- **flux-setup.jl** CNOT gate for single qubit with a flux-tuning control Hamiltonian.
- **cnot2-setup.jl** CNOT gate for a pair of coupled qubits with guard levels.
- **cnot3-setup.jl** Cross-resonance CNOT gate for a pair of qubits that are coupled by a cavity resonator. **Note:** This case reads an optimized solution from file."

## 2. Optimization
Once you have been assigned the **params** and **prob** objects, as well as the initial parameter vector **pcof0**, the optimizer is invoked by
- **pcof = Juqbox.run_optimizer(prob, pcof0 [, jld2_filename])**

## 3. Visualizing the results
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








![Control functions](examples/cnot2-pop.png)

# Juqbox.jl

Juqbox.jl is a package for solving quantum optimal control problems in closed quantum systems, where the evolution of the state vector is governed by Schroedinger's equation.

## Installation
The following instructions assume that you have already installed Julia (currently version 1.5.3) on your system. Before you do anything else, make sure you add the following line to your .bash_profile (or corresponding file):<br>
**export JULIA_PROJECT="@."**<br>
This environment variable tells Julia to look for Project.toml files in your current or parent directory.

### Building and testing **Juqbox**
shell> julia<br>
julia> ]<br>
(@v1.5) pkg> add  https://github.com/LLNL/Juqbox.jl.git<br>
(@v1.5) pkg> precompile<br>
(@v1.5) pkg> test Juqbox<br>
... all tests should pass ...<br>

To exit the package manager and Julia you do<br>
(@v1.5) pkg> (DEL) <br>
julia> exit()
 
## Documentation

The Juqbox.jl documentation can be found [here](https://software.llnl.gov/Juqbox.jl/).

## Examples

Examples of the setup procedure can be found in the following scripts in the **Juqbox.jl/examples** directory (invoke by, e.g. **include("cnot1-setup.jl")**) 
- **rabi-setup.jl** Pi-pulse (X-gate) for a qubit, i.e. a Rabi oscillator.
- **cnot1-setup.jl** CNOT gate for a single qudit with 4 essential and 2 guard levels. 
- **flux-setup.jl** CNOT gate for single qubit with a flux-tuning control Hamiltonian.
- **cnot2-setup.jl** CNOT gate for a pair of coupled qubits with guard levels.
- **cnot3-setup.jl** Cross-resonance CNOT gate for a pair of qubits that are coupled by a cavity resonator. **Note:** This case reads an optimized solution from file."

## Contributing to Juqbox.jl
Juqbox.jl is currently under development. The prefered method of contributing is through a pull request (PR). If you are interested in contributing, please contact Anders Petersson (petersson1@llnl.gov) or Fortino Garcia (fortino.garcia@colorado.edu).

## Credits
Most of the Julia code was written by Anders Petersson and Fortino Garcia. Important contributions were also made by Ylva Rydin and Austin Copeland. 

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






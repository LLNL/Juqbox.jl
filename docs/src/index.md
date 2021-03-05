# Juqbox.jl Documentation

## Installation
The following instructions assume that you have already installed Julia (currently version 1.5.3) on your system. Before installing Juqbox, we recommend that you add the following line to your .bash_profile (or corresponding file):

**export JULIA_PROJECT="@."**

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


## Index
```@index
Modules = [Juqbox]
```






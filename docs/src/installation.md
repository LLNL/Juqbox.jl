The following instructions assume that you have already installed Julia (currently version 1.6.3) on your system. Before proceeding, we recommend that you add the following lines to the file ~/.julia/config/startup.jl. You may have to first create the config folder under .julia in your home directory. Then add the following lines to the startup.jl file:

- **ENV["JULIA_PROJECT"]="@."**
- **ENV["PLOTS_DEFAULT_BACKEND"]="PyPlot"**

These are environment variables. The first one tells Julia to look for `Project.toml` files in your current or parent directory. The second one specifies the backend for plotting. Most of the examples in this document uses the PyPlot backend, which assumes that you have installed that package. If you have trouble installing PyPlot, you can instead install the "GR" package and set the default backend to "GR".

Start julia and type `]` to enter the package manager. Then do:
- (@v1.6) pkg> add  https://github.com/LLNL/Juqbox.jl.git
- (@v1.6) pkg> precompile
- (@v1.6) pkg> test Juqbox
- ... all tests should pass ...

To exit the package manager you type `<DEL>`, and to exit julia you type `exit()`.

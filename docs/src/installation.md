The following instructions assume that you have already installed Julia (currently version 1.8.5) on your system. Before proceeding, we recommend that you add the following to the file ~/.julia/config/startup.jl. You may have to first create the config folder under .julia in your home directory. Then add this line to the startup.jl file:

- **ENV["PLOTS_DEFAULT_BACKEND"]="GR"**

This is an environment variable. It specifies the backend for plotting. Most of the examples in this document uses the GR backend, which assumes that you have installed that package. If you have trouble with GR, you can instead install the "PyPlot" package and set the default backend to "PyPlot".

Start julia and type `]` to enter the package manager. Then do:
- (@v1.8) pkg> add  https://github.com/LLNL/Juqbox.jl
- (@v1.8) pkg> precompile
- (@v1.8) pkg> test Juqbox
- ... all tests should pass (case=flux gives a Warning message) ...

To exit the package manager you type `<DEL>`, and to exit julia you type `exit()`.

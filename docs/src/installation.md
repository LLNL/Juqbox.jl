The following instructions assume that you have already installed Julia (currently version 1.5.3) on your system. Before proceeding, we recommend that you add the following line to your `.bash_profile` (or corresponding) file:

**export JULIA_PROJECT="@."**

This environment variable tells Julia to look for `Project.toml` files in your current or parent directory.

Start julia and type `]` to enter the package manager. Then do:
- (@v1.5) pkg> add  https://github.com/LLNL/Juqbox.jl.git
- (@v1.5) pkg> precompile
- (@v1.5) pkg> test Juqbox
- ... all tests should pass ...

To exit the package manager you type `<DEL>`, and to exit julia you type `exit()`.

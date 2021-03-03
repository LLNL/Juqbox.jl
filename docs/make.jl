using Documenter, DocumenterLaTeX, Juqbox

clean = true
doctest = false # set to true once initial implementation is done


makedocs(
#    format= LaTeX(), # doesn't work?
    modules = [Juqbox],
    sitename="Juqbox.jl",
    repo = "https://github.com/LLNL/Juqbox.jl.git",
    root = ".",
    source = "src",
    build = "build"
)

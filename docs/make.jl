using Documenter, DocumenterLaTeX, Juqbox
root = ""
source = "src"
build = "build"
clean = true
doctest = false # set to true once initial implementation is done

repo = "https://lc.llnl.gov/bitbucket/scm/wave/juqbox.jl.git"


makedocs(
#    format= LaTeX(), # doesn't work?
    modules = [Juqbox],
    sitename="Juqbox.jl"
)

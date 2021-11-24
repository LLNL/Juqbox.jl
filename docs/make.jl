using Documenter, DocumenterTools, Juqbox

const ROOT = joinpath(@__DIR__, "..")



makedocs(
    modules = [Juqbox],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    #format = Documenter.LaTeX(platform = "docker"), # errors with no such file or directory???
    #clean = false,
    sitename="Juqbox.jl",
    authors = "Anders Petersson, Fortino Garcia, and contributors.",
    pages = [
        "Home" => "index.md",
        "Workflow" => "workflow.md",
        "Examples" => "examples.md",
        "Types" => "types.md",
        "Methods" => "methods.md",
        "Index" => "function-index.md",
    ],
    doctest = true, # set to true once initial implementation is done
    source = "src",
)

deploydocs(
    repo = "github.com/LLNL/Juqbox.jl",
    target = "build",
    devurl="docs",
)

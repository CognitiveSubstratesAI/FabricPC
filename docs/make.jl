using Documenter
using FabricPC

DocMeta.setdocmeta!(FabricPC, :DocTestSetup, :(using FabricPC); recursive=true)

makedocs(;
    modules=[FabricPC],
    authors="CognitiveSubstrates AI",
    repo=Remotes.GitHub("CognitiveSubstratesAI", "FabricPC"),
    sitename="FabricPC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cognitivesubstratesai.github.io/FabricPC/stable/",
        edit_link="main",
        assets=String[]
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting_started/installation.md",
            "Quickstart" => "getting_started/quickstart.md",
            "Architecture" => "getting_started/architecture.md",
            "JIT with Reactant" => "getting_started/jit.md"
        ],
        "API" => "api/index.md"
    ],
    warnonly=[:missing_docs, :cross_references, :docs_block]
)

deploydocs(;
    repo="github.com/CognitiveSubstratesAI/FabricPC",
    devbranch="main",
    push_preview=true
)

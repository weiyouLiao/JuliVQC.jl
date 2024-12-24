using Documenter
using JuliVQC 

makedocs(
    sitename = "JuliVQC.jl",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "gettingstarted.md",
        "Manual" => "manual.md",
        "Examples" => "examples.md"
    ],
    format = Documenter.HTML(
        repolink = "https://github.com/weiyouLiao/JuliVQC.jl" # 显式指定仓库链接
    ),
    repo = "https://github.com/weiyouLiao/JuliVQC.jl", # 指定仓库 URL
    clean = true
)


deploydocs(
    repo = "github.com/weiyouLiao/JuliVQC.jl.git", 
    devbranch = "main", 
    target = "gh-pages"
)

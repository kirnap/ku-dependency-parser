let _build = false
    for p in ("AutoGrad","ArgParse","Compat","JLD","Knet")
        if Pkg.installed(p) == nothing
            Pkg.add(p)
            _build = true
        end
    end
    if _build
        Pkg.build("Knet")
    end
end

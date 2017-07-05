# features that are being used by parser, taken from :
# https://github.com/denizyuret/KUparser.jl/blob/conll17/src/features.jl

function getanchor(f::AbstractString, p::Parser, ldep, rdep)
    f1 = f[1]; f2 = f[2]; flen = length(f)
    if isdigit(f2)
        i = f2 - '0'            # index
        n = 3                   # next character
    else
        i = 0
        n = 2
    end
    a = 0                       # target word
    if f1 == 's'
        if (p.sptr - i >= 1) # 25
            a = Int(p.stack[p.sptr - i])             # 456
        end
    elseif f1 == 'n'
        if p.wptr + i <= p.nword
            a = p.wptr + i
        end
    else 
        error("feature string should start with [sn]")
    end
    if a==0; return 0; end
    while n < flen
        f1 = f[n]; f2 = f[n+1]
        if isdigit(f2)
            i = f2 - '0'
            n = n+2
        else
            i = 1
            n = n+1
        end
        if i <= 0
            error("hlr indexing is one based") # 3 [lrh] is one based, [sn] was zero based
        end
        if f1 == 'l'                              # 2
            if a > p.wptr; error("buffer words other than n0 do not have ldeps"); end # 252
            if isassigned(ldep,a) && i <= length(ldep[a])
                a = Int(ldep[a][i])
            else
                return 0
            end
        elseif f1 == 'r'
            if a >= p.wptr; error("buffer words do not have rdeps"); end
            if isassigned(rdep,a) && i <= length(rdep[a])
                a = Int(rdep[a][i])
            else
                return 0
            end
        elseif f1 == 'h'
            for j=1:i       # 5
                a = Int(p.head[a]) # 147
                if a == 0
                    return 0
                end
            end
        else 
            break
        end
    end
    if n != length(f); error("n!=length(f)"); end
    return a
end


function getdeps2{T<:Parser}(p::T)
    nw = p.nword
    nd = p.ndeps
    dr = p.deprel
    ldep = Array(Any,nw)
    rdep = Array(Any,nw)
    @inbounds for d=1:nw
        h=Int(p.head[d])
        if h==0
            continue
        elseif d<h
            if !isassigned(ldep,h)
                ldep[h]=[d]
            else
                push!(ldep[h],d)
            end
        elseif d>h
            if !isassigned(rdep,h)
                rdep[h]=[d]
            else
                unshift!(rdep[h],d)
            end
        else
            error("h==d")
        end
    end
    return (ldep,rdep)
end


function getrdist(f::AbstractString, p::Parser, a::Integer)
    if f[1]=='s' && a > 0
        if isdigit(f[2])
            i=f[2]-'0'
        else
            i=0
        end
        if i>0
            return p.stack[p.sptr - i + 1] - a
        elseif p.wptr <= p.nword
            return p.wptr - a
        else
            return 0
        end
    else
        return 0 # dist only defined for stack words
    end
end


function features(parsers, feats, model)
    pvecs,dvecs,lvecs,rvecs,xvecs = postagv(model),deprelv(model),lcountv(model),rcountv(model),distancev(model)
    pvec0,dvec0,lvec0,rvec0,xvec0 = zeros(pvecs[1]),zeros(dvecs[1]),zeros(lvecs[1]),zeros(rvecs[1]),zeros(xvecs[1])
    s = parsers[1].sentence
    wvec0,fvec0,bvec0 = zeros(s.wvec[1]),zeros(s.fvec[1]),zeros(s.bvec[1])
    fmatrix = []
    for p in parsers
        s = p.sentence
        ldep,rdep = getdeps2(p)
        for f in feats
            a = getanchor(f,p,ldep,rdep)
            fn = f[end]
            if fn == 'v'
                push!(fmatrix, a>0 ? s.wvec[a] : wvec0)
            elseif fn == 'c'
                push!(fmatrix, a>0 ? s.fvec[a] : fvec0)
                push!(fmatrix, a>0 ? s.bvec[a] : bvec0)
            elseif fn == 'p'
                push!(fmatrix, a>0 ? pvecs[s.postag[a]] : pvec0)
            elseif fn == 'L'
                push!(fmatrix, a>0 ? dvecs[p.deprel[a]] : dvec0)
            elseif fn == 'A'
                push!(fmatrix, a==0 ? dvec0 : !isassigned(ldep,a) ? dvec0 : length(ldep[a])==1 ? dvecs[p.deprel[ldep[a][1]]] : sum(dvecs[p.deprel[ldep[a]]]))
            elseif fn == 'B'
                push!(fmatrix, a==0 ? dvec0 : !isassigned(rdep,a) ? dvec0 : length(rdep[a])==1 ? dvecs[p.deprel[rdep[a][1]]] : sum(dvecs[p.deprel[rdep[a]]]))
            elseif fn == 'd'
                d = getrdist(f,p,a)
                push!(fmatrix, d>0 ? xvecs[min(d,10)] : xvec0)
            elseif fn == 'a'
                d = (a>0 && isassigned(ldep,a) ? length(ldep[a]) : 0)
                push!(fmatrix, a>0 ? lvecs[min(d+1,10)] : lvec0)
            elseif fn == 'b'
                d = (a>0 && isassigned(rdep,a) ? length(rdep[a]) : 0)
                push!(fmatrix, a>0 ? rvecs[min(d+1,10)] : rvec0)
            elseif fn == 'w'
                error("Dense features do not support 'w'")
            else
                error("Unknown feature $(fn)")
            end
        end
    end
    fmatrix = vcatn(fmatrix...)
    ncols = length(parsers)
    nrows = div(length(fmatrix), ncols)
    reshape(fmatrix, nrows, ncols)
end

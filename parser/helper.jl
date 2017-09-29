# General helper functions, This implementation is used in julia5

# when fmatrix has mixed Rec and KnetArray, vcat does not do the right thing!  AutoGrad only looks at the first 2-3 elements!
#TODO: temp solution to AutoGrad vcat issue:
using AutoGrad
let cat_r = recorder(cat); global vcatn, hcatn
    function vcatn(a...)
        if any(x->isa(x,Rec), a)
            cat_r(1,a...)
        else
            vcat(a...)
        end
    end
    function hcatn(a...)
        if any(x->isa(x,Rec), a)
            cat_r(2,a...)
        else
            hcat(a...)
        end
    end
end

# KnetArray cleaning
function free_KnetArray()
    gc(); Knet.knetgc(); gc()
end


# To fill the constant word embeddings
function fillwvecs!(sentences, isents, wembed; GPUFEATS=false)
    for (s, isents) in zip(sentences, isents)
        empty!(s.wvec)
        for w in isents
            if GPUFEATS
                push!(s.wvec, wembed[:, w])
            else
                push!(s.wvec, Array(wembed[:, w]))
            end
        end
    end
end


# To fill the context embeddings of words in given sentences one by one
function fillcvecs!(sentences, forw, back; GPUFEATS=false)
    T = length(forw)
    for i in 1:length(sentences)
        s = sentences[i]
        empty!(s.fvec)
        empty!(s.bvec)
        N = length(s)
        for n in 1:N
            t = T-N+n
            if GPUFEATS #GPU
                push!(s.fvec, forw[t][:,i])
                push!(s.bvec, back[t][:,i])
            else #CPU
                push!(s.fvec, Array(forw[t][:,i]))
                push!(s.bvec, Array(back[t][:,i]))
            end
        end
    end
end


# fill both word and context embeddings
function fillvecs!(wmodel, sentences, vocab; batchsize=128)

    words, sents, maxwordlen, maxsentlen = maptoint(sentences, vocab)
    sow = vocab.cdict[vocab.sowchar]
    eow = vocab.cdict[vocab.eowchar]
    paw = vocab.cdict[vocab.unkchar]
 
    # word-embeddings calcutation
    wembed = Any[]
    free_KnetArray();
    for i=1:batchsize:length(words)
        j = min(i+batchsize-1,length(words))
        wij = view(words,i:j)
        maxij = maximum(map(length, wij))
        cdata, cmask = tokenbatch(wij, maxij, sow, eow)
        push!(wembed, charlstm(wmodel, cdata, cmask))
    end
    wembed = hcatn(wembed...)
    fillwvecs!(sentences, sents, wembed)

    sos,eos,unk = vocab.idict[vocab.sosword], vocab.idict[vocab.eosword], vocab.odict[vocab.unkword]
    result = zeros(2)
    free_KnetArray()
    for i=1:batchsize:length(sents)
        j = min(i+batchsize-1, length(sents))
        isentij = view(sents, i:j)
        maxij = maximum(map(length, isentij))
        wdata, wmask = tokenbatch(isentij, maxij, sos, eos)
        forw, back = wordlstm(wmodel, wdata, wmask, wembed)
        sentij = view(sentences, i:j)
        fillcvecs!(sentij, forw, back)
        odata, omask = goldbatch(sentij, maxij, vocab.odict, unk)
        lmloss(wmodel,odata,omask,forw,back; result=result) 
    end
    return exp(-result[1]/result[2])
end


# calculate the coverage
function unkrate(sentences)
    total = known = 0
    for s in sentences
        for w in s.word
            total += 1
            known += haskey(s.vocab.odict, w)
        end
    end
    return (total-known,total)
end


# convert lm-trained file to new col-major parser style file
function convertfile(infile, outfile)
    atr(x)=transpose(Array(x)) # LM stores in row-major Array, we use col-major Array
    d = load(infile); m = d["model"]
    save(outfile, "cembed", atr(m[:cembed]), "forw", map(atr,m[:forw]),
         "back",map(atr,m[:back]), "soft",map(atr,m[:soft]), "char",map(atr,m[:char]),
         "char_vocab",d["char_vocab"], "word_vocab",d["word_vocab"], 
         "sosword","<s>","eosword","</s>","unkword","<unk>",
         "sowchar",'\x12',"eowchar",'\x13',"unkchar",'\x11')
end


# convert KnetArrays to Arrays
function convert2cpu(infile, outfile)
    d = load(infile)
    jldopen(outfile, "w") do file
        for (k,v) in d
            write(file,k,map2cpu(v))
        end
    end
end


# save file in new format
function savefile(file, vocab, wmodel, pmodel, optim, arctype, feats)
    save(file,  "cembed", map2cpu(cembed(wmodel)),
         "char", map2cpu([wchar(wmodel),bchar(wmodel)]),
         "forw", map2cpu([wforw(wmodel),bforw(wmodel)]),
         "back", map2cpu([wback(wmodel),bback(wmodel)]),
         "soft", map2cpu([wsoft(wmodel),bsoft(wmodel)]),

         "char_vocab",vocab.cdict, "word_vocab",vocab.odict,
         "sosword",vocab.sosword,"eosword",vocab.eosword,"unkword",vocab.unkword,
         "sowchar",vocab.sowchar,"eowchar",vocab.eowchar,"unkchar",vocab.unkchar,
         "postags",vocab.postags,"deprels",vocab.deprels,

         "postagv",map2cpu(postagv(pmodel)),"deprelv",map2cpu(deprelv(pmodel)),
         "lcountv",map2cpu(lcountv(pmodel)),"rcountv",map2cpu(rcountv(pmodel)),
         "distancev",map2cpu(distancev(pmodel)),"parserv",map2cpu(parserv(pmodel)),

         "postago",map2cpu(postagv(optim)),"deprelo",map2cpu(deprelv(optim)),
         "lcounto",map2cpu(lcountv(optim)),"rcounto",map2cpu(rcountv(optim)),
         "distanceo",map2cpu(distancev(optim)),"parsero",map2cpu(parserv(optim)),
    
         "arctype",arctype,"feats",feats,
    )
end


function writeconllu(sentences, inputfile, outputfile)
    # We only replace the head and deprel fields of the input file
    out = open(outputfile,"w")
    v = sentences[1].vocab
    deprels = Array(String, length(v.deprels))
    for (k,v) in v.deprels; deprels[v]=k; end
    s = p = nothing
    ns = nw = nl = 0
    for line in eachline(inputfile)
        nl += 1
        if ismatch(r"^\d+\t", line)
            # info("$nl word")
            if s == nothing
                s = sentences[ns+1]
                p = s.parse
            end
            f = split(line, '\t')
            nw += 1
            if f[1] != "$nw"; error(); end
            if f[2] != s.word[nw]; error(); end
            f[7] = string(p.head[nw])
            f[8] = deprels[p.deprel[nw]]
            print(out, join(f, "\t"))
        else
            if line == "\n"
                # info("$nl blank")
                if s == nothing; error(); end
                if nw != length(s.word); error(); end
                ns += 1; nw = 0
                s = p = nothing
            else
                # info("$nl non-word")
            end
            print(out, line)
        end
    end
    if ns != length(sentences); error(); end
    close(out)
end


map2cpu(x)=(if isbits(x); x; else; map2cpu2(x); end)
map2cpu(x::KnetArray)=Array(x)
map2cpu(x::Tuple)=map(map2cpu,x)
map2cpu(x::AbstractString)=x
map2cpu(x::DataType)=x
map2cpu(x::Array)=map(map2cpu,x)
map2cpu{T<:Number}(x::Array{T})=x
map2cpu(x::Associative)=(y=Dict();for (k,v) in x; y[k] = map2cpu(x[k]); end; y)
map2cpu2(x)=(y=deepcopy(x); for f in fieldnames(x); setfield!(y,f,map2cpu(getfield(x,f))); end; y)

map2gpu(x)=(if isbits(x); x; else; map2gpu2(x); end)
map2gpu(x::KnetArray)=x
map2gpu(x::AbstractString)=x
map2gpu(x::DataType)=x
map2gpu(x::Tuple)=map(map2gpu,x)
map2gpu(x::Array)=map(map2gpu,x)
map2gpu{T<:AbstractFloat}(x::Array{T})=KnetArray(x)
map2gpu(x::Associative)=(y=Dict();for (k,v) in x; y[k] = map2gpu(x[k]); end; y)
map2gpu2(x)=(y=deepcopy(x); for f in fieldnames(x); setfield!(y,f,map2gpu(getfield(x,f))); end; y)


# Optimization parameters initialization
# initoptim creates optimization parameters for each numeric weight
# array in the model.  This should work for a model consisting of any
# combination of tuple/array/dict.
initoptim{T<:Number}(::KnetArray{T},otype)=eval(parse(otype))
initoptim{T<:Number}(::Array{T},otype)=eval(parse(otype))
initoptim(a::Associative,otype)=Dict(k=>initoptim(v,otype) for (k,v) in a) 
initoptim(a,otype)=map(x->initoptim(x,otype), a)

macro msg(_x)
    :(join(STDOUT,[Dates.format(now(),"HH:MM:SS"), $(esc(_x)),'\n'],' '); flush(STDOUT))
end

date(x)=(join(STDOUT,[Dates.format(now(),"HH:MM:SS"), x,'\n'],' '); flush(STDOUT))


type StopWatch;
    tstart; nstart; ncurr; nnext;
    StopWatch()=new(time(),0,0,1000)
end


function inc(s::StopWatch, n, step=1000)
    s.ncurr += n
    if s.ncurr >= s.nnext
        tcurr = time()
        dt = tcurr - s.tstart
        dn = s.ncurr - s.nstart
        s.tstart = tcurr
        s.nstart = s.ncurr
        s.nnext += step
        return dn/dt
    end
end


# (:epoch,0,:beam,1,:las,0.007938020871520395,0.013241609670749164) for randseed 4
# (:epoch,0,:beam,1,:las,0.007938020871520395,0.013241609670749164)

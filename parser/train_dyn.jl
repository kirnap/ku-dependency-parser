using AutoGrad: getval

# default feature set used in parser
FEATS=["s1c","s1v","s1p","s1A","s1a","s1B","s1b",
       "s1rL", # "s1rc","s1rv","s1rp",
       "s0c","s0v","s0p","s0A","s0B","s0a","s0b","s0d",
       "s0rL", # "s0rc","s0rv","s0rp",
       "n0lL", # "n0lc","n0lv","n0lp",
       "n0c","n0v","n0p","n0A","n0a",
       "n1c","n1v","n1p",
       ]
FTYPE=Float32                   # floating point type
GPUFEATURES=false               # Whether to compute gpu features or not
MAXWORD=32                      # truncate long words at this length. length("counterintelligence")=19
MAXSENT=64                      # skip longer sentences during training
MINSENT=2                       # skip shorter sentences during training


function main(o)
    println("opts=",[(k,v) for (k,v) in o]...)

    @msg o[:loadfile]
    d = load(o[:loadfile])

    if o[:seed] > 0
        srand(o[:seed])
    end

    if length(o[:dropout])==1
        o[:dropout]=ntuple(i->o[:dropout][1],2)
    end
    if length(o[:report])==1
        o[:report]=ntuple(i->o[:report][1], length(o[:datafiles]))
    end

    # assign transition based Parser type Abstraction from parser.jl
    o[:arctype] = (o[:arctype] == nothing ? get(d,"arctype",ArcHybridR1) : eval(parse(o[:arctype])))

    if haskey(d,"arctype") && o[:arctype] != d["arctype"]
        error("ArcType mismatch")
    end
    
    o[:feats] = (o[:feats] == nothing ? get(d,"feats",FEATS) : eval(parse(o[:feats])))
    if haskey(d,"feats") && o[:feats] != d["feats"]
        error("Feats mismatch")
    end
    # we specify three embedding dims for postag,deprel and count/distance for empty models.
    # the word and context embed sizes are given by chmodel dims.
    if isempty(o[:embed]) && !haskey(d,"parserv")
        o[:embed] = (128,32,16) # default embedding sizes
    elseif !isempty(o[:embed]) && haskey(d,"parserv") # check compat
        if !(length(d["postagv"][1])==o[:embed][1] &&
             length(d["deprelv"][1])==o[:embed][2] &&
             length(d["lcountv"][1])==length(d["rcountv"][1])==length(d["distancev"][1])==o[:embed][3])
            error("Embed mismatch")
        end
    end

    vocab = create_vocab(d)
    wmodel = makewmodel(d)
    corpora = []
    lfunc = (VERSION >= v"0.6.0" ? loadcorpus_v6 : loadcorpus)
    for f in o[:datafiles]; @msg f
        c = lfunc(f,vocab)
        push!(corpora,c)
    end
    
    cc = vcat(corpora...)
    info("Context and word embeddings are being generated")
    ppl = fillvecs!(wmodel, cc, vocab)
    unk = unkrate(cc)
    @msg "perplexity=$ppl unkrate=$(unk[1]/unk[2])"

    @msg :initmodel
    (pmodel,optim) = makepmodel(d,o,corpora[1][1])
    # add-hoc savefile function
    save1(file)=(@msg file; savefile(file, vocab, wmodel, pmodel, optim, o[:arctype], o[:feats]))


    # report function that is going to be used during training
    function report(epoch,beamsize=o[:beamsize])
        las_vals = zeros(length(corpora))
        for i=1:length(corpora)
            if o[:report][i] != 0 || (o[:bestfile] != nothing && i==length(corpora))
                las_vals[i] = beamtest(model=pmodel,
                                  corpus=corpora[i],
                                  vocab=vocab,
                                  arctype=o[:arctype],
                                  feats=o[:feats],
                                  beamsize=beamsize,
                                  batchsize=o[:batchsize])
            end
        end
        println((:epoch,epoch,:beam,beamsize,:las,las_vals...))
        return las_vals[end]
    end
    @msg :parsing
    bestlas = report(0,1)
    bestepoch = 0

    if o[:otrain]>0
        @msg :otrain
    end
    free_KnetArray()
    for epoch=1:o[:otrain]
        oracletrain(model=pmodel,
                    optim=optim,
                    corpus=corpora[1],
                    vocab=vocab,
                    arctype=o[:arctype],
                    feats=o[:feats],
                    batchsize=o[:batchsize],
                    pdrop=o[:dropout])
        currlas = report("oracle$epoch",1)
        if currlas > bestlas 
            bestlas = currlas
            bestepoch = epoch
            if o[:bestfile] != nothing
                save1(o[:bestfile])
            end
        end
        #if 5 < bestepoch < epoch - 5
        #    break
        #end
    end

    # savemodel
    if o[:savefile] != nothing
        save1(o[:savefile])
    end

    # output parse for corpora[1]
    if o[:output] != nothing    # TODO: parse all data files?
        @msg o[:output]
        if corpora[1][1].parse == nothing
            @msg :parsing
            beamtest(model=pmodel,corpus=corpora[1],vocab=vocab,arctype=o[:arctype],feats=o[:feats],beamsize=o[:beamsize],batchsize=o[:batchsize])
        end
        writeconllu(corpora[1], o[:datafiles][1], o[:output])
    end
    @msg :done

end


function beamtest(;model=_model, corpus=_corpus, vocab=_vocab, arctype=ArcHybridR1, feats=FEATS, beamsize=4, batchsize=128) # large batchsize does not slow down beamtest
    for s in corpus
        s.parse = nothing
    end
    
    sentbatches = minibatch(corpus,batchsize)
    for sentences in sentbatches
        beamloss(model, sentences, vocab, arctype, feats, beamsize; earlystop=false)
    end
    las(corpus)
end


function beamloss(pmodel, sentences, vocab, arctype, feats, beamsize; earlystop=true, steps=nothing, pdrop=(0,0))
    parsers = parsers0 = map(arctype, sentences)
    beamends = beamends0 = collect(1:length(parsers)) # marks end column of each beam, initially one parser per beam
    pcosts  = pcosts0  = zeros(Int, length(sentences))
    pscores = pscores0 = zeros(FTYPE, length(sentences))
    totalloss = stepcount = 0
    featmodel,mlpmodel = splitmodel(pmodel)
    
    while length(beamends) > 0
        # features (vcat) are faster on cpu, mlp is faster on gpu
        fmatrix = features(parsers, feats, featmodel) # nfeat x nparser
        if GPUFEATURES && gpu()>=0 #GPU
            @assert isa(getval(fmatrix),KnetArray{FTYPE,2})
        else #CPU
            @assert isa(getval(fmatrix),Array{FTYPE,2})
            if gpu()>=0; fmatrix = KnetArray(fmatrix); end
        end
        cscores = Array(mlp(mlpmodel, fmatrix; pdrop=pdrop)) # candidate scores: nmove x nparser
        cscores = cscores .+ pscores' # candidate cumulative scores
        parsers,pscores,pcosts,beamends,loss = nextbeam(parsers, cscores, pcosts, beamends, beamsize; earlystop=earlystop)
        totalloss += loss
        stepcount += 1
    end
    # emptyvecs!(sentences)       # if we don't empty, gc cannot clear these vectors; empty if finetuning wvecs
    if steps != nothing
        steps[1] += stepcount
        steps[2] += length(sentences[1])*2-2
    end
    return totalloss / length(sentences)
end


function nextbeam(parsers, mscores, pcosts, beamends, beamsize; earlystop=true)
    n = beamsize * length(beamends) + 1
    newparsers, newscores, newcosts, newbeamends, loss = Array(Any,n),Array(Int,n),Array(Int,n),Int[],0.0
    nmoves,nparsers = size(mscores)                     # mscores[m,p] is the score of move m for parser p
    mcosts = Array(Any, nparsers)                       # mcosts[p][m] will be the cost vector for parser[p],move[m] if needed
    n = p0 = 0
    for p1 in beamends                                  # parsers[p0+1:p1], mscores[:,p0+1:p1] is the current beam belonging to a common sentence
        s0,s1 = 1 + nmoves*p0, nmoves*p1                # mscores[s0:s1] are the linear indices for mscores[:,p0:p1]
        nsave = n                                       # newparsers,newscores,newcosts[nsave+1:n] will the new beam for this sentence
        ngold = 0                                       # ngold, if nonzero, will be the index of the gold path in beam
        sorted = sortperm(getval(mscores)[s0:s1], rev=true)	
        for isorted in sorted
            linidx = isorted + s0 - 1
            (move,parent) = ind2sub(size(mscores), linidx) # find cartesian index of the next best score
            parser = parsers[parent]

            # skip illegal move
            if !moveok(parser,move)
                continue
            end  
            if earlystop && !isassigned(mcosts, parent) # not every parent may have children, avoid unnecessary movecosts
                mcosts[parent] = movecosts(parser, parser.sentence.head, parser.sentence.deprel)
            end
            n += 1
            newparsers[n] = copy(parser); move!(newparsers[n], move)
            newscores[n] = linidx
            #TEST: no newcosts during test
            if earlystop
                newcosts[n] = pcosts[parent] + mcosts[parent][move]
                if newcosts[n] == 0
                    if ngold==0
                        ngold=n
                    else
                        # @msg("multiple gold moves for $(parser.sentence)")
                    end
                end
            end
            if n-nsave == beamsize; break; end
        end
        if n == nsave
            if parsers[p1].nword == 1                   # single word sentences give us no moves
                s = parsers[p1].sentence
                s.parse = parsers[p1]
            else
                error("No legal moves?")                # otherwise this is suspicious
            end
        #TEST: there will be no ngold during test
        elseif earlystop && ngold == 0                  # gold path fell out of beam, early stop
            gindex = goldindex(parsers,pcosts,mcosts,(p0+1):p1)
            if gindex != 0
                newscores[n+1] = gindex
                loss = loss - mscores[gindex] + logsumexp2(mscores, newscores[nsave+1:n+1])
            end
            n = nsave
        elseif endofparse(newparsers[n])                # all parsers in beam have finished, gold among them if earlystop
            #TEST: there will be no ngold during test, cannot return beamloss, just return the highest scoring parse, no need for normalization
            if earlystop
                gindex = newscores[ngold]
                loss = loss - mscores[gindex] + logsumexp2(mscores, newscores[nsave+1:n])
            end
            s = newparsers[n].sentence
            # @assert s == newparsers[nsave+1].sentence
            # @assert mscores[newscores[nsave+1]] >= mscores[newscores[n]] "s[$(1+nsave)]=$(mscores[newscores[nsave+1]]) s[$n]=$(mscores[newscores[n]])"
            s.parse = newparsers[nsave+1]
            n = nsave                                   # do not add finished parsers to new beam
        else                                            # all good keep going
            push!(newbeamends, n)
        end
        p0 = p1
    end
    return newparsers[1:n], mscores[newscores[1:n]], newcosts[1:n], newbeamends, loss
end


function logsumexp2(a, r)
    amax = getval(a)[r[1]]
    log(sum(exp(a[r] - amax))) + amax
end


function goldindex(parsers, pcosts, mcosts, beamrange)
    parent = findfirst(view(pcosts, beamrange), 0)
    if parent == 0
        error("cannot find gold parent in $beamrange")
    end
    parent += first(beamrange) - 1
    if !isassigned(mcosts, parent)
        p = parsers[parent]
        mcosts[parent] = movecosts(p, p.sentence.head, p.sentence.deprel)
    end
    move = findfirst(mcosts[parent],0)
    if move == 0
        return 0
    else
        msize = (parsers[1].nmove,length(parsers))
        return sub2ind(msize,move,parent)
    end
end


endofparse(p)=(p.sptr == 1 && p.wptr > p.nword)


function minibatch(corpus, batchsize; maxlen=typemax(Int), minlen=1, shuf=false)
    data = Any[]
    sorted = sort(corpus, by=length)
    i1 = findfirst(x->(length(x) >= minlen), sorted)
    if i1==0; error("No sentences >= $minlen"); end
    i2 = findlast(x->(length(x) <= maxlen), sorted)
    if i2==0; error("No sentences <= $maxlen"); end
    for i in i1:batchsize:i2
        j = min(i2, i+batchsize-1)
        push!(data, sorted[i:j])
    end
    if shuf
        data=shuffle(data)
    end
    return data
end


function las(corpus)
    nword = ncorr = 0
    for s in corpus
        p = s.parse
        nword += length(s)
        ncorr += sum((s.head .== p.head) & (s.deprel .== p.deprel))
    end
    ncorr / nword
end


function splitmodel(pmodel)
    # optimization: do all getindex operations outside, otherwise each getindex creates a new node
    # TODO: fix this in general
    mlpmodel = Any[]
    mlptemp = parserv(pmodel)
    for i=1:length(mlptemp); push!(mlpmodel, mlptemp[i]); end
    featmodel = Array(Any,5)
    for k in 1:5 # (:postag,:deprel,:lcount,:rcount,:distance)
        featmodel[k] = Any[]
        pmodel_k = pmodel[k]
        for i in 1:length(pmodel_k)
            push!(featmodel[k], pmodel_k[i])
        end
    end
    return (featmodel,mlpmodel)
end


function oracletrain(;model=_model, optim=_optim, corpus=_corpus, vocab=_vocab, arctype=ArcHybridR1, feats=FEATS, batchsize=16, maxiter=typemax(Int), pdrop=(0,0))
    sentbatches = minibatch(corpus,batchsize; maxlen=MAXSENT, minlen=MINSENT, shuf=true)
    nsent = sum(map(length,sentbatches)); nsent0 = length(corpus)
    nword = sum(map(length,vcat(sentbatches...))); nword0 = sum(map(length,corpus))
    @msg("nsent=$nsent/$nsent0 nword=$nword/$nword0")
    nwords = StopWatch()
    losses = Any[0,0,0]
    niter = 0
    @time for sentences in sentbatches
        grads = oraclegrad(model, sentences, vocab, arctype, feats; losses=losses, pdrop=pdrop)
        update!(model, grads, optim)
        nw = sum(map(length,sentences))
        if (speed = inc(nwords, nw)) != nothing
            date("$(nwords.ncurr) words $(round(Int,speed)) wps $(losses[3]) avgloss")
            free_KnetArray()
        end
        if (niter+=1) >= maxiter; break; end
    end
    println()
end


function oracleloss(pmodel, sentences, vocab, arctype, feats; losses=nothing, pdrop=(0,0))
    parsers = map(arctype, sentences)
    mcosts = Array(Cost, parsers[1].nmove)
    parserdone = falses(length(parsers))
    totalloss = 0
    featmodel,mlpmodel = splitmodel(pmodel)

    nsteps = 0
    while !all(parserdone)
        fmatrix = features(parsers, feats, featmodel)
        #fmatrix = rand(FTYPE, 4664, length(parsers))
        if GPUFEATURES && gpu()>=0 #GPU
            @assert isa(getval(fmatrix),KnetArray{FTYPE,2})
        else #CPU
            @assert isa(getval(fmatrix),Array{FTYPE,2})
            if gpu()>=0
                fmatrix = KnetArray(fmatrix)
            end
        end
        scores = mlp(mlpmodel, fmatrix; pdrop=pdrop)
        logprob = logp(scores, 1)
        for (i,p) in enumerate(parsers)
            if parserdone[i]
                continue
            end
            movecosts(p, p.sentence.head, p.sentence.deprel, mcosts)
            goldmove = indmin(mcosts)
            if mcosts[goldmove] == typemax(Cost)
                parserdone[i] = true
                p.sentence.parse = p
            else
                totalloss -= logprob[goldmove,i]
                ##########
                # This part is done to make dynamic-oracle possible
                mmax = 0; smax = -Inf;
                logprob1 = Array((getval(logprob)));
                for m =1:size(logprob1, 1)
                    if mcosts[m] < typemax(Cost) && logprob1[m, i] > smax
                        smax = logprob1[m, i]; mmax = m;
                    end
                end
                move!(p, deepcopy(mmax))
                #########
                if losses != nothing
                    loss1 = -getval(logprob)[goldmove,i]
                    losses[1] += loss1
                    losses[2] += 1
                    if losses[2] < 1000
                        losses[3] = losses[1]/losses[2]
                    else
                        losses[3] = 0.999 * losses[3] + 0.001 * loss1
                    end
                end
            end
        end
        nsteps +=1;
    end
    return totalloss / length(sentences)
end

oraclegrad = grad(oracleloss)

using AutoGrad
import Base: sortperm, ind2sub
@zerograd sortperm(a;o...)
@zerograd ind2sub(a,i...)


# static oracle approx max ~.81 in Eng-Par-TUT
# Here is the general command to run the parser:
# julia main.jl --load data/english_chmodel.jld --datafiles /mnt/ai/data/nlp/conll17/ud-treebanks-v2.0/UD_English-LinES/en_lines-ud-train.conllu /mnt/ai/data/nlp/conll17/ud-treebanks
# For Hungarian:
# julia main.jl --load /mnt/ai/data/nlp/conll17/competition/chmodel_converted/hungarian_chmodel.jld --datafiles /mnt/ai/data/nlp/conll17/ud-treebanks-v2.0/UD_Hungarian/hu-ud-train.conllu /mnt/ai/data/nlp/conll17/ud-treebanks-v2.0/UD_Hungarian/hu-ud-dev.conllu

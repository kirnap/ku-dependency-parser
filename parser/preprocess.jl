# Preprocessing the parse data
# Designed to read from conllu-format file, 

# regular expression handles which column to take
function loadcorpus(file,v::Vocab)
    corpus = Any[]
    s = Sentence(v)
    for line in eachline(file)
        if line == "\n"
            push!(corpus, s)
            s = Sentence(v)
        elseif (m = match(r"^\d+\t(.+?)\t.+?\t(.+?)\t.+?\t.+?\t(.+?)\t(.+?)(:.+)?\t", line)) != nothing
            #                id   word   lem  upos   xpos feat head   deprel
            word = m.captures[1]
            push!(s.word, word)
            
            postag = get(v.postags, m.captures[2], 0)
            if postag==0
                Base.warn_once("Unknown postags")
            end
            push!(s.postag, postag)
            
            head = tryparse(Position, m.captures[3])
            head = isnull(head) ? -1 : head.value
            if head==-1
                Base.warn_once("Unknown heads")
            end
            push!(s.head, head)

            deprel = get(v.deprels, m.captures[4], 0)
            if deprel==0
                Base.warn_once("Unknown deprels")
            end
            push!(s.deprel, deprel)
        end
    end
    return corpus
end

# To create vocabulary from pre-trained lstm model
function create_vocab(d)
    Vocab(d["char_vocab"],
          Dict{String, Int}(),
          d["word_vocab"],
          d["sosword"],
          d["eosword"],
          d["unkword"],
          d["sowchar"],
          d["eowchar"],
          d["unkchar"],
          get(d, "postags", UPOSTAG),
          get(d, "deprels", UDEPREL)
          )
end


function maptoint(sentences, v::Vocab)
    MAXWORD = 32
    wdict = empty!(v.idict) # it is already empty ?
    cdict = v.cdict
    unkcid = cdict[v.unkchar]
    words = Vector{Int}[]
    sents = Vector{Int}[]

    maxwordlen = 0; maxsentlen = 0;
    for w in (v.sosword, v.eosword)
        wid = get!(wdict, w, 1+length(wdict))
        word = Array(Int, length(w))
        wordi = 0 # to check 2 byte characters
        for c in w
            word[wordi+=1] = get(cdict, c, unkcid)
        end
        (wordi != length(w)) && error("Missing in single word process")
        (wordi > maxwordlen) && (maxwordlen = wordi)
        push!(words, word)
    end

    for s in sentences
        sent = Array(Int, length(s.word))
        senti = 0
        for w in s.word
            ndict = length(wdict)
            wid = get!(wdict, w, 1+ndict)
            sent[senti+=1] = wid
            if wid == 1+ndict
                word = Array(Int, length(w))
                wordi = 0
                for c in w
                    word[wordi+=1] = get(cdict, c, unkcid)
                end
                (wordi != length(w)) && error("Missing in single word process")
                if wordi > MAXWORD; wordi=MAXWORD; word = word[1:wordi]; end;
                (wordi > maxwordlen) && (maxwordlen = wordi) 
                push!(words, word)
            end
        end
        @assert(senti == length(s.word))
        (senti > maxsentlen) && (maxsentlen = senti)
        push!(sents, sent)
    end
    @assert(length(wdict) == length(words))
    return words, sents, maxwordlen, maxsentlen
end


# To make ready for character based lstm, data[i] corresponds to ith time step input, to charlstm
function tokenbatch(words, maxlen, sos, eos, pad=eos)
    B = length(words) # batchsize
    T = maxlen + 2
    data = [ Array(Int, B) for t in 1:T ]
    mask = [ Array(Float32, B) for t in 1:T ]
    @inbounds for t in 1:T
        for b in 1:B
            N = length(words[b]) # wordlen
            n = t - T + N + 1 # cursor 
            if n < 0
                mask[t][b] = 0
                data[t][b] = pad
            else
                mask[t][b] = 1
                if n == 0
                    data[t][b] = sos
                elseif n <= N
                    data[t][b] = words[b][n]
                elseif n == N+1
                    data[t][b] = eos
                else
                    error()
                end
            end
        end
    end
    return data, mask
end


# Gold-batch for final layer of the bilstm implementation
function goldbatch(sentences, maxlen, wdict, unkwid, pad=unkwid)
    B = length(sentences)
    T = maxlen
    data = [ Array(Int, B) for t in 1:T ]
    mask = [ Array(Float32, B) for t in 1:T ]
    for t in 1:T
        for b in 1:B
            N = length(sentences[b])
            n = t - T + N
            if n <= 0
                mask[t][b] = 0
                data[t][b] = pad
            else
                mask[t][b] = 1
                data[t][b] = get(wdict, sentences[b].word[n], unkwid)
            end
        end
    end
    return data, mask
end

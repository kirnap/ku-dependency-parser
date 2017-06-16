# Infinite stream processing version 2, don't give <unk> as an input to model, which can handle infinite amount of tokenized-raw text data
const SOS = "<s>"
const EOS = "</s>"
const UNK = "<unk>"

# character constants
const PAD = Char(0x0011)
const SOW = Char(0x0012)
const EOW = Char(0x0013)


function create_vocab(vocabfile::AbstractString)
    ercount = 0
    result = Dict{AbstractString, Int}(SOS=>1, EOS=>2, UNK=>3)
    open(vocabfile) do f
        for line in eachline(f)
            words= split(line)
            try
                @assert(length(words) == 1, "The vocabulary file seems broken")
            catch e
                ercount += 1
                if (length(words) == 0)
                    if (ercount == 5)
                        info("There is empty line in vocabulary file")
                        ercount = 0
                    end
                    continue
                else
                    warn("Something unexpected happen in vocabfile")
                end
            end
            word = words[1]
            if length(word) > 65
                isinteractive() && warn("Too long words in file $vocabfile")
                word = word[1:65]
            end
            get!(result, word, 1+length(result))
        end
    end
    return result
end


"""
Reads from infinite stream modifies the s paramater in the light of sequence lengths,
it modifies the s argument and such that it contains what i called information tuple:
(outid, inid) -> outid is the the output vocabulary to use in softmax, inid is the input vocabulary to use in char level lstm
"""
function readstream!(f::IO,
                     s::Dict{Int64, Array{Any, 1}},
                     vocab::Dict{AbstractString, Int64},
                     realvocab::Dict{AbstractString, Int64};
                     maxlines=100, llimit=3, ulimit=300)
    
    k = 0
    while (k != maxlines)
        eof(f) && return false
        words = split(readline(f))
        wlen = length(words)
        (wlen < llimit) || (wlen > ulimit-2) && continue
        seq = Array(Tuple{Int32, Int32}, wlen+2)
        seq[1] = (vocab[SOS], vocab[SOS])
        for i=1:wlen
            word = words[i]
            index = get(vocab, word, vocab[UNK])
            realindex = get(realvocab, word, realvocab[UNK]) # no unking activity desired
            seq[i+1] = (index, realindex)
        end
        seq[end] = (vocab[EOS], vocab[EOS])
        skey = wlen+2
        (!haskey(s, skey)) && (s[skey] = Any[])
        push!(s[skey], seq)
        k += 1
    end
end


function mbatch(sequences::Array{Any, 1}, batchsize::Int)
    seqlen = length(sequences[1])
    data = Array(Any, seqlen)
    for cursor=1:seqlen
        d = Array(Tuple{Int32, Int32}, batchsize)
        for i=1:batchsize
            d[i] = sequences[i][cursor]
        end
        data[cursor] = d
    end
    return data
end


function nextbatch(f::IO,
                   sdict::Dict{Int64, Array{Any, 1}},
                   vocab::Dict{AbstractString, Int64},
                   realvocab::Dict{AbstractString, Int64},
                   batchsize;
                   o...)
    
    slens = collect(filter(x->length(sdict[x])>=batchsize, keys(sdict)))
    if length(slens) < 10
        readstream!(f, sdict, vocab, realvocab; o...)
        slens = collect(filter(x->length(sdict[x])>=batchsize, keys(sdict)))
        if length(slens) == 0
            return nothing
        end
    end
    slen = rand(slens)
    sequence = sdict[slen][1:batchsize]
    deleteat!(sdict[slen], 1:batchsize)
    return mbatch(sequence, batchsize)
end


"""
bytpe: out -> give "unk"ed version of the word, give normal version otherwise.
"""
function ibuild_sentence(i2w::Array{AbstractString, 1}, sequence::Array{Any, 1}, kth::Int; btype=:out, verbose=true)
    sentence = Any[]
    for i=1:length(sequence)
        z = sequence[i][kth]
        index = ((btype == :out) ? z[1] : z[2])
        push!(sentence, i2w[index])
    end
    if verbose
        for item in sentence; print("$item ");end;
    else
        return sentence
    end
end


function create_chvocab(word_vocab::Dict{AbstractString, Int64})
    res = Dict{Char, Int}(PAD=>1, SOW=>2, EOW=>3)
    for k in keys(word_vocab)
        for ch in k
            if ch == PAD || ch == SOW || ch == EOW
                warn("$ch is used in vocabulary")
            end
            (ch == ' ') && continue
            get!(res, ch, 1+length(res))
        end
    end
    return res
end


function create_chvocab(f::AbstractString, ulimit::Int=20000)
    res = Dict{Char, Int}(PAD=>1, SOW=>2, EOW=>3)
    counter = 0
    open(f) do file
        for line in eachline(file)
            counter += 1; (counter > ulimit) && break
            word = line
            # word = split(line)[2] no need because not to use count file
            for c in word
                if c == PAD || c == SOW || c == EOW
                    warn("$c is used in vocabulary")
                end
                (c == ' ') && continue
                get!(res, c, 1+length(res))
            end
        end
    end
    return res
end


longest_word{T}(word_vocab::Dict{T, Int}) = findmax(map(length, keys(word_vocab)))[1]


"""
wids : batch of words in (outid, inid) -> outid is the softmax layer id, inid is the char level id,
charlup only returns the input to char level lstms
"""
function charlup(wids::Array{Tuple{Int32,Int32},1}, i2w_all::Array{AbstractString, 1}, ch::Dict{Char, Int})
    words = map(x->i2w_all[x[2]], wids)
    critic = findmax(map(length, words))[1]

    batchsize = length(words)
    data = Array(Any, critic+2)
    data[1] = fill!(zeros(Int32, batchsize), ch[SOW])
    masks = Array(Any, critic+2)
    masks[1] = ones(Int32, batchsize, 1)
    for cursor=1:critic+1 # to pad EOW to the end of the word
        d = Array(Int32, batchsize)
        mask = ones(Float32, batchsize, 1)
        @inbounds for i=1:batchsize
            word = words[i]
            if length(word) < critic
                if length(word) >= cursor
                    try
                        d[i] = get(ch, word[cursor], ch[PAD]) #ch[word[cursor]], there may be unk characters
                    catch
                        d[i] = ch[PAD]
                    end
                elseif length(word)+1 == cursor
                    d[i] = ch[EOW]
                else
                    d[i] = ch[PAD]
                    mask[i] = 0
                end
            else
                if cursor>critic
                    d[i] = ch[EOW]
                else
                    try
                        d[i] = get(ch, word[cursor], ch[PAD]) # unking operation
                    catch
                        d[i] = ch[PAD]
                    end
                end
            end
        end
        data[cursor + 1] = d
        masks[cursor + 1] = mask
    end
    return data, masks
end


function charlup2(wids::Array{Tuple{Int32,Int32},1}, i2w_all::Array{AbstractString, 1}, ch::Dict{Char, Int})
    words = map(x->i2w_all[x[2]], wids)
    pad_critic = findmax(map(length, words))[1] + 2 # + 2 for start and and end characters

    batchsize = length(words)
    data = Array(Any, pad_critic)
    masks = Array(Any, pad_critic)

    for cursor=1:pad_critic-1
        d = Array(Int32, batchsize)
        mask = ones(Float32, batchsize, 1)
        @inbounds for i=1:batchsize
            word = words[i]
            tostart = pad_critic - (length(word)+2) + 1
            if cursor == tostart
                d[i] = ch[SOW]
            elseif cursor > tostart
                try
                   ch_place = cursor - tostart # it shows which chracter of word in
                   d[i] = ch[word[ch_place]]
                catch
                    d[i] = ch[PAD]
                    mask[i] = 0.0
                end
            else
                d[i] = ch[PAD]
                mask[i] = 0.0
            end
        end
        data[cursor] = d
        masks[cursor] = mask
    end
    data[end] = fill!(zeros(Int32, batchsize), ch[EOW])
    masks[end] = ones(Float32, batchsize, 1)
    return data, masks
end


function ibuild_wordflup(i2c::Array{Char, 1}, data::Array{Any, 1}, kth::Int; verbose=false)
    word = Any[]
    for i=1:length(data)
        z = data[i][kth]
        push!(word, i2c[z])
    end
    if verbose
        for item in word; print("$item");end;println()
    else
        return word
    end
end


"""
Find the maximum length word in vocabulary, padds each word based on that length
"""
function create_conv_vocab(word_vocab_all::Dict{AbstractString, Int})
    maxlen = longest_word(word_vocab_all)
    result = Dict{AbstractString, Int}()
    for word in keys(word_vocab_all)
        word2 = string(SOW) * word * string(EOW)
        new_word = string(word2, string(PAD)^(maxlen-length(word)))
        result[new_word] = word_vocab_all[word]
    end
    return result
end


function cbatch4conv(wids::Array{Tuple{Int32, Int32}, 1}, i2w_all_conv::Array{AbstractString, 1}, ch::Dict{Char, Int}, atype::DataType)
    batchsize = length(wids)
    words = mapreduce(x->i2w_all_conv[x[2]], string, "", wids) # concatanete each word
    maxlen = length(words)

    data = zeros(Int32, maxlen)
    mask = zeros(Int32, maxlen, 1)
    for (i,c) in enumerate(words)
        if c == PAD
            data[i] = ch[PAD]
        else
            data[i] = get(ch, c, ch[PAD])
            mask[i, 1] = 1
        end
    end
    mcon = convert(atype, mask)
    return (data, mcon)
end


function calculate_coverage(vocabfile::AbstractString, vocabsize::Int)
    total = covered = 0.0
    open(vocabfile) do file
        counter = 0
        for line in eachline(file)
            counter += 1
            occurence = parse(Float64, split(line)[1])
            (counter <= vocabsize) && (covered += occurence)
            total += occurence
        end
    end
    rate = (covered / total) * 100
    println("Vocabsize $vocabsize | Coverage rate $rate")
    return rate
end


word_arrange(pad_critic::Int, x::AbstractString) = string(string(PAD)^(pad_critic - length(x) -2), string(SOW), x, string(EOW))


function charlup3(wids::Array{Tuple{Int32,Int32},1}, i2w_all::Array{AbstractString, 1}, ch::Dict{Char, Int})
    words = map(x->i2w_all[x[2]], wids)
    pad_critic = findmax(map(length, words))[1] + 2 # + 2 for start and and end characters
    batchsize = length(words)
   
    data = [ zeros(Int32, batchsize) for i=1:pad_critic ]
    masks = [ ones(Float32, batchsize, 1) for i=1:pad_critic ]
    words = map(x->word_arrange(pad_critic, x), words)

    @inbounds for i=1:batchsize
        word = words[i]
        for (cursor, char) in enumerate(word)
            (char == PAD) && (masks[cursor][i] = 0.0)
            data[cursor][i] = get(ch, char, ch[PAD])
        end
    end
    return data, masks
end

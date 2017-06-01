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

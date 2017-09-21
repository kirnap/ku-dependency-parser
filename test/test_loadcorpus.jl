# Tests the load corpus function
include(Pkg.dir("KUparser/conll17/rnnparser.jl"))
include("set_testenv.jl")
include("../src/types.jl")
include("../src/preprocess.jl")
using JLD

# add-hoc solution for vocab type of the parser
function eqtypes{T}(a::T, b::T)
    for item in fieldnames(a)
        if item == :vocab
            x = eqtypes(getfield(a, item), getfield(b, item))
            (x != true) && error("field $item are not equal")
        else
            @assert(getfield(a, item) == getfield(b, item), "$item test failed in $(typeof(a))")
        end
    end
    return true
end


function test_loadcorpus()
    d = load(chmodel)
    vocab = create_vocab(d)
    corpus = loadcorpus(real_tdata, vocab)

    orig_corpora, orig_vocab = main("--load $chmodel --datafiles $real_tdata $real_ddata --otrain 0")
    orig_corpus = orig_corpora[1]

    for item in fieldnames(orig_vocab)
        @assert(getfield(orig_vocab, item) == getfield(vocab, item), "$item are not equal in those vocabularies")
    end
    info("Vocabulary assertion passed")

    for i=1:length(corpus)
        eqtypes(corpus[i], orig_corpus[i])
    end
    info("Datafiles read as the with original rnnparser")
    return vocab, corpus, orig_corpus, orig_vocab
end
!isinteractive() && test_loadcorpus()


# TODO locate en_chmodel.jld
# test with rnnparser.jl implementation and your implementation

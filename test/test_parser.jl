# To test implementation of parser, and understand the working structure
using JLD, Knet
include("../src/types.jl")
include("../src/preprocess.jl")
include("../src/parser.jl")
include("../src/helper.jl")
include("../src/modelutils.jl")
include("set_testenv.jl")


function test_parser()
    d = load(chmodel); vocab = create_vocab(d)
    corpora = []
    for f in [real_tdata2, real_ddata2]
        c = loadcorpus(f, vocab)
        push!(corpora, c)
    end
    sentence = corpora[1][4]
    p1 = ArcHybridR1(sentence)
    return sentence, p1
end


# LEFT-ARC : (σ|wi,wj|β,A) ⇒ (σ,wj|β,A∪{(wj,r,wi)})

# To perform Left-Arc operation you need two preconditions:
# - Stack and buffer are non-empty and wi ≠ ROOT

# RIGHT-ARC : (σ|wi , wj|β, A) ⇒ (σ, wi|β, A∪{(wi , r, wj)})
# To perform right-arc operation the only precondition is that the stack and buffer are non-empty.

# we have total of p.moves that are represented as :
# shiftmove has a value of 1
# leftmoves are the even numbers from 2 to p.nmove-2
# rightmoves are the odd numbers from 3 to p.nmove-1

# Here are some other notes on that:
# nword stands for number of words in that sentence, ndeps stands for number of dependency labels in that sentence

# init! function modifies the number of possible moves in that parser, and then applies the shift operation to the parser

# In shift move:
# - increase p.sptr by 1,
# - make p.stack[p.sptr] = p.wptr -> that puts ith word on buffer to stack.
# - increase the p.wptr by 1


# SHIFT MOVE implementation
# shift(p::Parser)=(p.stack[p.sptr+=1]=p.wptr; p.wptr+=1)


# Parser fields:
# p.wptr -> indicates the word pointer in Buffer, therefore if the shift move happens, then -> p.wptr += 1
# p.sptr -> indicates the word pointer in stack,



# movecosts take parser, parser.sentence.head, parser.sentence.deprel as an input

# Here are the steps to do:
# You need to find a correct move set of a given sentence
# You need to make 3 experiments:
# - Use random moves  
# - Use correct moves
# - Use predicted moves

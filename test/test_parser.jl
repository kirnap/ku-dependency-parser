# To test implementation of parser, and understand the working structure
using JLD, Knet
include("../parser/types.jl")
include("../parser/preprocess.jl")
include("../parser/parser.jl")
include("../parser/helper.jl")
include("../parser/modelutils.jl")
include("set_testenv.jl")


function test_parser(;mode=:random)
    p1, s1 = get_parser_sentence()

    

end


function compare_moves(p1, s1)
    p_r = deepcopy(p1);
    old_acs = nothing
    for i in 1:500
        p_r = deepcopy(p1)
        
        acs = done_randmoves(p_r, s1)
        (acs == old_acs) && prinlnt("Previous and currrent actions are the same")
        old_acs = acs
        if p_r.head == s1.head
            println("Correct sequence is catch")
        end
    end
end


# extract the gold moves from given parser
function done_goldmoves(porig, s1)
    actions = []
    p1 = deepcopy(porig)
    totmoves = 0
    parserdone = false
    while !parserdone
        mcosts = movecosts(p1, p1.sentence.head, p1.sentence.deprel)
        goldmove = indmin(mcosts)
        if mcosts[goldmove] == typemax(Cost)
            parserdone = true
        else
            push!(actions, goldmove)
            move!(p1, goldmove)
            totmoves += 1;
        end
    end
    println("GoldMoves: Total moves $totmoves total words $(length(s1.word))")
    return p1, actions
end


# Done random moves
function done_randmoves(pin, s1)
    p1 = deepcopy(pin)
    ac_seq = []
    totmoves = 0
    mcosts = movecosts(p1, p1.sentence.head, p1.sentence.deprel)
    n = length(mcosts)
    while anyvalidmoves(p1)
        mcosts = movecosts(p1, p1.sentence.head, p1.sentence.deprel)
        validmoves = find(mcosts .< typemax(Cost))
        m = validmoves[rand(1:length(validmoves))]
        push!(ac_seq, m)
        totmoves += 1;
        move!(p1, m)
    end
    println("RandomMoves: Total moves $totmoves total words $(length(s1.word))")
    return ac_seq
end


function get_parser_sentence()
    d = load(chmodel); vocab = create_vocab(d)
    corpora = []
    for f in [real_tdata3, real_ddata3]
        c = loadcorpus(f, vocab)
        push!(corpora, c)
    end
    sentence = corpora[1][4]
    p1 = ArcHybridR1(sentence)
    return p1, sentence
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

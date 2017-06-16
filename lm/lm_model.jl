include("lm_util.jl")
# language model implementation
function initmodel(atype, hiddens, charhidden, chembedding, wordvocab, charvocab, init=xavier)
    model = Dict{Symbol, Any}()
    wordembedding = charhidden[1]

    model[:forw] = initweights(atype, hiddens, wordembedding, init)
    model[:back] = initweights(atype, hiddens, wordembedding, init)
    model[:char] = initweights(atype, charhidden, chembedding, init)
    model[:cembed] = atype(init(charvocab, chembedding))
    model[:soft] = [ atype(init(2hiddens[end], wordvocab)), atype(zeros(1, wordvocab)) ]
    return model
end


function chlstm(weight, bias, hidden, cell, input; mask=nothing)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)

    # masking operation
    cell    = cell .* mask
    hidden  = hidden .* mask
    return (hidden,cell)
end


"""
Multilayer lstm forward function in single time step
"""
function chforw(weight, states, input; mask=nothing)
    x = input
    for i=1:2:length(states)
        (states[i], states[i+1]) = chlstm(weight[i], weight[i+1], states[i], states[i+1], x; mask=mask)
        x = states[i]
    end
    return x
end


"""
mchar, mcembed, states: model's character lstm, model's character embeddings, char-lstm states respectively.
wids, i2w_all, ch: batch of words, index to word array and character vocabulary respectively.
"""
function charembed(mchar, mcembed, states, wids, i2w_all, ch, atype)
    schar = copy(states)

    (data, masks) = charlup3(wids, i2w_all, ch)
    h = similar(schar[1])
    for (c, m) in zip(data, masks)
        embed = mcembed[c, :]
        mbon = convert(atype, m)
        h = chforw(mchar, schar, embed; mask=mbon)
    end
    return h
end



function charbilstm(model, chstates, states, sequence, i2w_all, chvocab, lval=[])
    total = 0.0
    count = 0
    atype = typeof(states[1])

    # extract the embeddings on character reading
    embeddings = Array(Any, length(sequence))
    for i=1:length(sequence)
        embeddings[i] = charembed(model[:char], model[:cembed], chstates, sequence[i], i2w_all, chvocab, atype)
    end

    # forward lstm
    fhiddens = Array(Any, length(sequence)-2)
    sf = copy(states)
    for i=1:length(sequence)-2
        x = embeddings[i]
        h = forward(model[:forw], sf, x)
        fhiddens[i] = copy(h)
    end

    # backward lstm
    bhiddens = Array(Any, length(sequence)-2)
    sb = copy(states)
    for i=length(sequence):-1:3
        x = embeddings[i]
        h = forward(model[:back], sb, x)
        bhiddens[i-2] = copy(h)
    end
 
    # concatenate layer
    for i=1:length(fhiddens)
        ypred = hcat(fhiddens[i], bhiddens[i]) * model[:soft][1] .+ model[:soft][2]
        ygold = map(x->x[1], sequence[i+1])
        total += logprob(ygold, ypred)
        count += length(ygold)
    end
    val = - total / count
    push!(lval, AutoGrad.getval(val))
    return val
end


gradcharbilstm = grad(charbilstm)


function train(model, chstates, states, sequence, i2w_all, chvocab, opts)
    lval = []
    gloss = gradcharbilstm(model, chstates, states, sequence, i2w_all, chvocab, lval)
    update!(model, gloss, opts)
    return lval[1]
end


function devperp(model, chstates, states, i2w_all, chvocab)
    devloss = []
    for d in dev
        charbilstm(model, chstates, states, d, i2w_all, chvocab, devloss)
    end
    return exp(mean(devloss))
end

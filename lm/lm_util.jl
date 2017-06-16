# common lm-model file functions

function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end


# multilayer lstm forward, returns the final hidden
function forward(weight, states, input)
    x = input
    for i=1:2:length(states)
        (states[i], states[i+1]) = lstm(weight[i], weight[i+1], states[i], states[i+1], x)
        x = states[i]
    end
    return x
end


# lstm weights initialization
# w[2k-1], w[2k] : weight and bias for kth layer respectively
function initweights(atype, hiddens, embedding, init=xavier)
    weights = Array(Any, 2length(hiddens))
    input = embedding
    for k = 1:length(hiddens)
        weights[2k-1] = init(input+hiddens[k], 4hiddens[k])
        weights[2k] = zeros(1, 4hiddens[k])
        weights[2k][1:hiddens[k]] = 1 # forget gate bias
        input = hiddens[k]
    end
    return map(w->convert(atype, w), weights)
end


# state initialization
# s[2k-1], s[2k] : hidden and cell respectively
function initstate(atype, hiddens, batchsize)
    state = Array(Any, 2length(hiddens))
    for k=1:length(hiddens)
        state[2k-1] = atype(zeros(batchsize, hiddens[k]))
        state[2k] = atype(zeros(batchsize, hiddens[k]))
    end
    return state
end


function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = similar(output)
    @inbounds for i=1:length(output)
        index[i] = i + (output[i]-1)*nrows
    end
    o1 = logp(ypred,2)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end

# model converter for saving
convertmodel{T<:Number}(x::KnetArray{T}) = convert(Array{T}, x)
convertmodel{T<:Number}(x::Array{T}) = convert(Array{T}, x)
convertmodel(a::Associative)=Dict(k=>convertmodel(v) for (k,v) in a)
convertmodel(a) = map(x->convertmodel(x), a)

# optimization parameter creator for parameters
oparams{T<:Number}(::KnetArray{T},otype; o...)=otype(;o...)
oparams{T<:Number}(::Array{T},otype; o...)=otype(;o...)
oparams(a::Associative,otype; o...)=Dict(k=>oparams(v,otype;o...) for (k,v) in a)
oparams(a,otype; o...)=map(x->oparams(x,otype;o...), a)


# reverse model converted for loading from file
revconvertmodel{T<:Number}(x::Array{T}) = convert(KnetArray{T}, x)
revconvertmodel(a::Associative) = Dict(k=>revconvertmodel(v) for (k, v) in a)
revconvertmodel(a) = map(x->revconvertmodel(x), a)

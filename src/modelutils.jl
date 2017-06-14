# Implementation of common model utilies

# character based model utilities
cembed(m) = m[1]
wchar(m) = m[2]; bchar(m) = m[3];
wforw(m) = m[4]; bforw(m) = m[5];
wback(m) = m[6]; bback(m) = m[7];
wsoft(m) = m[8]; bsoft(m) = m[9];

# To-load uniquely trained word-based language model
makewmodel1(d)=[ d["cembed"],
                 d["char"][1],
                 d["char"][2],
                 d["forw"][1],
                 d["forw"][2],
                 d["back"][1],
                 d["back"][2],
                 d["soft"][1],
                 d["soft"][2] ]


# In order to load the word-model either as gpu or normal Array
function makewmodel(d)
    d1 = makewmodel1(d)
    if gpu() >= 0
        return map(KnetArray, d1)
    else
        return map(Array, d1)
    end
end


# col-major lstm
function lstm(weight, bias, hidden, cell, input; mask=nothing)
    gates = weight * vcat(input, hidden) .+ bias
    H = size(hidden, 1)
    forget = sigm(gates[1:H, :])
    ingate = sigm(gates[1+H:2H, :])
    outgate = sigm(gates[1+2H:3H, :])
    change = tanh(gates[1+3H:4H, :])
    (mask != nothing) && (mask = reshape(mask, 1, length(mask)))

    cell = cell .* forget + ingate .* change
    hidden = outgate .* tanh(cell)
    if mask != nothing
        hidden = hidden .* mask
        cell = cell .* mask
    end
    return (hidden, cell)
end


# Token-batched words are coming as input to the system, col-major lstm
function charlstm(model, data, mask)
    weight, bias, embeddings = wchar(model), bchar(model), cembed(model)
    T = length(data)
    B = length(data[1])
    H = div(length(bias), 4)

    
    if isa(weight, KnetArray)
        mask = map(KnetArray, mask)
    end
    
    czero = fill!(similar(bias, H, B), 0)
    hidden = cell = czero
    for t in 1:T
        (hidden, cell) = lstm(weight, bias, hidden, cell, embeddings[:, data[t]]; mask=mask[t])
    end
    return hidden
end


# col-major bilstm implementation
function wordlstm(model, data, mask, embeddings)
    weight, bias = wforw(model), bforw(model)
    T = length(data)
    B = length(data[1])
    H = div(length(bias), 4)


    if isa(weight, KnetArray)
        mask = map(KnetArray, mask)
    end
    
    wzero = fill!(similar(bias, H, B), 0)
    # forward lstm
    hidden = cell = wzero
    fhiddens = Array(Any, T-2)
    for t in 1:T-2
        (hidden, cell) = lstm(weight, bias, hidden, cell, embeddings[:, data[t]]; mask=mask[t])
        fhiddens[t] = hidden
    end

    # backward lstm
    weight_b, bias_b = wback(model), bback(model)
    hidden = cell = wzero
    bhiddens = Array(Any, T-2)
    for t in T:-1:3
        (hidden, cell) = lstm(weight_b, bias_b, hidden, cell, embeddings[:, data[t]]; mask=mask[t])
        bhiddens[t-2] = hidden
    end
    return fhiddens, bhiddens
end


# col-major loss function implementation, as explained in paper
function lmloss(model, data, mask, forw, back; result=nothing)
    T = length(data)
    B = length(data[1])
    weight, bias = wsoft(model), bsoft(model)
    idx(t,b,n) = data[t][b] + (b-1)*n

    total = count = 0
    for t in 1:T
        ypred = weight * vcat(forw[t], back[t]) .+ bias
        nrows,ncols = size(ypred)
        index = Int[]
        for b=1:B
            if mask[t][b]==1
                push!(index, idx(t,b,nrows))
            end
        end
        o1 = logp(ypred, 1)
        o2 = o1[index]
        total += sum(o2)
        count += length(o2)
    end
    
    if result != nothing
        result[1] += AutoGrad.getval(total)
        result[2] += count
    end
    return total
end

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


# xavier initialization
function initx(d...; ftype=Float32)
    if gpu() >=0
        KnetArray{ftype}(xavier(d...))
    else
        Array{ftype}(xavier(d...))
    end
end


# random normal initialization
function initr(d...; ftype=Float32, GPUFEATS=false)
    if GPUFEATS && gpu() >=0
        KnetArray{ftype}(0.1*randn(d...))
    else
        Array{ftype}(0.1*randn(d...))
    end
end

# zero initialization
function initzeros(d...; ftype=Float32, GPUFEATS=false)
    if GPUFEATS && gpu() >=0
        KnetArray{ftype}(zeros(d...))
    else
        Array{ftype}(zeros(d...))
    end
end


function makepmodel1(d; GPUFEATS=false)
    m = ([d["postagv"],d["deprelv"],d["lcountv"],d["rcountv"],d["distancev"],d["parserv"]],
         [d["postago"],d["deprelo"],d["lcounto"],d["rcounto"],d["distanceo"],d["parsero"]])

    if gpu() >= 0
        if GPUFEATS
            return map2gpu(m)
        else
            m = map2cpu(m)
            m[1][6] = map2gpu(m[1][6])
            m[2][6] = map2gpu(m[2][6])
            return m
        end
    else
        return map2cpu(m)
    end
end


function makepmodel2(o, s; ftype=Float32, intype=:normal)
    model = Any[]
    dpostag, ddeprel, dcount = o[:embed]
    for (k,n,d) in ((:postag,17,dpostag),(:deprel,37,ddeprel),(:lcount,10,dcount),(:rcount,10,dcount),(:distance,10,dcount))
        if intype == :normal
            push!(model, [ initr(d) for i=1:n ])
            @msg "Random normal initialization"
        else
            push!(model, [ initzeros(d) for i=1:n ])
            @msg "Zero initialization"
        end
    end
    p = o[:arctype](s)
    f = features([p], o[:feats], model)
    mlpdims = (length(f), o[:hidden]..., p.nmove)
    info("mlpdims=$mlpdims")
    parser = Any[]
    for i=2:length(mlpdims)
        push!(parser, initx(mlpdims[i],mlpdims[i-1]))
        push!(parser, initx(mlpdims[i],1))
    end
    push!(model,parser)
    optim = initoptim(model,o[:optimization])
    return model,optim
end


function makepmodel_lstm(o, s; ftype=Float32, intype=:normal)
    model = Any[]
    dpostag, ddeprel, dcount = o[:embed]
    for (k,n,d) in ((:postag,17,dpostag),(:deprel,37,ddeprel),(:lcount,10,dcount),(:rcount,10,dcount),(:distance,10,dcount))
        if intype == :normal
            push!(model, [ initr(d) for i=1:n ])
            @msg "Random normal initialization"
        else
            push!(model, [ initzeros(d) for i=1:n ])
            @msg "Zero initialization"
        end
    end
    p = o[:arctype](s)
    f = features([p], o[:feats], model)
    lstmdims = (length(f), o[:hidden]..., p.nmove)
    info("lstmdims=$lstmdims")
    parser = Any[]

    # For now we have a single layer lstm
    # TODO: fix it for multi-layer implementation
    push!(parser, initx(4lstmdims[2], o[:hidden][1]+lstmdims[1]))
    bias = zeros(FTYPE, 4lstmdims[2], 1)
    bias[1:o[:hidden][1]] = 1 # forget gate bias
    if gpu() >= 0
        bias = convert(KnetArray, bias)
    end
    push!(parser, bias)
    
    # output layer implementation
    push!(parser, initx(lstmdims[end], o[:hidden][end])) # weight
    bfin = (gpu()>=0 ? KnetArray(zeros(FTYPE, lstmdims[end], 1)) : zeros(FTYPE, lstmdims[end], 1))
    push!(parser, bfin) # bias
    push!(model,parser)
    optim = initoptim(model,o[:optimization])
    return model,optim
end


# Parser model initialization, and parameter selection
makepmodel(d, o, s) = (haskey(d, "parserv") ? makepmodel1(d) : makepmodel2(o, s; intype=:normal))
postagv(m)=m[1]; deprelv(m)=m[2]; lcountv(m)=m[3]; rcountv(m)=m[4]; distancev(m)=m[5]; parserv(m)=m[6]


function mlp(w,x; pdrop=(0,0))
    x = dropout(x,pdrop[1])
    for i=1:2:length(w)-2
        x = relu(w[i]*x .+ w[i+1])
        x = dropout(x,pdrop[2])
    end
    return w[end-1]*x .+ w[end]
end


function lstm_parser(lmodel, states, input; mask=nothing, pdrop = (0,0))
    input = dropout(input, pdrop[1])
    states[1], states[2]  = lstm(lmodel[1], lmodel[2], states[1], states[2], input)
    x = dropout(states[1], pdrop[2])
    return lmodel[3] * x .+ lmodel[4]
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

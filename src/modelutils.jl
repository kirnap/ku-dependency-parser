# Implementation of common model utilies

# character based model utilities
cembed(m) = m[1]
wchar(m) = m[2]; bchar(m) = m[3];
wforw(m) = m[4]; bforw(m) = m[5];
wback(m) = m[6]; bback(m) = m[7];
wsoft(m) = m[8]; bsoft(m) = m[9];

# To-load uniquely trained word-based language model
makewmodel1(d)=[d["cembed"],d["char"][1],d["char"][2],d["forw"][1],d["forw"][2],d["back"][1],d["back"][2],d["soft"][1],d["soft"][2]]

# col-major lstm
function lstm(weigth, bias, hidden, cell, input; mask=nothing)
    gates = weight * vcat(input, hidden) .+ bias
    H = size(hidden, 2)
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

# Token-batched words are coming as input to the system
function charlstm(model, data, mask)
    weight, bias, cembed = wchar(model), bchar(model), cembed(model)
    T = length(data)
    B = length(data[1])
    H = div(length(bias), 4)
    
end

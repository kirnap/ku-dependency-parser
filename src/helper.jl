# General helper functions

# when fmatrix has mixed Rec and KnetArray, vcat does not do the right thing!  AutoGrad only looks at the first 2-3 elements!
#TODO: temp solution to AutoGrad vcat issue:
using AutoGrad
let cat_r = recorder(cat); global vcatn, hcatn
    function vcatn(a...)
        if any(x->isa(x,Rec), a)
            cat_r(1,a...)
        else
            vcat(a...)
        end
    end
    function hcatn(a...)
        if any(x->isa(x,Rec), a)
            cat_r(2,a...)
        else
            hcat(a...)
        end
    end
end

# KnetArray cleaning
function free_KnetArray()
    gc(); Knet.knetgc(); gc()
end


# To fill the constant word embeddings
function fillwvecs!(sentences, isents, wembed; GPUFEATS=false)
    for (s, isents) in zip(sentences, isents)
        empty!(s.wvec)
        for w in isents
            if GPUFEATS
                push!(s.wvec, wembed[:, w])
            else
                push!(s.wvec, Array(wembed[:, w]))
            end
        end
    end
end


# To fill the context embeddings of words in given sentences one by one
function fillcvecs!(sentences, forw, back; GPUFEATS=false)
    T = length(forw)
    for i in 1:length(sentences)
        s = sentences[i]
        empty!(s.fvec)
        empty!(s.bvec)
        N = length(s)
        for n in 1:N
            t = T-N+n
            if GPUFEATS #GPU
                push!(s.fvec, forw[t][:,i])
                push!(s.bvec, back[t][:,i])
            else #CPU
                push!(s.fvec, Array(forw[t][:,i]))
                push!(s.bvec, Array(back[t][:,i]))
            end
        end
    end
end

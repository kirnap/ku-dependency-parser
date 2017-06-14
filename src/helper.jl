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

# To create a vocabulary file
# for a given text file it counts all the words and writes result to a given save file
using ArgParse

function count_words(file::AbstractString; wfile=nothing)
    res = Dict{AbstractString, Int}()
    open(file) do f
        for line in eachline(f)
            words = split(line)
            for word in words
                if word in keys(res)
                    res[word] += 1
                else
                    res[word] = 1
                end
            end
        end
    end

    sorted_pairs = sort(collect(res), by=x->res[x[1]], rev=true)
    if wfile != nothing
        open(wfile, "w") do f
            for (k, v) in sorted_pairs
                write(f, "$v $k\n")
            end
        end
    else
        for (k, v) in sorted_pairs; println("$v $k");end;
    end

end


function main(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--textfile"; required=true; help="Raw text file containing all the information")
        ("--output"; help="Output count file contains the word frequency information")
        
    end
    
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    count_words(o[:textfile]; wfile=o[:output])
    
end
!isinteractive() && main(ARGS)

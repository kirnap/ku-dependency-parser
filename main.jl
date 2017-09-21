using JLD, Knet, ArgParse
include("parser/types.jl")
include("parser/preprocess.jl")
include("parser/parser.jl")
include("parser/helper.jl")
include("parser/modelutils.jl")
include("parser/features.jl")
include("parser/train.jl")


function train(args=ARGS)
    s = ArgParseSettings()
    s.description="Koc-University transition based parser"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--datafiles"; nargs='+'; help="Input in conllu format. If provided, use first file for training, last for dev. If single file use both for train and dev.")
        ("--output"; help="Output parse of first datafile in conllu format to this file")
        ("--loadfile"; help="Initialize model from file")
        ("--savefile"; help="Save final model to file")
        ("--bestfile"; help="Save best model to file")
        ("--hidden"; nargs='+'; arg_type=Int; default=[2048]; help="Sizes of parser mlp hidden layers.")
        ("--optimization"; default="Adam()"; help="Optimization algorithm and parameters.")
        ("--seed"; arg_type=Int; default=-1; help="Random number seed.")
        ("--otrain"; arg_type=Int; default=0; help="Epochs of oracle training.")
        ("--btrain"; arg_type=Int; default=0; help="Epochs of beam training.")
        ("--arctype"; help="Move set to use: ArcEager{R1,13}, ArcHybrid{R1,13}, default ArcHybridR1")
        ("--feats"; help="Feature set to use, default $FEATS")
        ("--batchsize"; arg_type=Int; default=16; help="Number of sequences to train on in parallel.")
        ("--beamsize"; arg_type=Int; default=1; help="Beam size.")
        ("--dropout"; nargs='+'; arg_type=Float64; default=[0.5,0.5]; help="Dropout probabilities. default 0.5.")
        ("--report"; nargs='+'; arg_type=Int; default=[1]; help="choose which files to report las for, default all.")
        ("--embed"; nargs='+'; arg_type=Int; help="embedding sizes for postag(17),deprel(37),counts(10). default 128,32,16.")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println(s.description)
    main(o)
end
!isinteractive() && train(ARGS)

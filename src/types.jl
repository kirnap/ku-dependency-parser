# Types that are taken from conll file and used in parser
typealias Word String           # stands for words in sentence
typealias PosTag  UInt8         # 17 universal part-of-speech tags
typealias DepRel UInt8          # 37 universal depedency relations
typealias Position Int16        # sentence position
typealias Cost Position         # [0:nword]
typealias Move Int              # [1:nmove]
typealias WordId Int32          # [1:nvocab]
typealias Pvec Vector{Position} # used for stack, head ? don't sure what it is
typealias Dvec Vector{DepRel}   # used for deprel


# Universal POS tags (17)
const UPOSTAG = Dict{String,PosTag}(
"ADJ"   => 1, # adjective
"ADP"   => 2, # adposition
"ADV"   => 3, # adverb
"AUX"   => 4, # auxiliary
"CCONJ" => 5, # coordinating conjunction
"DET"   => 6, # determiner
"INTJ"  => 7, # interjection
"NOUN"  => 8, # noun
"NUM"   => 9, # numeral
"PART"  => 10, # particle
"PRON"  => 11, # pronoun
"PROPN" => 12, # proper noun
"PUNCT" => 13, # punctuation
"SCONJ" => 14, # subordinating conjunction
"SYM"   => 15, # symbol
"VERB"  => 16, # verb
"X"     => 17, # other
)

# Universal Dependency Relations (37)
const UDEPREL = Dict{String,DepRel}(
"root"       => 1,  # root
"acl"        => 2,  # clausal modifier of noun (adjectival clause)
"advcl"      => 3,  # adverbial clause modifier
"advmod"     => 4,  # adverbial modifier
"amod"       => 5,  # adjectival modifier
"appos"      => 6,  # appositional modifier
"aux"        => 7,  # auxiliary
"case"       => 8,  # case marking
"cc"         => 9,  # coordinating conjunction
"ccomp"      => 10, # clausal complement
"clf"        => 11, # classifier
"compound"   => 12, # compound
"conj"       => 13, # conjunct
"cop"        => 14, # copula
"csubj"      => 15, # clausal subject
"dep"        => 16, # unspecified dependency
"det"        => 17, # determiner
"discourse"  => 18, # discourse element
"dislocated" => 19, # dislocated elements
"expl"       => 20, # expletive
"fixed"      => 21, # fixed multiword expression
"flat"       => 22, # flat multiword expression
"goeswith"   => 23, # goes with
"iobj"       => 24, # indirect object
"list"       => 25, # list
"mark"       => 26, # marker
"nmod"       => 27, # nominal modifier
"nsubj"      => 28, # nominal subject
"nummod"     => 29, # numeric modifier
"obj"        => 30, # object
"obl"        => 31, # oblique nominal
"orphan"     => 32, # orphan
"parataxis"  => 33, # parataxis
"punct"      => 34, # punctuation
"reparandum" => 35, # overridden disfluency
"vocative"   => 36, # vocative
"xcomp"      => 37, # open clausal complement
)


immutable Vocab
    cdict::Dict{Char, Int}         # character vocabulary
    idict::Dict{String, Int}       # word dictionary (input) obtained from conll file
    odict::Dict{String, Int}       # word dictionary (output) obtained from lm training
    sosword::String                # start of sentence word (<s>)
    eosword::String                # end of sentence word (</s>)
    unkword::String                # unknown word (<unk>)
    sowchar::Char                  # sow character chosen from hardware character
    eowchar::Char                  # eow character chosen from hardware character
    unkchar::Char  
    postags::Dict{String, PosTag}
    deprels::Dict{String, DepRel} 
end

                                # CONLLU FORMAT
type Sentence                   # 1. ID: Word index, integer starting at 1 for each new sentence
    word::Vector{Word}          # 2. FORM: Word form or punctuation symbol
    # stem::Vector{Stem}        # 3. LEMMA: Lemma for m or punctiation symbol
    postag::Vector{PosTag}      # 4. UPOSTAG: Universal part-of-speech tag
    # xpostag::Vector{XPosTag}  # 5. Language specific pos-tag
    # feats::Vector{Feats}      # 6. FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available
    head::Vector{Position}      # 7. HEAD: Head of the current word, which is either a vvalue of ID or zero 
    deprel::Vector{DepRel}       # 8. DEPREL: Universal dependency relation to the HEAD(root iff HEAD=0) or a defined language-specific subtype
    # deps::Vector{Deps}        # 9. DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
    # misc::Vector{Misc}        # 10. MISC: Any other annotation.

    # language model dependent features
    wvec::Vector                # word vectors
    fvec::Vector                # forw context vectors
    bvec::Vector                # backw context vectors
    vocab::Vocab                # go to 13
    parse

    Sentence(v::Vocab) = new([],[], [], [], [], [], [], v, nothing)
end

Base.length(s::Sentence) = length(s.word)

# add-hoc solution for parser.jl not sure whether it is need?
typealias Corpus AbstractVector{Sentence}

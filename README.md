# Koç University Dependency Parser 
Dependency parser implementation used by [Koç University](https://www.ku.edu.tr) team in [Conll17 shared task](http://universaldependencies.org/conll17/). Our team ranked 7th as posted in the [results](http://universaldependencies.org/conll17/results.html).

## Getting started 
This document will guide you to get a working copy of dependency parser software on your machine. The system has two parts; language modelling and dependency parsing. Most updated version of source can be found on [the official repo](https://github.com/kirnap/ku-dependency-parser).


### Prerequisites
We use text files tokenized by [UDPipe](http://ufal.mff.cuni.cz/udpipe), please make sure that you have installed it from their official [repository](https://github.com/ufal/udpipe).

Our entire software runs on [Julia](https://julialang.org/), so it should be installed on your system as well. Julia can be installed from their [official download page](https://julialang.org/downloads/). After the requirements are met, follow the installation instructions below. 

### Installing
Clone the repository to install the parser and dependencies:

```sh
git clone https://github.com/kirnap/ku-dependency-parser.git && cd ku-dependency-parser
julia installer.jl

```

### Code Structure
Dependency parser related code is under [parser](https://github.com/kirnap/ku-dependency-parser/tree/master/parser) folder and Language Model related code is under [lm](https://github.com/kirnap/ku-dependency-parser/tree/master/lm).

### Running
To be able to train parser on a specific language, first you need to have a pre-trained language model so that you can generate *context* and *word* embeddings for that language. Here are the steps to train a language model:

#### Bi-LSTM based Language Model

Switch to language model directory: 
```sh
cd lm
```

If you do not have raw version of `.conllu` formatted file, run the following to obtain tokenized raw text:
```sh
udpipe --output=horizontal none --outfile texts/{}.txt *.conllu
```

Create a vocabulary file from the text file that is tokenized by UDPipe (provided by Conll17 task organizers), please notice that output file contains word-frequency information for the supplied text file:
```sh
julia wordcount.jl --textfile 'input text file' --output 'vocabulary-file'
```
Language model training expects the vocabulary file not contain any frequency information, thus using linux tools remove that frequency information:
```sh
awk '{$1="";print $0}' path/to/vocabulary-file > path/to/words-file
```

Create a file that includes top N (e.g. 10000) words to be used during the training:
```sh
head -n10000 path/to/words-file > path/to/vocab-file
```

To train the language model, you need to run the following command:
```sh
julia lm_train.jl --trainfile 'udpipe-output.txt' --vocabfile 'path/to/vocab-file'  --wordsfile 'path/to/words-file' --savefile model.jld
```

**Warning:** Please be aware that language model training takes approximately 24 hours on Tesla K80 GPU.

#### Dependency parser
Go to the parent directory and run the following command:
```sh
julia main.jl --load '/path/to/pre-trained language model' --datafiles 'path/to/train_file.conllu' 'path/to/dev_file.conllu' --otrain 'number of epochs'
```
For more detailed options, run:
```sh
julia main.jl --help
```

## Additional help
For more help, you are welcome to [open an issue](https://github.com/kirnap/ku-dependency-parser/issues/new), or directly contact [okirnap@ku.edu.tr](mailto:okirnap@ku.edu.tr).

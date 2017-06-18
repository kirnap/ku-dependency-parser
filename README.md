# Koç University Dependency Parser 
Dependency parser implementation used in [Conll17 shared task](http://universaldependencies.org/conll17/), our team Koç-University ranked 7th that can be found in [results](http://universaldependencies.org/conll17/results.html).

## Getting started 
These instructions will get you a copy of dependency parser software on your machine. Once you installed whole system, you will have language modelling part and dependency parsing part. Most updated version can be found in [here](https://github.com/kirnap/dependencyParser)

### Prerequisites
Entire software runs on Julia, so you need to install it from their official [download](https://julialang.org/downloads/) page. After successfully downloading Julia run the following command from terminal to install package dependencies.
We use text file tokenized by [UDPipe](http://ufal.mff.cuni.cz/udpipe), please make sure that you have installed the (UDPipe) from official download page.


### Installing
Clone the repository to install the parser:

```sh
git clone git@github.com:kirnap/dependencyParser.git
julia installer.jl

```





### Code Structure
Dependency parser related code is under [src](https://github.com/kirnap/dependencyParser/tree/master/src) folder and Language Model related code is under [lm](https://github.com/kirnap/dependencyParser/tree/master/lm).

### Running
To be able to train parser on a specific language, first you need to have pre-trained language model so that you can generate *context* and *word* embeddings for that specific language. Here are the steps to train a language model:

#### BiLSTM based LM

Go to the directory where you download all the software and then to language model directory
```sh
cd lm
```

If you don not raw version of .conllu formatted file run the following to obtain tokenized raw text:
```sh
udpipe --output=horizontal none --outfile texts/{}.txt *.conllu
```

Create a vocabulary from a text file that is tokenized  by UDPipe (provided by Conll17 task organizers):
```sh
julia wordcount.jl --textfile 'your text file' --countfile 'your-vocabulary-file'
```
Lm trainer expects the vocabulary file that does not contain frequency information, thus one can use linux tools to get rid of frequency information:
```sh
awk '{$1="";print $0}' path/to/your-vocabulary-file
```

To train lm you need to run the following command:
```sh
julia lm_train.jl --trainfile 'your text file' --vocabfile 'your output vocabfile'  --wordsfile 'your input vocabfile' --savefile your_model.jld
```

#### To run dependency parser
Go to the parent directory and run the following command:
```sh
julia main.jl --load '/path/to/pre-trained language model' --datafiles 'path/to/your_train_file.conllu' 'path/to/your_dev_file.conllu' --otrain 'number of epochs'
```
For more detailed options you can run:
```sh
julia main.jl --help
```





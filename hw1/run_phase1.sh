#!/bin/sh

if [ -d "w2v.models" ] || [ -d $2  ] ; then
  rm -rf w2v.models $2
fi
mkdir w2v.models $2

python2.7 glove.py -i $1
python2.7 myWord2vec.py --train_data=$1 --eval_data=questions-words.txt --save_path=w2v.models/ 
python2.7 filterVocab.py fullVocab.txt < vec_glove > $2/filter_glove.txt
python2.7 filterVocab.py fullVocab.txt < vec_word2vec > $2/filter_word2vec.txt

#!/bin/sh

if [ -d "w2v.models" ] || [ -d $2  ] ; then
  rm -rf w2v.models $2
fi
mkdir w2v.models $2

python2.7 myWord2vec_ptt.py --train_data=$1 --eval_data=questions-words-phase2-dev.txt --save_path=w2v.models/
python2.7 filterVocab.py fullVocab_phase2.txt < vec_word2vec_ptt > $2/filter_vec.txt

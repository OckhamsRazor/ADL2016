#!/bin/sh

python2.7 rnn.int.eval.py -i atis.test.iob -c int.dr40.class -m int.dr40/ -o answer.txt -hi 128

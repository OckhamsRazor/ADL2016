#!/bin/sh

python2.7 rnn.eval.py -i atis.test.iob -c slot.128.dr20.class -o answer.txt -m slot.128.dr20 -hi 128

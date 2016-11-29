import argparse
import random
import types
from os import mkdir, path
from shutil import rmtree
from sys import stdout

import numpy as np
import tensorflow as tf


num_layers = 2
seq_max_len = 28


def load_word_embs(f):
    embs = {}
    with open (f) as fin:
        for line in fin:
            tokens = line.rstrip().split()
            embs[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
    return embs


def load_test_data(f):
    sents = []
    with open (f) as fin:
        for line in fin:
            sent = line.rstrip().split()
            sents.append(sent)
    return sents


def make_sent_batch(sents, embs, n_input, pos, b_size):
    b_x, b_seqlen = [], []
    def sent2emb(sent):
        inps = []
        for i, w in enumerate(sent):
            if i == seq_max_len:
                break

            if w not in embs:
                emb = embs["UNK"]
            else:
                emb = embs[w]
            inps.append(emb.reshape(n_input, 1))
        for _ in range(len(sent), seq_max_len):
            inps.append(np.zeros((n_input, 1)))

        return np.concatenate(inps, axis=1).transpose()

    for b in range(b_size):
        b_x.append(sent2emb(sents[pos+b]).reshape(1, seq_max_len, n_input))
        l = len(sents[pos+b])
        if l > seq_max_len:
            l = seq_max_len
        b_seqlen.append(l)
    b_x = np.concatenate(b_x)
    b_seqlen = np.array(b_seqlen)

    return b_x, b_seqlen


def dynamic_rnn(x, seq_len, W, b, n_hidden, n_input, keep_prob):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, seq_max_len, x)

    x = [tf.nn.dropout(x_ele, keep_prob) for x_ele in x]

    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    lstm_cell_dropped = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob)
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_dropped]*num_layers)
    stacked_lstm_dropped = tf.nn.rnn_cell.DropoutWrapper(stacked_lstm, output_keep_prob=keep_prob)
    outputs, states = tf.nn.rnn(stacked_lstm, x, dtype=tf.float32, sequence_length=seq_len)

    outputs = tf.pack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * seq_max_len + (seq_len - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    return tf.matmul(outputs, W['out']) + b['out']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNN_slot')
    parser.add_argument(
        "-i", "--input", help="testing data", required=True
    )
    parser.add_argument(
        "-c", "--class_f", help="input class file", required=True
    )
    parser.add_argument(
        "-m", "--model", help="model loading dir", required=True
    )
    parser.add_argument(
        "-o", "--output", help="output file", required=True
    )
    parser.add_argument(
        "-hi", "--n-hidden", default=300
    )
    args = parser.parse_args()

    input_f = args.input
    output = args.output
    class_f = args.class_f
    model_dir = args.model

    n_input = 200
    n_hidden = int(args.n_hidden)
    n_classes = None # determined after data loaded

    embs = load_word_embs("vec_word2vec")
    sents = load_test_data(input_f)
    id2type = []
    type2id = {}
    with open(class_f) as class_fin:
        for line in class_fin:
            id2type.append(line.rstrip())

    for idx, t in enumerate(id2type):
        type2id[t] = idx

    n_classes = len(id2type)
    
    x = tf.placeholder("float", [None, seq_max_len, n_input])
    seq_len = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32)

    W = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    b = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    preds = dynamic_rnn(x, seq_len, W, b, n_hidden, n_input, keep_prob)
    saver = tf.train.Saver()
    with tf.Session() as sess, open(output, "w") as fout:
        saver.restore(sess, path.join(model_dir, "model.ckpt"))
        for pos in range(len(sents)):
            b_x, b_seqlen = make_sent_batch(sents, embs, n_input, pos, 1)
            pred = sess.run(preds, feed_dict={x: b_x, seq_len:b_seqlen, keep_prob:1.0})
            fout.write(id2type[np.argmax(pred)])
            fout.write("\n")

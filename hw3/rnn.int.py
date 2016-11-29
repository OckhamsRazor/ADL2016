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


def load_data(f):
    sents = []
    labs = []
    with open (f) as fin:
        for line in fin:
            sent, lab = line.rstrip().split("\t")
            sent = sent.split()
            sents.append(sent)
            labs.append(lab.split()[-1])
    return sents, labs


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


def make_lab_batch(labs, type2id, pos, b_size):
    n_classes = len(type2id)
    b_y = []
    for b in range(b_size):
        yy = [0]*n_classes
        yy[type2id[labs[pos+b]]] = 1
        b_y.append(yy)

    return np.array(b_y)


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
        "-i", "--input", help="training data", required=True
    )
    parser.add_argument(
        "-v", "--dev_set", help="validation data", default="atis.dev.iob"
    )
    parser.add_argument(
        "-c", "--class_f", help="output class file", default="int.class"
    )
    parser.add_argument(
        "-m", "--model", help="model saving dir", required=True
    )
    parser.add_argument(
        "-s", "--steps", default=1e5
    )
    parser.add_argument(
        "-hi", "--n-hidden", default=128
    )
    parser.add_argument(
        "-l", "--lr", default=1e-1
    )
    parser.add_argument(
        "-d", "--dropout", default=0.6
    )
    args = parser.parse_args()

    input_f = args.input
    dev_f = args.dev_set
    class_f = args.class_f
    model_dir = args.model
    if path.exists(model_dir):
        rmtree(model_dir)
    mkdir(model_dir)

    total_steps = int(args.steps)
    n_input = 200
    n_hidden = int(args.n_hidden)
    n_classes = None # determined after data loaded
    init_lr = float(args.lr)
    kp_prb = float(args.dropout)
    batch_size = 8

    embs = load_word_embs("vec_word2vec")
    sents, labs = load_data(input_f)
    dev_sents, dev_labs = load_data(dev_f)
    id2type = []
    type2id = {}
    for l in labs:
        if l not in type2id:
            type2id[l] = len(id2type)
            id2type.append(l)

    n_classes = len(id2type)
    with open(class_f, "w") as class_fout:
        for class_ in id2type:
            class_fout.write(class_+"\n")
    
    x = tf.placeholder("float", [None, seq_max_len, n_input])
    y = tf.placeholder("float", [None, n_classes])
    seq_len = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)

    W = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    b = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    preds = dynamic_rnn(x, seq_len, W, b, n_hidden, n_input, keep_prob)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(preds, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
    
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        current_lr = init_lr
        ep_size = len(sents) / batch_size
        errs = []
        prev_dev_err = None
        for step in range(total_steps):
            pos = (step * batch_size) % len(sents)
            if pos + batch_size >= len(sents):
                pos = 0
            if step % (ep_size/10) == 0 and len(errs) > 0:
                err = sum(errs) / len(errs)
                print "ep "+str(float(step)/ep_size)+" err: "+str(err)
            if pos == 0 and len(errs) > 0:
                print "  ep "+str(float(step)/ep_size)+" passed."
                errs = []
                dev_errs = []
                for dev_pos in range(0, len(dev_sents), batch_size):
                    if dev_pos+batch_size >= len(dev_sents):
                        break
                    b_x, b_seqlen = make_sent_batch(dev_sents, embs, n_input, dev_pos, batch_size)
                    b_y = make_lab_batch(dev_labs, type2id, dev_pos, batch_size)
                    c = sess.run(cost, feed_dict={x: b_x, y: b_y, seq_len:b_seqlen, keep_prob:1.0})
                    dev_errs.append(c)
                dev_err = sum(dev_errs) / len(dev_errs)
                print " dev err: "+str(dev_err)
                if prev_dev_err is None or dev_err < prev_dev_err:
                    prev_dev_err = dev_err
                    saver.save(sess, path.join(model_dir, "model.ckpt"))
                else:
                    current_lr *= 0.9

            b_x, b_seqlen = make_sent_batch(sents, embs, n_input, pos, batch_size)
            b_y = make_lab_batch(labs, type2id, pos, batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: b_x, y: b_y, seq_len:b_seqlen, keep_prob:kp_prb, lr: current_lr})
            errs.append(c)
            stdout.flush()

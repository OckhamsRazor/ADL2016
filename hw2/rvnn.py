import argparse
import random
import types
from os import mkdir, path
from shutil import rmtree
from sys import stdout

import numpy as np
import tensorflow as tf
from pyparsing import *


my_printables = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'*+,-./:;<=>?@[\]^_`{|}~"


def load_word_embs(f):
    embs = {}
    with open (f) as fin:
        for line in fin:
            tokens = line.rstrip().split()
            embs[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
    return embs


def load_sent(f):
    sents = []
    with open (f) as fin:
        for line in fin:
            sents.append(line.rstrip().split())
    return sents


def load_tree_to_exprlist(f):
    trees = []
    enclosed = Forward()
    nestedParens = nestedExpr('(', ')', content=enclosed)
    enclosed << (Word(my_printables) | ',' | nestedParens)
    with open (f) as tree_file:
        tree = ""
        for line in tree_file:
            line = line.strip()
            if len(line) > 0:
                tree += line
            else:
                tree = tree.replace("'", "`")
                try:
                    trees.append(enclosed.parseString(tree).asList())
                except ParseException:
                    trees.append([])
                tree = ""
    return trees


def add_expr_to_list(exprlist, expr):
    # if expr is a atomic type
    if isinstance(expr, types.ListType):
        # Now for rest of expression
        for e in expr[1:]:
            # Add to list if necessary
            if not (e in exprlist):
                add_expr_to_list(exprlist, e)
    # Add index in list.
    exprlist.append(expr)


def expand_subexprs(exprlist):
    new_exprlist = []
    orig_indices = []
    for e in exprlist:
        add_expr_to_list(new_exprlist, e)
        orig_indices.append(len(new_exprlist)-1)
    return new_exprlist, orig_indices


def compile_expr(exprlist, expr):
    # start new list starting with head
    new_expr = [expr[0]]
    for e in expr[1:]:
        new_expr.append(exprlist.index(e))
    return new_expr


def compile_expr_list(exprlist):
    new_exprlist = []
    for e in exprlist:
        if isinstance(e, types.ListType):
            new_expr = compile_expr(exprlist, e)
        else:
            new_expr = e
        new_exprlist.append(new_expr)
    return new_exprlist


def expand_and_compile(exprlist):
    l, orig_indices = expand_subexprs(exprlist)
    return compile_expr_list(l), orig_indices


def new_weight(N1,N2):
    return tf.Variable(tf.random_normal([N1,N2]))


def new_bias(N_hidden):
    return tf.Variable(tf.random_normal([N_hidden]))


def build_weights(exprlists, inp_vec_len, N_hidden, out_vec_len):
    W, b = {}, {}
    for exprlist in exprlists:
        try:
            l, _ = expand_and_compile(exprlist)
        except IndexError:
            continue

        for expr in l:
            if isinstance(expr, types.ListType):
                idx = expr[0]
                size = len(expr)-1
                if not W.has_key(idx):
                    W[idx] = [new_weight(N_hidden,N_hidden) for i in range(2)]
                    #b[idx] = new_bias(N_hidden)
                    #W[idx] = size
                    b[idx] = new_bias(N_hidden)
                #elif W[idx] < size:
                #    W[idx] = size
    

    W['input']  = new_weight(inp_vec_len, N_hidden)
    W['output'] = new_weight(N_hidden, out_vec_len)
    b['input']  = new_weight(1, N_hidden)
    b['output'] = new_weight(1, out_vec_len)
    return (W,b)


def build_rnn_graph(in_vars, exprlist, W, b, inp_vec_len, out_idx):
    #in_vars = [e for e in exprlist if not isinstance(e,types.ListType)]
    N_input = len(in_vars)
    inp_tensor = tf.placeholder(tf.float32, (N_input, inp_vec_len), name='input1')
    V = []      # list of variables corresponding to each expr in exprlist
    i = 0
    for expr in exprlist:
        if isinstance(expr, types.ListType):
            # intermediate variables
            idx = expr[0]
            # add bias
            new_var = b[idx]
            # add input variables * weights
            for i in range(1,len(expr)):
                #new_var = tf.add(new_var, tf.matmul(V[expr[i]], W[idx][i-1]))
                new_var = tf.add(new_var, tf.matmul(V[expr[i]], W[idx][1]))
            new_var = tf.sigmoid(new_var)
        else:
            # base (input) variables
            # TODO : variable or placeholder?
            #i = in_vars.index(expr)
            i_v = tf.slice(inp_tensor, [i,0], [1,-1])
            new_var = tf.sigmoid(tf.add(tf.matmul(i_v, W['input']), b['input']))
            i += 1
        V.append(new_var)

    out_tensor = tf.placeholder(tf.float32, (1, 1), name='output1')
    ce = tf.reduce_sum(tf.zeros((1, 1)))
    o = tf.sigmoid(tf.add(tf.matmul(V[out_idx], W['output']), b['output']))
    #o = tf.nn.softmax(oo)
    ce = tf.add(ce, -(out_tensor * tf.log(o) + (1-out_tensor) * tf.log(1-o)), name='loss')
    #ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(o, out_tensor))

    return (inp_tensor, V, out_tensor, ce, o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RvNN')
    parser.add_argument(
        "-i", "--input", help="test data -- sentences", required=True
    )
    parser.add_argument("-t", "--tree", help="test data -- trees", required=True)
    parser.add_argument(
        "-o", "--output", help="output", required=True
    )
    parser.add_argument(
        "-ep", "--epoch", default=100
    )
    parser.add_argument(
        "-hi", "--n-hidden", default=50
    )
    parser.add_argument(
        "-l", "--lr", default=1e-2
    )
    args = parser.parse_args()

    input_f = args.input
    tree_f = args.tree
    output = args.output

    epoch = int(args.epoch)
    inp_vec_len = 200
    N_hidden = int(args.n_hidden)
    out_vec_len = 1
    lr = float(args.lr)

    embs = load_word_embs("vec_word2vec")
    pos = load_sent("training_data.pos")
    neg = load_sent("training_data.neg")
    pos_trees = load_tree_to_exprlist("training_data.pos.tree")
    neg_trees = load_tree_to_exprlist("training_data.neg.tree")
    sents = pos+neg
    trees = pos_trees+neg_trees
    labels = [1 if i < len(pos) else 0 for i in range(len(sents))]
    order = range(len(sents))

    test_sents = load_sent(input_f)
    test_trees = load_tree_to_exprlist(tree_f)

    W, b = build_weights(trees, inp_vec_len, N_hidden, out_vec_len)
    init = tf.initialize_all_variables()
    bad_data = set()
    with tf.Session() as sess:
        sess.run(init)
        for e in range(epoch):
            random.shuffle(order)
            for ii, idx in enumerate(order):
                if idx in bad_data:
                    continue
                
                exprlist, out_idx = expand_and_compile(trees[idx])
                i, V, o, ce, out = build_rnn_graph(sents[idx], exprlist, W, b, inp_vec_len, out_idx[0])
                train_step = tf.train.GradientDescentOptimizer(lr).minimize(ce)
                inps = []
                for w in sents[idx]:
                    if w not in embs:
                        emb = embs["UNK"]
                    else:
                        emb = embs[w]
                    inps.append(emb.reshape(inp_vec_len, 1))
                inps = np.concatenate(inps, axis=1).transpose()
                try:
                    sess.run([train_step], feed_dict={i:inps, o:np.array([[labels[idx]]])})
                except tf.errors.InvalidArgumentError:
                    bad_data.add(idx)
                    continue
                if ii % (0.2*len(order)) == 10:# and ii > 0:
                    with open(output, "w") as fout:
                        for i_tst in range(len(test_sents)):
                            try:
                                exprlist, out_idx = expand_and_compile(test_trees[i_tst])
                                i, V, o, ce, out = build_rnn_graph(test_sents[i_tst], exprlist, W, b, inp_vec_len, out_idx[0])
                                inps = []
                                for w in test_sents[i_tst]:
                                    if w not in embs:
                                        emb = embs["UNK"]
                                    else:
                                        emb = embs[w]
                                    inps.append(emb.reshape(inp_vec_len, 1))
                                inps = np.concatenate(inps, axis=1).transpose()
                                pred = sess.run([out], feed_dict={i:inps})
                            except Exception:
                                pred = random.random()
                            if pred >= 0.5:
                                fout.write("1\n")
                            else:
                                fout.write("0\n")
                            fout.flush()
            lr *= 0.8

import argparse
from os import mkdir, path
from shutil import rmtree

import tf_glove


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GloVe encoder')
    parser.add_argument(
        "-i", "--corpus", help="training data file", default="corpus/text8"
    )
    parser.add_argument(
        "-o", "--output", help="output", default="vec_glove"
    )
    parser.add_argument(
        "-l", "--log-dir", help="log file directory", default="log"
    )
    parser.add_argument(
        "-e", "--emb-size", help="GloVe embedding size", default=128
    )
    parser.add_argument(
        "-c", "--context-size", help="GloVe context size", default=10
    )
    parser.add_argument(
        "-ep", "--epoch", default=100
    )
    args = parser.parse_args()

    corpus_file = args.corpus
    output = args.output
    log_dir = args.log_dir

    if path.exists(log_dir):
        rmtree(log_dir)
    mkdir(log_dir)

    emb_size = int(args.emb_size)
    context_size = int(args.context_size)
    epoch = int(args.epoch)

    model = tf_glove.GloVeModel(embedding_size=emb_size, context_size=context_size)
    text8 = []
    with open (corpus_file) as fin:
        for line in fin:
            text8 = line.rstrip().split()

    corpus, sent = [], []
    for w in text8:
        sent.append(w)
        if len(sent) == 1000:
            corpus.append(sent)
            sent = []

    model.fit_to_corpus(corpus)
    model.train(num_epochs=int(epoch), log_dir=log_dir)
    with open (output, "w") as fout:
        for word in model.words:
            emb = model.embedding_for(word).tolist()
            fout.write(word)
            for dim in emb:
                fout.write(" ")
                fout.write(str(dim))
            fout.write("\n")

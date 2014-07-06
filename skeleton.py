import sys, os, re
import numpy as np
from operator import itemgetter as ig
from sklearn.linear_model import LogisticRegression as LR
from os.path import isfile, join
from collections import OrderedDict


vocab = OrderedDict()  # the features used in the classifier
stop_words = {}


def get_file_names(path):
    return [path + "/" + file for file in os.listdir(path) if isfile(
        join(path, file))]


def build_stop_words():
    global stop_words
    stop_words = open('stopwords.txt').read().lower().split()
    stop_words = dict(zip(stop_words, stop_words))


def no_punc(in_string):
    return re.sub("[^\w]", " ", in_string)


def get_words_no_punc(inputfile):
    word_list = []
    with open(inputfile, "r") as infile:
        file_words = infile.read().lower()
        word_list.append(no_punc(file_words).split())
    return word_list


# build vocabulary
def buildvocab(num_words):
    global vocab
    global stop_words
    build_stop_words()
    training_files = get_file_names('pos') + get_file_names('neg')
    num_files = len(training_files)
    word_count = OrderedDict()
    i = 0
    for filename in training_files:
        file_words = get_words_no_punc(filename)
        for word in file_words[0]:
            if word not in stop_words.keys():
                if not word.isdigit():
                    if len(word) > 1:
                        if word not in word_count.keys():
                            word_count[word] = 1
                        else:
                            word_count[word] += 1
        i += 1
        print "Initial file processing {:.2%} percent done\r".format(
            i/float(num_files))

    word_count = sorted(word_count.items(), key=lambda t: t[1], reverse=True)
    vocab = OrderedDict(word_count[0:num_words])
    return vocab


def vectorize(inputfile):
    global vocab
    vector = np.zeros(len(vocab))

    words = get_words_no_punc(inputfile)
    for word in words:
        if word in vocab.keys():
            vector[vocab.keys().index(word)] += 1

    return vector

def make_classifier():
    pos_files = get_file_names('pos')
    neg_files = get_file_names('neg')
    pos_labels = np.ones(len(pos_files))
    neg_labels = -np.ones(len(neg_files))
    y = np.concatenate((pos_labels, neg_labels), axis=0)
    m_pos = len(pos_files)
    m_neg = len(neg_files)
    m = m_pos + m_neg
    dimensions = len(vocab)

    X = np.zeros((m, dimensions))
    for i in xrange(m_pos):
        X[i, :] = vectorize(pos_files[i])
        print "Positive review processing {:.2%} percent done\r".format(
            i/float(m))
    for j in xrange(m_neg):
        X[j + m_pos, :] = vectorize(neg_files[j])
        print "Negative review processing{:.2%} percent done\r".format(
            (j + m_pos)/float(m))

    print "y is {}".format(y)
    print "X is {}".format(X)

    lr = LR()
    lr.fit(X,y)

    return lr

def test_classifier(lr):
    global vocab
    test = np.zeros((len(os.listdir('test')),len(vocab)))
    test_file_names = []
    i = 0
    y = []
    for file_name in os.listdir('test'):
        test_file_names.append(file_name)
        test[i] = vectorize(join('test', file_name))
        ind = int(file_name.split('_')[0][-1])
        y.append(1 if ind == 3 else -1)
        i += 1

    assert(sum(y) == 0)
    p = lr.predict(test)

    r, w = 0, 0
    for i, x in enumerate(p):
        if x == y[i]:
            r += 1
        else:
            w += 1
            print(test_file_names[i])
    print(r, w)


if __name__=='__main__':
    buildvocab(100)
    lr = make_classifier()
    test_classifier(lr)

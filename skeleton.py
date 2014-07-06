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
    # print word_list
    return word_list


# build vocabulary
def buildvocab():
    global vocab
    global stop_words
    build_stop_words()
    training_files = get_file_names('pos') + get_file_names('neg')
    word_count = {}
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

    vocab = OrderedDict(sorted(word_count.items(), key=lambda t: t[1]))
    # print vocab
    # print len(vocab)
    return None


def vectorize(fn):
    global vocab
    vector = np.zeros(len(vocab))

    ###TODO: Create vector representation of

    return vector

def make_classifier():

    #TODO: Build X matrix of vector representations of review files, and y vector of labels

    lr = LR()
    lr.fit(X,y)

    return lr

def test_classifier(lr):
    global vocab
    test = np.zeros((len(os.listdir('test')),len(vocab)))
    testfn = []
    i = 0
    y = []
    for fn in os.listdir('test'):
        testfn.append(fn)
        test[i] = vectorize(os.path.join('test',fn))
        ind = int(fn.split('_')[0][-1])
        y.append(1 if ind == 3 else -1)
        i += 1

    assert(sum(y)==0)
    p = lr.predict(test)

    r,w = 0,0
    for i,x in enumerate(p):
        if x == y[i]:
            r += 1
        else:
            w +=1
            print(testfn[i])
    print(r,w)


if __name__=='__main__':
    buildvocab()
    #lr = make_classifier()
    #test_classifier(lr)

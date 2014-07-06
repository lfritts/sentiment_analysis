import os
import sys
import numpy as np
from operator import itemgetter as ig
from sklearn.linear_model import LogisticRegression as LR
import operator

vocab = [] #the features used in the classifier

#build vocabulary
def buildvocab():
    global vocab
    neg_words = {}; pos_words = {}
    neg_text = ""; pos_text = ""
    neg_sorted = []; pos_sorted = []
    maps = {"neg_words": neg_words, "pos_words": pos_words,
            "neg_text": neg_text, "pos_text": pos_text,
            "neg_sorted": neg_sorted, "pos_sorted": pos_sorted}

    stopwords = open('stopwords.txt').read().lower().split()
    here = os.getcwd()
    vals = ["pos", "neg"]
    for val in vals:
        for afile in os.listdir(here + "/" + val):
            if afile.endswith(".txt"):
                maps[val + '_text'] = maps[val + '_text'] + \
                    open(here + '/' + val + '/' + afile, 'r').read()

        for word in maps[val + '_text'].split():
            if word in stopwords:
                pass
            elif word in maps[val + '_words']:
                maps[val + '_words'][word] += 1
            else:
                maps[val + '_words'][word] = 1

        maps[val + "_sorted"] = sorted(maps[val + '_words'].iteritems(), key=operator.itemgetter(1))
        maps[val + "_sorted"] = maps[val + "_sorted"][-100:]
        maps[val + "_sorted"].reverse()

        print val + ":" + str(maps[val + "_sorted"])



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
    # lr = make_classifier()
    # test_classifier(lr)

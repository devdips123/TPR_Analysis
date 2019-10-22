import KENTstemmer3
import importlib
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np


importlib.reload(KENTstemmer3)
dic = KENTstemmer3.readDictionary("tokenLemma.txt")


def apply_kent_stemmer(book):
    """
    params
        book -> type nltk.Text for e.g. nltk.book.text1
             -> type a list of list for e.g. [ word_tokenize('This is a test') ]
                                             [word_tokenize("i am a good boy."), word_tokenize('how are you')]

    returns
        stemmed_book -> a list of list containing stemmed words of the params

    """
    my_stemmed_book = []
    for sent in book:
        tags = nltk.pos_tag(sent)
        # print(tags)
        stems = []
        for tag in tags:
            stems.append(KENTstemmer3.KENTstemmer3(tag, dic))
        # print(stems)
        my_stemmed_book.append(stems)
    # print(my_stemmed_book)
    return my_stemmed_book


def nGramCount3(data):
    N = 3
    gramsC = {}  # auxilliary dictionary to store n-gram counts
    # seg = 1 sentence
    for seg in data:

        itm = seg.copy()

        # insert sentence starting and sentence ending symbols
        # !!! adjust starting symbol for different length of n-grams N
        itm.insert(0, "///")
        itm.insert(0, "///")
        itm.append("///")
        # print(itm)
        # p(a | b ) == count(ab) / count(b)
        for i in range(len(itm) - N + 1):
            # produce ngram (b)
            b = ' '.join(itm[i:i + N - 1]).lower()
            # produce an ngram (ab)
            ab = ' '.join(itm[i:i + N]).lower()
            # print("t:{}\tg:{}".format(t,g))
            gramsC.setdefault(b, {})
            gramsC[b].setdefault(ab, 0)
            # count the ngram
            gramsC[b][ab] += 1
    return (gramsC)

def nGramProbs3(gramsC) :
    nGrams = {}  # final n-gram dictionary with log-prob entries
    nMin = 0
    for b in gramsC:
        # v = number of n-grams that start with b
        v = float(sum(gramsC[b].values()))
        for ab in gramsC[b]:
            # np.log2(v/gramG[b][ab]) == - np.log2(gramG[b][ab]/v)
            nGrams[ab] = np.log2(v/gramsC[b][ab])
            if(nGrams[ab] > nMin): nMin = nGrams[ab]
            #print("ab:{:<20} count:{:<6}\tb:{:<10}  count:{}\tlog(1/p):{:<5.4}".format(ab, gramsC[b][ab], b, v,  nGrams[ab]))
    nGrams["|||OOV|||"] = nMin+1
    return(nGrams)


def perplexity3(data, nGrams):
    N = 3
    PP = []
    for seg in data:
        itm = seg.copy()

        # !!! adjust starting symbol for different length of n-grams N
        itm.insert(0, "///")
        itm.insert(0, "///")
        itm.append("///")

        H = 0
        for i in range(len(itm) - N):
            ab = ' '.join(itm[i:i + N]).lower()
            try:
                H += nGrams[ab]
                print("nGram: {:20}\t{:4.4}".format(ab, nGrams[ab]))
            except:
                print("nGram: {:20}\t{:4.4}\tundef".format(ab, nGrams["|||OOV|||"]))
                H += nGrams["|||OOV|||"]
        p = (2 ** H) ** (1 / float(len(seg)))
        PP.append(p)
        print("PP:{:5.2f}\tlen:{}\tseg:{}".format(p, len(seg), seg))
    return (PP)
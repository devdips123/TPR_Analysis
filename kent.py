import re
import numpy as np
import pandas as pd

##################################################
# kent.stemmer: read a tuple (word, PoS) and exception dictionary return stem
def Stemmer(t, dic) :  
        
    (wrd, pos) = t
    stem = wrd.lower()
    if(stem in dic) : return(dic[stem])
                 
    if(pos == "VB"):
        # updated by me
        stem = re.sub(r'(s|ed)$', '', stem)
    if(pos == "VBD"):
        # updated by me
        stem = re.sub(r'(tted)$', 't', stem)
        stem = re.sub(r'(ssed)$', 'ss', stem)
        #stem = re.sub(r'(lled)$', 'l', stem)
        stem = re.sub(r'(gged)$', 'g', stem)
        stem = re.sub(r'(pped)$', 'p', stem)
        stem = re.sub(r'(rred)$', 'r', stem)
        stem = re.sub(r'([v])(ed)$', r'\1e', stem)
        stem = re.sub(r'(ced)$', 'ce', stem)
        stem = re.sub(r'(sed)$', 'se', stem)
        stem = re.sub(r'(ied)$', 'y', stem)
        stem = re.sub(r'(ed)$', '', stem)

    elif(pos == "VBN"):
        # updated by me
        stem = re.sub(r'(ssed)$', 'ss', stem)
        stem = re.sub(r'(tted)$', 't', stem)
        #stem = re.sub(r'(lled)$', 'l', stem)
        stem = re.sub(r'(rred)$', 'r', stem)
        stem = re.sub(r'(gged)$', 'g', stem)
        stem = re.sub(r'(pped)$', 'p', stem)
        stem = re.sub(r'([v])(ed)$', r'\1e', stem)
        stem = re.sub(r'(ced)$', 'ce', stem)
        stem = re.sub(r'(sed)$', 'se', stem)
        stem = re.sub(r'(ied)$', 'y', stem)
        stem = re.sub(r'(ed)$', '', stem)
        
    elif(pos == "VBG"):
        # updated by me
        #stem = re.sub(r'([pndrmt])([pndrmt])(ing)$', r"\1", stem)
        stem = re.sub(r'(ating)$', r"ate", stem)
        stem = re.sub(r'(tting)$', r"t", stem)
        stem = re.sub(r'(pping)$', r"p", stem)
        stem = re.sub(r'(nning)$', r"n", stem)
        stem = re.sub(r'(dding)$', r"d", stem)
        stem = re.sub(r'(rring)$', r"r", stem)
        stem =re.sub(r'([vzcm])(ing)$',r'\1e',stem)
        #stem = re.sub(r'(ving)$', r"ve", stem)
        stem = re.sub(r'(ing)$', '', stem)             
        
    elif(pos == "VBZ"):
        # updated by me
        stem = re.sub(r'([rtgsc])(ies)$', r'\1y', stem)
        #stem = re.sub(r'(es)$', '', stem)
        stem = re.sub(r'(s)$', '', stem)
        
     
    elif(pos == "NN"):
        # updated by me
        stem = re.sub(r'(tting)$', r"t", stem)
        stem = re.sub(r'(pping)$', r"p", stem)
        stem = re.sub(r'(nning)$', r"n", stem)
        stem = re.sub(r'(dding)$', r"d", stem)
        stem = re.sub(r'(rring)$', r"r", stem)
        stem = re.sub(r'([vz])(ing)$', r'\1e', stem)
        #stem = re.sub(r'(ving)$', r"ve", stem)
        stem = re.sub(r'(iest)$', 'y', stem)
        stem = re.sub(r'(s|ment|ence|ation|er|ed)$', '', stem)
        stem = re.sub(r'(ing)$', '', stem)

    elif(pos == "NNS"):
        stem = re.sub(r'([^aeiou])(ies)$', r'\1y', stem)
        stem = re.sub(r'(s|ies)$', '', stem)
        
    elif(pos == "NNP"):
        stem = re.sub(r'(ed)$', '', stem)
        
    elif(pos =="JJ"):
        stem = re.sub(r'([rgtpndr])\1(ed|ing)$', r'\1', stem)
        #stem = re.sub(r'(rred)$', 'r', stem)
        #stem = re.sub(r'(gged)$', 'g', stem)
        #stem = re.sub(r'(tting)$', r"t", stem)
        #stem = re.sub(r'(pping)$', r"p", stem)
        #stem = re.sub(r'(nning)$', r"n", stem)
        #stem = re.sub(r'(dding)$', r"d", stem)
        #stem = re.sub(r'(rring)$', r"r", stem)
        stem = re.sub(r'([lkvz])(ing)$', r'\1e', stem)
        #stem = re.sub(r'([tv])(ed)$', r'\1e', stem)
        stem = re.sub(r'(ced)$', 'ce', stem)
        stem = re.sub(r'(ied)$', 'y', stem)
        stem = re.sub(r'(iest)$', 'y', stem)
        stem = re.sub(r'(est|ing|ly|s|ed|er)$', '', stem)

    elif(pos == "JJR"):
        stem = re.sub(r'(er)$', '', stem)               
        
    elif(pos == "JJS"):
        # updated by me
        stem = re.sub(r'(iest)$', 'y', stem)
        stem = re.sub(r'(est)$', '', stem)
        
    elif(pos == "RBR"):
        stem = re.sub(r'(er)$', '', stem)               
        
    elif(pos == "RBS"):
        stem = re.sub(r'(est)$', '', stem)                
        
    elif(pos == "RB"):
        stem = re.sub(r'(ly|ed)$', '', stem)    
        
    elif(pos == "FW"):
        stem = re.sub(r'(o)$', '', stem)                
    
    return(stem)

#read an exception dictionary
def readDictionary(dictionary) :
    Dic = {}
    with open(dictionary,"r", encoding="utf8") as file:
        for entry in file:
            lem, tok = re.findall("^(.*?)[\s]+(.*?)$", entry)[0]
            Dic[tok.lower()] = lem.lower()
    return (Dic)

#############################################
# compute n-grams dictionary for source text

#N = 2        # length of n-gram
# add conditional n-gram probabilities to dictionary
# data: list of sentences 
# gramsC: return dictionary with ngram counts
def nGramCount2(data, gramsC, N=2):

    for seg in data:
        
        itm = seg.copy()
            
        # insert sentence starting and sentence ending symbols
        for i in range(N-1): itm.insert(0, "///")
        itm.append("///")

        # p(a | b ) == count(ab) / count(b)  
        for i in range(len(itm)-N+1):
            # produce ngram (b)
            b  = ' '.join(itm[i:i+N-1]).lower()
            # produce an ngram (ab)
            ab = ' '.join(itm[i:i+N]).lower()
            # print("t:{}\tg:{}".format(t,g))
            gramsC.setdefault(b, {})
            gramsC[b].setdefault(ab, 0)
            # count the ngram
            gramsC[b][ab] += 1
    return(gramsC)


# compute probability of n-grams
# p(a | b ) == count(ab) / count(b) == count(g) / count(t)
# loop over all words
def nGramProbs(gramsC) :
    nGrams = {}  # final n-gram dictionary with log-prob entries
    nMin = 0
    for b in gramsC:
        # v = number of n-grams that start with b
        v = float(sum(gramsC[b].values()))
        for ab in gramsC[b]:
            # np.log2(v/gramG[b][ab]) == - np.log2(gramG[b][ab]/v)
            nGrams[ab] = np.log2(v/gramsC[b][ab])
            if(nGrams[ab] > nMin): nMin = nGrams[ab]
#            print("ab:{:<20} count:{:<6}\tb:{:<10}  count:{}\tlog(1/p):{:<5.4}".format(ab, gramsC[b][ab], b, v,  nGrams[ab]))
    nGrams["|||OOV|||"] = nMin+1
    return(nGrams)

# compute perplexity of segments
def perplexity(data, nGrams, N=2):
    PP = []
    for seg in data:  
        itm = seg.copy()
        
        # !!! adjust starting symbol for different length of n-grams N
        for i in range(N-1): itm.insert(0, "///")        
        itm.append("///")

        H = 0
        for i in range(len(itm)-N):
            ab = ' '.join(itm[i:i+N]).lower()
            try: 
                H += nGrams[ab]
                print("nGram: {:20}\t{:4.4}".format(ab, nGrams[ab]))
            except:
                print("nGram: {:20}\t{:4.4}\tundef".format(ab, nGrams["|||OOV|||"]))
                H += nGrams["|||OOV|||"]
        p = 2**(H / float(len(seg)))
        PP.append(p)
        print("PP:{:5.2f}\tlen:{}\tseg:{}".format(p, len(seg), seg))
    return(PP)

# return list probabilities list words (segment)
# N needs to be set correctly!
# set verbose=1 to print out the n-gram values
def ppWord(seg, nGrams, N=2, verbose=0):
    PP = []
    itm = seg.copy()
        
    # insert segment starting symbol for different length of n-grams N
    for i in range(N-1): itm.insert(0, "///")        
    itm.append("///")

    for i in range(len(itm)-N):
        ab = ' '.join(itm[i:i+N]).lower()
        try: 
            if(verbose) : print("nGram: {:10}\t{:4.4}".format(ab, nGrams[ab]))
            PP.append(float(nGrams[ab]))
        except:
            if(verbose) : print("**OOV: {:10}\t{:4.4}".format(ab, nGrams["|||OOV|||"]))
            PP.append(float(nGrams["|||OOV|||"]))
    return(PP)



##################################################
# Access to TPRDB
# read a set of table_type (extension) from a list of studies

import pandas as pd
import glob

def readTPDDBtables(studies, table_type, verbose=0, path="/data/critt/tprdb/TPRDB/"):
    df = pd.DataFrame()
    
    for study in studies:
        if(verbose) : print("Reading: " + study + table_type)
        for fn in glob.glob(path + study + table_type):
            if(verbose) : print("Reading: " + fn)
            df = pd.concat([df, pd.read_csv(fn, sep="\t", dtype=None)], ignore_index=True)
        
    return(df)

##################################################
# sentence segment and tokenize nltk books and 
# articles: list of tokenized articles,
# returns: list of tokenized sentences
from nltk.tokenize import sent_tokenize, word_tokenize 

def NLTKbooks2Sent(articles):
    data = []
    for article in articles: 
        for sent in sent_tokenize(' '.join(article)): 
            data.append(word_tokenize(sent))
    return(data)

##############################
# statistical analysis
from numpy import mean
from numpy import var
from math import sqrt
 
# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
	return (u1 - u2) / s

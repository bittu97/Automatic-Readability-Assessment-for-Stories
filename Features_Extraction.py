#word_count,avg_sentence_length,sentence_count,difficult_words,avg_syllables_per_word,poly_syllable_count,smog_index,gunning_fog,flesch_reading_ease,dale_chall_readability_score




import os
all_files = os.listdir("Ele-Txt/")
print(all_files)

files = {}
 
for filename in all_files:
    with open('Ele-Txt/'+filename, "r", encoding="utf8") as file:
        if filename in files:
            continue
        files[filename] = file.read()


import spacy 
from textstat.textstat import textstatistics, easy_word_set, legacy_round 


# ------------------------------------------------------------------


# Splits the text into sentences, using 
# Spacy's sentence segmentation which can 
# be found at https://spacy.io/usage/spacy-101 
def break_sentences(text): 
    nlp = spacy.load('en') 
    doc = nlp(text) 
    #print(doc)
    
    return list(doc.sents)


# ------------------------------------------------------------------
    


# Returns Number of Words in the text 
def word_count(text): 
    sentences = break_sentences(text) 
    words = 0
    for sentence in sentences: 
        words += len([token for token in sentence]) 
    return words 


# ------------------------------------------------------------------
    



# Returns the number of sentences in the text 
def sentence_count(text): 
    sentences = break_sentences(text) 
    
    return len(sentences) 


# ------------------------------------------------------------------
    



# Returns average sentence length 
def avg_sentence_length(text): 
    words = word_count(text) 
    sentences = sentence_count(text) 
    average_sentence_length = float(words / sentences) 
    return average_sentence_length 


# ------------------------------------------------------------------

# Textstat is a python package, to calculate statistics from 
# text to determine readability, 
# complexity and grade level of a particular corpus. 
# Package can be found at https://pypi.python.org/pypi/textstat 
def syllables_count(dummy): 
    
    return textstatistics().syllable_count(dummy) 


# ------------------------------------------------------------------



# Returns the average number of syllables per 
# word in the text 
def avg_syllables_per_word(text): 
    syllable = syllables_count(text) 
    words = word_count(text) 
    ASPW = float(syllable) / float(words) 
    return legacy_round(ASPW, 1) 


# ------------------------------------------------------------------
    


# Return total Difficult Words in a text 
def difficult_words(text): 

    # Find all words in the text 
   
    sentences = break_sentences(text) 
    words = [str(x) for x in sentences]
    # difficult words are those with syllables >= 2 
    # easy_word_set is provide by Textstat as 
    # a list of common words 
    diff_words_set = set() 
    
    for word in words: 
        syllable_counter = syllables_count(word) 
        if word not in easy_word_set and syllable_counter >= 2: 
            diff_words_set.add(word) 

    return len(diff_words_set) 


# ------------------------------------------------------------------
    



# A word is polysyllablic if it has more than 3 syllables 
# this functions returns the number of all such words 
# present in the text 
def poly_syllable_count(text): 
    count = 0  
    sentences = break_sentences(text) 
    words = [str(x) for x in sentences]
    
    for word in words:
        syllable_count = syllables_count(word) 
        if syllable_count >= 3: 
            count += 1

    return count 


# ------------------------------------------------------------------



def flesch_reading_ease(text): 
    """ 
        Implements Flesch Formula: 
        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW) 
        Here, 
        ASL = average sentence length (number of words 
                divided by number of sentences) 
        ASW = average word length in syllables (number of syllables 
                divided by number of words) 
    """
    FRE = 206.835 - float(1.015 * avg_sentence_length(text)) -float(84.6 * avg_syllables_per_word(text)) 
    return legacy_round(FRE, 2) 


# ------------------------------------------------------------------


def gunning_fog(text): 
    per_diff_words = (difficult_words(text) / word_count(text) * 100) + 5
    grade = 0.4 * (avg_sentence_length(text) + per_diff_words) 
    return grade 


# ------------------------------------------------------------------
    


def smog_index(text): 
    if sentence_count(text) >= 3: 
        #print(text)
        poly_syllab = poly_syllable_count(text) 
        SMOG = (1.043 * (30*(poly_syllab / sentence_count(text)))**0.5) + 3.1291
        return legacy_round(SMOG, 1) 
    else: 
        return 0
    
    
# ------------------------------------------------------------------
        
    

#diff_word = 0
def dale_chall_readability_score(text): 
    """ 
        Implements Dale Challe Formula: 
        Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365 
        Here, 
            PDW = Percentage of difficult words. 
            ASL = Average sentence length 
    """
    words = word_count(text) 
    # Number of words not termed as difficult words 
    #diff_word1 = difficult_words(text)
    
    count = word_count(text) - difficult_words(text)
    if words > 0: 

        # Percentage of words not on difficult word list 

        per = float(count) / float(words) * 100
    
    # diff_words stores percentage of difficult words 
    diff_words = 100 - per 

    raw_score = (0.1579 * diff_words) + (0.0496 * avg_sentence_length(text)) 
    
    # If Percentage of Difficult Words is greater than 5 %, then; 
    # Adjusted Score = Raw Score + 3.6365, 
    # otherwise Adjusted Score = Raw Score 

    if diff_words > 5:     

        raw_score += 3.6365
        
    return legacy_round(raw_score, 2) 


# ------------------------------------------------------------------


import collections as coll
import math
import pickle
import string

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize



# removing stop words plus punctuation.
def Avg_wordLength(str):
    str.translate(string.punctuation)
    tokens = word_tokenize(str, language='english')
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    stop = stopwords.words('english') + st
    words = [word for word in tokens if word not in stop]
    return np.average([len(word) for word in words])


# ------------------------------------------------------------------

# returns avg number of characters in a sentence
def Avg_SentLenghtByCh(text):
    tokens = sent_tokenize(text)
    return np.average([len(token) for token in tokens])


# ------------------------------------------------------------------

# returns avg number of words in a sentence
def Avg_SentLenghtByWord(text):
    tokens = sent_tokenize(text)
    return np.average([len(token.split()) for token in tokens])

# ------------------------------------------------------------------


# COUNTS SPECIAL CHARACTERS NORMALIZED OVER LENGTH OF CHUNK
def CountSpecialCharacter(text):
    st = ["#", "$", "%", "&", "(", ")", "*", "+", "-", "/", "<", "=", '>',
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    count = 0
    for i in text:
        if (i in st):
            count = count + 1
    return count / len(text)


# ------------------------------------------------------------------



def CountPuncuation(text):
    st = [",", ".", "'", "!", '"', ";", "?", ":", ";"]
    count = 0
    for i in text:
        if (i in st):
            count = count + 1
    return float(count) / float(len(text))


# ------------------------------------------------------------------


# RETURNS NORMALIZED COUNT OF FUNCTIONAL WORDS FROM A Framework for
# Authorship Identification of Online Messages: Writing-Style Features and Classification Techniques

def CountFunctionalWords(text):
    functional_words = """a between in nor some upon
    about both including nothing somebody us
    above but inside of someone used
    after by into off something via
    all can is on such we
    although cos it once than what
    am do its one that whatever
    among down latter onto the when
    an each less opposite their where
    and either like or them whether
    another enough little our these which
    any every lots outside they while
    anybody everybody many over this who
    anyone everyone me own those whoever
    anything everything more past though whom
    are few most per through whose
    around following much plenty till will
    as for must plus to with
    at from my regarding toward within
    be have near same towards without
    because he need several under worth
    before her neither she unless would
    behind him no should unlike yes
    below i nobody since until you
    beside if none so up your
    """

    functional_words = functional_words.split()
    words = RemoveSpecialCHs(text)
    count = 0

    for i in text:
        if i in functional_words:
            count += 1

    return count / len(words)


# ------------------------------------------------------------------


# c(w)  = ceil (log2 (f(w*)/f(w))) f(w*) frequency of most commonly used words f(w) frequency of word w
# measure of vocabulary richness and connected to zipfs law, f(w*) const rak kay zips law say rank nikal rahay hein
def AvgWordFrequencyClass(text):
    words = RemoveSpecialCHs(text)
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    maximum = float(max(list(freqs.values())))
    return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])


# ------------------------------------------------------------------


# TYPE TOKEN RATIO NO OF DIFFERENT WORDS / NO OF WORDS
def typeTokenRatio(text):
    words = word_tokenize(text)
    return len(set(words)) / len(words)


# ------------------------------------------------------------------


def RemoveSpecialCHs(text):
    text = word_tokenize(text)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

    words = [word for word in text if word not in st]
    return words


# ------------------------------------------------------------------


# logW = V-a/log(N)
# N = total words , V = vocabulary richness (unique words) ,  a=0.17
# we can convert into log because we are only comparing different texts
def BrunetsMeasureW(text):
    words = RemoveSpecialCHs(text)
    a = 0.17
    V = float(len(set(words)))
    N = len(words)
    B = (V - a) / (math.log(N))
    return B


# ------------------------------------------------------------------


# K  10,000 * (M - N) / N**2
# , where M  Sigma i**2 * Vi.
def YulesCharacteristicK(text):
    words = RemoveSpecialCHs(text)
    N = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    vi = coll.Counter()
    vi.update(freqs.values())
    M = sum([(value * value) * vi[value] for key, value in freqs.items()])
    K = 10000 * (M - N) / math.pow(N, 2)
    return K


# ------------------------------------------------------------------


# -1*sigma(pi*lnpi)
# Shannon and sympsons index are basically diversity indices for any community
def ShannonEntropy(text):
    words = RemoveSpecialCHs(text)
    lenght = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    arr = np.array(list(freqs.values()))
    distribution = 1. * arr
    distribution /= max(1, lenght)
    import scipy as sc
    H = sc.stats.entropy(distribution, base=2)
    # H = sum([(i/lenght)*math.log(i/lenght,math.e) for i in freqs.values()])
    return H


# ------------------------------------------------------------------


# 1 - (sigma(n(n - 1))/N(N-1)
# N is total number of words
# n is the number of each type of word
def SimpsonsIndex(text):
    words = RemoveSpecialCHs(text)
    freqs = coll.Counter()
    freqs.update(words)
    N = len(words)
    n = sum([1.0 * i * (i - 1) for i in freqs.values()])
    D = 1 - (n / (N * (N - 1)))
    return D


# ------------------------------------------------------------------



import csv
for i in files:
    # =============================================================================
     
     
     word_count1 = word_count(files[i])
     #print(word_count1)
     
     
     avg_sentence_length1 = avg_sentence_length(files[i])
     #print(avg_sentence_length1)
     
     
     sentence_count1 = sentence_count(files[i])
     #print(sentence_count1)
     
     
     difficult_words1 = difficult_words(files[i])
     #print(difficult_words1)
     
     
     avg_syllables_per_word1 = avg_syllables_per_word(files[i])
     #print(avg_syllables_per_word1)
     
     
     poly_syllable_count1 = poly_syllable_count(files[i])
     #print(poly_syllable_count1)
     
     
     
     Avg_wordLength1 = Avg_wordLength(files[i])
     #print(Avg_wordLength1)
     
     
     Avg_SentLenghtByCh1 = Avg_SentLenghtByCh(files[i])
     #print(Avg_SentLenghtByCh1)
     
     
     Avg_SentLenghtByWord1 = Avg_SentLenghtByWord(files[i])
     #print(Avg_SentLenghtByWord1)
     
     
     CountSpecialCharacter1 = CountSpecialCharacter(files[i])
     #print(CountSpecialCharacter1)
     
     
     CountPuncuation1 = CountPuncuation(files[i])
     #print(CountPuncuation1)
     
     
     CountFunctionalWords1 = CountFunctionalWords(files[i])
     #print(CountFunctionalWords1)
     
     
     AvgWordFrequencyClass1 = AvgWordFrequencyClass(files[i])
     #print(AvgWordFrequencyClass1)
     
     
     typeTokenRatio1 = typeTokenRatio(files[i])
     #print(typeTokenRatio1)
     
     
     BrunetsMeasureW1 = BrunetsMeasureW(files[i])
     #print(BrunetsMeasureW1)
     
     
     YulesCharacteristicK1 = YulesCharacteristicK(files[i])
     #print(YulesCharacteristicK1)
     
     
     ShannonEntropy1 = ShannonEntropy(files[i])
     #print(ShannonEntropy1)
     
     
     SimpsonsIndex1 = SimpsonsIndex(files[i])
     #print(SimpsonsIndex1)
     
     
     smog_index1 = smog_index(files[i])
     #print(smog_index1)
     
     
     gunning_fog1 = gunning_fog(files[i])
     #print(gunning_fog1)
     
     
     flesch_reading_ease1 = flesch_reading_ease(files[i])
     #print(flesch_reading_ease1)
     
     
     dale_chall_readability_score1 = dale_chall_readability_score(files[i])
     #print(dale_chall_readability_score1)
     
     
     #break
     
     
     
     with open('output.csv', 'a') as f:
         write = csv.writer(f)
         write.writerow([ word_count1, avg_sentence_length1, sentence_count1, difficult_words1, avg_syllables_per_word1, poly_syllable_count1,
                         Avg_wordLength1, Avg_SentLenghtByCh1, Avg_SentLenghtByWord1,
                         CountSpecialCharacter1, CountPuncuation1,
                         CountFunctionalWords1, AvgWordFrequencyClass1,
                         typeTokenRatio1, BrunetsMeasureW1,
                         YulesCharacteristicK1, ShannonEntropy1,
                         SimpsonsIndex1, smog_index1, gunning_fog1,
                    flesch_reading_ease1, dale_chall_readability_score1,0])
         print('File Created')
     
     
     #print(syllables_count(files[i]))
     
     
     
     #print(break_sentences(files[i]))
     #break
 

 
    
    
    
import pandas as pd

dataset = pd.read_csv('OneStopEnglishCorpus-master/allfeatures-ose-final.csv')


import pandas as pd
import numpy as np
import re
from string import punctuation
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import csv


with open('output.csv', 'a') as f:
    write = csv.writer(f)
    write.writerow([ word_count, avg_sentence_length, sentence_count, difficult_words, avg_syllables_per_word, poly_syllable_count, smog_index, gunning_fog,
                    flesch_reading_ease, dale_chall_readability_score])
    print('File Created')


import numpy as np
import nltk 
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(token):
    return stemmer.stem(token.lower().strip())


def bag_of_words(tokenized_sentence,words):
    bag = np.zeros(len(words),dtype=np.float32)
    sentence_words = [stem(word) for word in tokenized_sentence]

    for index,word in enumerate(words):
        if word in sentence_words:
            bag[index] = 1
    return bag
'''
Created on 19 Feb 2018

@author: Jack
'''
import os
import math
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import nltk
import string

# Initalise our global corpus variable
corpus = {}

# Sets up our corpus dictionary
def read_corpus():
    # Get list of file names in our data directory
    files = os.listdir("C:/Users/Jack/workspace/politifact_classifier/src/Politifact_Articles")
    
    # load all files into a dictionary, with filename as key and array of words as data
    diction = {}
    for y in files:
        vocab = []
        with open("Politifact_Articles/" + y, encoding='utf-8', errors='ignore') as f:
            vocab = f.read().splitlines()
        diction[y] = vocab
    
    return diction

# Function to evaluate bm25 score / weight for corpus with input word
def bm25(word):
    # Number of documents containing the word
    nt = 0
    # The dictionary holding term frequencies
    tf_dict = {}
    # The dictionary holding the weights
    weights = {}
    
    for name, words in corpus.items():

        # This way we only create new entry if word appears
        if words.count(word) > 0:
            tf_dict[name] = words.count(word)
        
        # Add to number of documents containing the word if necessary
        if name in tf_dict:
            nt += 1
    
    # If the term appeared at least once perform idf
    if nt > 0:
        idf = math.log10(len(corpus) / nt)
    
    # Evaluate scores
    for name, tf in tf_dict.items():
        weights[name] = idf * tf #((tf / 1) + tf)
        
    return weights
    

# Set up corpus dictionary
corpus = read_corpus()

# Set up punctuation we want removed
punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

# Read in tweets from text document
with open("harambesmall.txt", encoding='utf-8', errors='ignore') as f:
    tweets_string= f.read().splitlines()

for tweet_string in tweets_string:
    # Remove commas cause some reason wasn't working 
    tweet_string = tweet_string.replace(",", "")
    
    # Tokenize the string
    words = nltk.word_tokenize(tweet_string)
    
    # Make it lower case and remove all punctuation
    tweet=[word.lower() for word in words if word not in punctuations]
    
    # Remove stopwords, tweet_token is what we want to save
    stopWords = set(stopwords.words('english'))
    tweet_token = []
    for w in tweet:
        if w not in stopWords:
            tweet_token.append(w)
    
    # Get weights
    weights = {}
    for i in tweet_token:
        temp = bm25(i)
        for name, tf in temp.items():
            if name in weights:
                weights[name] = weights[name] + tf
            else:
                weights[name] = tf
    
    # get max and second max score and names to print out
    maxi = 0
    maxi_2 = 0
    second = ""
    top = ""
    for name, weight in weights.items():
        if weights[name] > maxi:
            maxi_2 = maxi
            second = top
            maxi = weights[name]
            top = name
            
    # Print top and second name and score
    print("Name: " + top + " Score: " + str(maxi)) 
    print("Name: " + second + " Score: " + str(maxi_2))
    print()
    

            
            
            
            
            
            
            
            
            
            
            
            
            
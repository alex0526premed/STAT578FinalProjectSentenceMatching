###STAT 578 Final Project
###Quora Question Pairs
###Data cleaning, preprocessing, exploration and visualization
###Author: Yan Liu

############################################################################
#Change working directory
import os
os.chdir("/Users/victoria_DFB/Desktop/Spring2017/STAT 578/FinalProject")

#temp storage
import pickle
#pickle.dump(a, open('tmp.txt','wb'))
question1 = pickle.load(open('question1.txt','rb')) #'ab'
question2 = pickle.load(open('question2.txt','rb'))
glove_vec1 = pickle.load(open('glove_vec1.txt','rb'))
glove_vec2 = pickle.load(open('glove_vec2.txt','rb'))
glove_vec1.pop(0)
glove_vec2.pop(0)

#Read the data and put all the questions in a list
import csv

question = list()
question1 = list()
question2 = list()
is_pair = list()
f = open('train.csv','r',encoding = 'utf-8')
reader = csv.reader(f)
#Split each sentence to words
#Convert to lower case 
for line in reader:
	question.append(line)
	question1.append(line[3].lower().split())
	question2.append(line[4].lower().split())
	is_pair.append(line[5])

#question[0] is the head of the table
#For the i-th item in list 'question':
#question[i][0], question[i][1], question[i][2]: id
#question[i][3], question[i][4]: pair of questions
#question[i][5]: whether the pair of questions are the same?
n = len(question)-1 #number of pairs
is_pair.pop(0)
is_pair = [float(i) for i in is_pair]
pickle.dump(is_pair, open("is_pair",'wb'))

############################################################################
#Data Cleaning
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from string import punctuation

#Function to complete data cleaning
###Input:
#text: a list of words
#ifstem: logical, whether to transform words to their stems
#ifrmstop: logical, whether to remove stopwords
#Output:
#text: a list of words without punctuations and stemmed
def text_clean(text, ifrmstop, ifstem):
	#Remove stopwords:
	if ifrmstop:
		st = set(stopwords.words("english")) #Remove repeated stopwords
		text = [w for w in text if not w in st]

	#Remove punctuations
	text = " ".join(text)
	text = "".join([w for w in text if w not in punctuation])
	text = text.split()

	#Transform words to their stems
	if ifstem:
		stemmer = SnowballStemmer("english")
		stemwords = [stemmer.stem(w) for w in text]
		text = " ".join(stemwords)
		text = text.split()

	return(text)

for i in range(n):
	question1[i+1] = text_clean(question1[i+1], False, False)
	question2[i+1] = text_clean(question2[i+1], False, False)

############################################################################
###Data exploration and visualization
from matplotlib import pyplot
import pandas
import numpy as np
import pylab

#Length of the questions
length1 = list()
length2 = list()
for i in range(n):
        length1.append(len(question1[i]))
        length2.append(len(question2[i]))

pickle.dump(length1, open("length1",'wb'))
pickle.dump(length2, open("length2",'wb'))

mean1 = np.mean(length1)
mean2 = np.mean(length2)

#Histogram of question lengths
pylab.hist(length1,bins = 10,normed = 1)
pylab.show()
pylab.hist(length2,bins = 10,normed = 1)
pylab.show()

#Number of common words in the question pairs
num_common_word = list()
for ind in range(n):
	num_common_word.append(len(set(question1[ind]).intersection(set(question2[ind]))))
pickle.dump(num_common_word, open("num_common_word",'wb'))

############################################################################
#t-SNE
import re
import nltk
from gensim.models import word2vec
from sklearn.manifold import TSNE
from nltk.corpus import wordnet as wn

############################################################################
#Spacy
import spacy
#Load model data
nlp = spacy.load("en")

#Lemmatize/Stemming with Spacy
#Input: a list of words
def lemmatize(text):
	text = " ".join(text)
	doc = nlp(text)
	for w in doc:
		w = w.lemma_
	text = str(doc)
	text = text.split()
	return(text)

for i in range(n):
	question1[i+1] = lemmatize(question1[i+1])
	question2[i+1] = lemmatize(question2[i+1])
#Save the data cleaning results
pickle.dump(question1, open('question1.txt','wb'))
pickle.dump(question2, open('question2.txt','wb'))

#POS Tagging
def pos_tag(text):
	pos = list()
	text = " ".join(text)
	doc = nlp(text)
	for w in doc:
		pos.append(w.pos_)
	return(pos)

#What is the distribution of POS?
poslist = list()
for q in question1:
	poslist.extend(pos_tag(q))
for q in question2:
	poslist.extend(pos_tag(q))
#Plot a histogram
x = range(len(set(poslist)))
y = list()
for pos in set(poslist):
	y.append(poslist.count(pos))
pyplot.bar(x,y)
pyplot.xticks(x,list(set(poslist)),rotation = 'vertical')
pyplot.show()


############################################################################
#Obtain corresponding vectors calculated by Glove
#from gensim.utils import tokenize
#Take the mean vector of all the words in the sentence
glove_vec1 = list()
glove_vec2 = list()
for q in question1:
	text = " ".join(q)
	doc = nlp(text)
	glove_vec1.append(doc.vector)

for q in question2:
	text = " ".join(q)
	doc = nlp(text)
	glove_vec2.append(doc.vector)

#SVD Decomposition
import numpy as np
x = glove_vec1[1:404291]
x = np.array(x)
U,D,V = np.linalg.svd(x)
D = np.diag(D)


############################################################################
###TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

question_all = question1
question_all.extend(question2)
word_all = list()
word_all = sum(question_all,[])
tfidf = TfidfVectorizer()

#Put the results in a dictionary
tfidf.fit_transform(question_all)
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

del question_all





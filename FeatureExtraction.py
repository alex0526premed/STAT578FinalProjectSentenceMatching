###STAT 578 Final Project
###Quora Question Pairs
###Similarity Features Extraction and Combination
###Author: Yan Liu

from scipy import spatial
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# from sklearn.metrics import jaccard_similarity_score

#Cosine
cos_sim = list()
manhattan = list()
minkowski = list()
canberra = list()
for ind in range(n):
	a = glove_vec1[ind]
	b = glove_vec2[ind]
	cos_sim.append(1 - spatial.distance.cosine(a,b))
#Save cosine similarity
pickle.dump(cos_sim, open("cos_sim",'wb'))

#Manhattan distance
def manhattan_distance(start, end):
    return sum(abs(e - s) for s,e in zip(start, end))

for ind in range(n):
	a = glove_vec1[ind]
	b = glove_vec2[ind]
	manhattan.append(manhattan_distance(a,b))
#Save Manhattan distance
pickle.dump(manhattan, open("manhattan",'wb'))

#Jaccard similarity score: not applicable for glove vectors
jaccard_similarity_score(a,b, normalize = False)
#Euclidean
glove_vec_np1 = np.asarray(glove_vec1)
glove_vec_np2 = np.asarray(glove_vec2)
glove_diff = glove_vec_np1-glove_vec_np2
pickle.dump(glove_diff, open("glove_diff",'wb'))

#Minkowski distance
for ind in range(n):
	a = glove_vec1[ind]
	b = glove_vec2[ind]
	minkowski.append(spatial.distance.minkowski(a,b,3))
#Save Manhattan distance
pickle.dump(minkowski, open("minkowski",'wb'))

#Canberra distance
for ind in range(n):
	a = glove_vec1[ind]
	b = glove_vec2[ind]
	canberra.append(spatial.distance.canberra(a,b))
#Save Manhattan distance
pickle.dump(canberra, open("canberra",'wb'))

#Levenshtein distance
from fuzzywuzzy import fuzz
fuzzq = list()
fuzzw = list()
fuzz_partial = list()
fuzz_set = list()
fuzz_sort = list()
fuzz_partial_set = list()
fuzz_partial_sort = list()
for ind in range(n):
	a = question[ind][0]
	b = question[ind][1]
	fuzzq.append(fuzz.QRatio(a,b))
	fuzzw.append(fuzz.WRatio(a,b))
	fuzz_partial.append(fuzz.partial_ratio(a,b))
	fuzz_set.append(fuzz.token_set_ratio(a,b))
	fuzz_sort.append(fuzz.token_sort_ratio(a,b))
	fuzz_partial_set.append(fuzz.partial_token_set_ratio(a,b))
	fuzz_partial_sort.append(fuzz.partial_token_sort_ratio(a,b))

pickle.dump(fuzzq, open("fuzzq",'wb'))
pickle.dump(fuzzw, open("fuzzw",'wb'))
pickle.dump(fuzz_partial, open("fuzz_partial",'wb'))
pickle.dump(fuzz_set, open("fuzz_set",'wb'))
pickle.dump(fuzz_sort, open("fuzz_sort",'wb'))
pickle.dump(fuzz_partial_set, open("fuzz_partial_set",'wb'))
pickle.dump(fuzz_partial_sort, open("fuzz_partial_sort",'wb'))

#######################################################################################
#Combine all the features
import os
os.chdir("/Users/victoria_DFB/Desktop/Spring2017/STAT 578/FinalProject/Features")


#Naive features
length1 = pickle.load(open('length1','rb'))
length2 = pickle.load(open('length2','rb'))
num_common_word = pickle.load(open('num_common_word','rb'))
fuzzq = pickle.load(open('fuzzq','rb'))
fuzzw = pickle.load(open('fuzzw','rb'))
fuzz_set = pickle.load(open('fuzz_set','rb'))
fuzz_sort = pickle.load(open('fuzz_sort','rb'))
fuzz_partial = pickle.load(open('fuzz_partial','rb'))
fuzz_partial_set = pickle.load(open('fuzz_partial_set','rb'))
fuzz_partial_sort = pickle.load(open('fuzz_partial_sort','rb'))

#Short sentence emantic similarity
sim = pickle.load(open('sim','rb'))

#Features from Glove embedding
glove_diff = pickle.load(open('glove_diff','rb'))
cos_sim = pickle.load(open('cos_sim','rb'))
manhattan = pickle.load(open('manhattan','rb'))
minkowski = pickle.load(open('minkowski','rb'))
canberra = pickle.load(open('canberra','rb'))

#Combine all the features
#Need to change when comparing different feature sets
feature = np.column_stack((length1,length2,num_common_word,
	fuzzq,fuzzw,fuzz_set,fuzz_sort,fuzz_partial,fuzz_partial_set,fuzz_partial_sort,
	sim,
	glove_diff,cos_sim,manhattan,minkowski,canberra)) #requires array input
pickle.dump(feature, open("feature",'wb'))

#Manually features + GloVe
# feature = np.column_stack((length1,length2,num_common_word,
# 	fuzzq,fuzzw,fuzz_set,fuzz_sort,fuzz_partial,fuzz_partial_set,fuzz_partial_sort,
# 	glove_diff,cos_sim,manhattan,minkowski,canberra))

#semantic similarity score + GloVe
#feature = np.column_stack((sim, glove_diff,cos_sim,manhattan,minkowski,canberra))

#Only GloVe Embedding
#feature = np.column_stack((glove_diff,cos_sim,manhattan,minkowski,canberra))



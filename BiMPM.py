###STAT 578
###BiMPM Preparation
###Author: Yan Liu
###Note: The original package was written by Z. Wang.
###See: https://github.com/zhiguowang/BiMPM

###############################################################################
import os
import csv
import numpy as np
import pickle
import sklearn.feature_extraction.text as text
from sklearn.cross_validation import train_test_split

# Work directory
os.chdir("/Users/victoria_DFB/Desktop/Spring2017/STAT 578/FinalProject")

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

question.pop(0)
n = len(question)

for i in range(n):
	question[i][0] = question[i][5]
	question[i][1] = question[i][3]
	question[i][2] = question[i][4]
	question[i].pop(5)
	question[i].pop(4)
	question[i].pop(3)

# Load Glove word embedding
glove_vec1 = pickle.load(open('glove_vec1.txt','rb'))
glove_vec2 = pickle.load(open('glove_vec2.txt','rb'))
glove_vec1.pop(0)
glove_vec2.pop(0)

q_train,q_test = train_test_split(question, test_size = 0.3, random_state = 0)
q_train,q_dev = train_test_split(q_train, test_size = 2/7, random_state = 0)

# Write the file into working directory
with open('q_train.txt','w') as file:
	file.writelines('\t'.join(i) + '\n' for i in q_train)

with open('q_dev.txt','w') as file:
	file.writelines('\t'.join(i) + '\n' for i in q_dev)

with open('q_test.txt','w') as file:
	file.writelines('\t'.join(i) + '\n' for i in q_test)


##########################################################################################
# The following lines should be run in terminal
# Need to input the right working directory
# Training
python ./src/BiMPM-master/src/SentenceMatchTrainer.py --train_path q_train.txt --dev_path q_dev.txt test_path q_test.txt --word_vec_path wordvec.txt --suffix sample --fix_word_vec --model_dir models/ --MP_dim 20
# Testing
python ./src/BiMPM-master/src/SentenceMatchDecoder.py --in_path q_test.txt word_vec_path wordvec.txt --mode prediction --model_prefix models/SentenceMatch.sample --out_path test.prediction





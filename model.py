###STAT 578 Final Project
###Quora Question Pairs
###Model Training
###Author: Yan Liu

#Working directory
import os
os.chdir("/Users/victoria_DFB/Desktop/Spring2017/STAT 578/FinalProject/Features")
feature = pickle.load(open('feature','rb'))
is_pair = pickle.load(open('is_pair','rb'))

import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import confusion_matrix, mean_squared_error
#######################################################################################
is_pair_np = np.asarray(is_pair)

#Split train data and test data
Xtrain, Xtest, ytrain, ytest = train_test_split(feature, is_pair_np, test_size = 0.3, random_state = 0)
dtrain = xgb.DMatrix(Xtrain, label = ytrain)
dtest = xgb.DMatrix(Xtest, label = ytest)


############################################################################
#XGBoost
le = LabelEncoder()
#Load the numpy array into DMatrix
# glove_vec_np1 = np.asarray(glove_vec1)
# glove_vec_np2 = np.asarray(glove_vec2)
# is_pair_np = np.asarray(is_pair)
# glove_diff = glove_vec_np1-glove_vec_np2

#Cross-validation
num_round = 2
print('running cross-validation')
#Parameters
param = {'booster': 'gbtree',
         'objective':'binary:logistic',
         'subsample': 0.85,
         'colsample_bytree': 0.95,
         'eta':0.01,
         'max_depth':5, 
         'silent':0,
         'eval_metric':'rmse'}

# model = xgb.XGBClassifier()
# model.fit(Xtrain,ytrain)
# print(model)


# clf = grid_search.GridSearchCV(estimator = model, param_grid = dict())
# xgb_cv_result = xgb.cv(param, dtrain, num_round, nfold = 5, metrics = {'error'}, 
#     callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

xgbfit = xgb.train(param, dtrain)
xgbfit.save_model('xgb1.model') #Save the model

#Prediction
is_pair_pred = xgbfit.predict(dtest)
is_pair_pred_01 = is_pair_pred
for ind in range(len(is_pair_pred)):
    if is_pair_pred[ind]>0.5:
        is_pair_pred_01[ind] = 1
    else:
        is_pair_pred_01[ind] = 0

err_rate = sum(abs(np.asarray(is_pair_pred_01)-np.asarray(ytest)))/len(is_pair_pred)

############################################################################
train_is_pair_ind = [ind for ind in range(len(ytrain)) if ytrain[ind]==1]
train_sim_is_pair = [sim[ind] for ind in train_is_pair_ind]
train_not_pair_ind = [ind for ind in range(len(ytrain)) if ytrain[ind]==0]
train_sim_not_pair = [sim[ind] for ind in train_not_pair_ind]
import pylab
train_sim_is_pair.sort()
train_sim_not_pair.sort()
pylab.plot(train_sim_is_pair)
pylab.show()
pylab.plot(train_sim_not_pair)
pylab.show()





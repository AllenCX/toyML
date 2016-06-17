
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV 
from sklearn import svm
#%matplotlib inline
#%load_ext autoreload
#%autoreload 2


# In[ ]:

data = pd.read_csv('data.csv')


# In[ ]:

#to caculate the sum of different action type such as jump shot or running jump shot
total_actions = dict(data.action_type.value_counts())

threshold = 10
data['type'] = data.apply(lambda row: row['action_type'] if total_actions[row['action_type']] >= threshold                           else row['combined_shot_type'], axis = 1)

data['time_remaining'] = data.apply(lambda row: row['minutes_remaining']*60 + row['seconds_remaining'], axis = 1)

# TODO: tune this parameter
threshold = 3
# TODO: find out why he cant hit @ 14 secs to go
anomaly = 14

data['last moment'] = data.apply(lambda row: row['time_remaining'] <= threshold or row['time_remaining'] == anomaly, 
                                 axis = 1)
data['shot_distance'] = data.apply(lambda row: 28 if row['shot_distance']>28 else row['shot_distance'], axis=1)


# In[ ]:

def get_acc(data, field):
    ct = pd.crosstab(data.shot_made_flag, data[field]).apply(lambda x: x / x.sum(), axis=0)
    x, y = ct.columns, ct.values[1, :]
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel(field)
    plt.ylabel('% shots made')
    plt.show()


# In[ ]:

def sort_encode(data, field):
    ct = pd.crosstab(data.shot_made_flag, data[field]).apply(lambda x: x / x.sum(), axis=0)
    temp = list(zip(ct.values[1, :], ct.columns))
    temp.sort()
    new_map = {}
    for index, (acc, old_number) in enumerate(temp):
        new_map[old_number] = index
    new_field = field + '_sort_enumerated'
    data[new_field] = data[field].map(new_map)
    get_acc(data, new_field)


# In[ ]:

'''
data.drop([
        'action_type',
        'combined_shot_type',
        'game_event_id',
        'game_id',
        #'lat',
        #'lon',
        'minutes_remaining',
        'seconds_remaining',
        'time_remaining',
        'team_id',
        'team_name',
        'matchup',
        'game_date',
        'shot_type',
        'playoffs',
        'season',
        # TODO: find out whether these two features matter or not
        #'loc_x',
        #'loc_y',
    ], axis=1, inplace=True)'''

data.drop([
        #'lat',
        #'lon',
        'minutes_remaining',
        'seconds_remaining',
        'game_id',
        'game_event_id',
        'game_date',#TODO: check if it's needed
        #'time_remaining',
        # TODO: find out whether these two features matter or not
        #'loc_x',
        #'loc_y',
    ], axis=1, inplace=True)

dummies = [
    'period',
    'type',
    'shot_zone_area',
    'shot_zone_basic',
    'shot_zone_range',
    'opponent',
    'team_id',
    'team_name',
    'matchup',
    #'game_date',
    'shot_type',
    'playoffs',
    'season',
    'action_type',
    'combined_shot_type',
 #   'game_event_id',
]
dummie_counter = {}
for dummy in dummies:
    dummie_counter[dummy] = len(data[dummy].unique())
data = pd.get_dummies(data, columns=dummies)


# In[27]:

dummie_counter


# In[ ]:

# split into train and test
train = data[~data['shot_made_flag'].isnull()]
test = data[data['shot_made_flag'].isnull()]
print('train size: ' + str(len(train)))
print('test size:  ' + str(len(test)))

# prepare data for estimators
target = 'shot_made_flag'
features = data.columns.tolist()
features.remove(target)
features.remove('shot_id')
X_test = test[features]
X_train = train[features]
y_train = train[[target]]['shot_made_flag'].values
print(X_train.shape)
print(y_train.shape)


# In[ ]:

def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds, metrics=['logloss'])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    # Test params
    X_train, X_valid, y_train, y_valid = train_test_split(dtrain[predictors], dtrain[target], test_size=0.2)
    alg.fit(X_train, y_train)
    y_pred = alg.predict_proba(X_valid)[:,1]
    
    result = log_loss(y_valid, y_pred)
    print(result)
    
    return result


# In[ ]:

clf = svm.SVC(probability=True)
clf.fit(X_train, y_train)

predictors = [x for x in X_train.columns if x not in target]
result = modelfit(clf,train,predictors,useTrainCV=False, cv_folds=5, early_stopping_rounds=50)

s_result = pd.Series(result, index = ['svm_loss_result'])
s_result.to_csv("predict_result_svm.csv")

test.shot_made_flag = [i[1] for i in clf.predict_proba(X_test)]
test[['shot_id', 'shot_made_flag']].to_csv('sub_svm.csv', index=False)
predictions_train = clf.predict_proba(X_train)
features_train = pd.DataFrame({'shot_made_flag': predictions_train[:, 1]})
features_train[['shot_made_flag']].to_csv('features_train_svm.csv', index=False)

# In[ ]:

# fit, predict and generate submission file
'''params = {
    'base_score': 0.5, 
    'colsample_bylevel': 1,
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'max_depth': 7,
    'min_child_weight': 1,
    'n_estimators': 200,
    'nthread': -1,
    'objective': 'binary:logistic',
    'seed': 2,
    'subsample': 0.8
}

clf = xgb.XGBClassifier()
clf.set_params(**params)


predictors = [x for x in X_train.columns if x not in target]
modelfit(clf,train,predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50)

param_test1 = [{
 'max_depth':[3,5,7,9],
 'min_child_weight':[3,4,5],
    'learning_rate':[0.05,0.1,0.15,0.2]
}]
gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
                        param_grid = param_test1, scoring='log_loss',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

'''


# In[ ]:




import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold


# LOAD
X_train = pd.read_csv('flight_delays_train.csv')
X_test = pd.read_csv('flight_delays_test.csv')

y_train = X_train.iloc[:, -1].map({'Y': 1, 'N': 0})
X_train = X_train.iloc[:, :-1]



# CATEGORIAL PREPROCESS
train_month = X_train['Month'].astype('category')
test_month = X_test['Month'].astype('category', categories=list(train_month.cat.categories))
train_month = pd.get_dummies(train_month)
test_month = pd.get_dummies(test_month)

train_dayofmonth = X_train['DayofMonth'].astype('category')
test_dayofmonth = X_test['DayofMonth'].astype('category', categories=list(train_dayofmonth.cat.categories))
train_dayofmonth = pd.get_dummies(train_dayofmonth)
test_dayofmonth = pd.get_dummies(test_dayofmonth)

train_dayofweek = X_train['DayOfWeek'].astype('category')
test_dayofweek = X_test['DayOfWeek'].astype('category', categories=list(train_dayofweek.cat.categories))
train_dayofweek = pd.get_dummies(train_dayofweek)
test_dayofweek = pd.get_dummies(test_dayofweek)

train_uniquecarrier = X_train['UniqueCarrier'].astype('category')
test_uniquecarrier = X_test['UniqueCarrier'].astype('category', categories=list(train_uniquecarrier.cat.categories))
train_uniquecarrier = pd.get_dummies(train_uniquecarrier)
test_uniquecarrier = pd.get_dummies(test_uniquecarrier)

train_origin = X_train['Origin'].astype('category')
test_origin = X_test['Origin'].astype('category', categories=list(train_origin.cat.categories))
train_origin = pd.get_dummies(train_origin)
test_origin = pd.get_dummies(test_origin)

train_dest = X_train['Dest'].astype('category')
test_dest = X_test['Dest'].astype('category', categories=list(train_dest.cat.categories))
train_dest = pd.get_dummies(train_dest)
test_dest = pd.get_dummies(test_dest)

train_route = (X_train['Dest'] + ' - ' + X_train['Origin']).astype('category')
test_route = (X_test['Dest'] + ' - ' + X_test['Origin']).astype('category', categories=list(train_route.cat.categories))
train_route = pd.get_dummies(train_dest)
test_route = pd.get_dummies(test_dest)

X_train_preproc = np.hstack((train_month, train_dayofmonth, train_dayofweek, X_train['DepTime'].reshape(-1, 1), train_uniquecarrier, train_origin, train_dest, X_train['Distance'].reshape(-1, 1),train_route))
X_test_preproc = np.hstack((test_month, test_dayofmonth, test_dayofweek, X_test['DepTime'].reshape(-1, 1), test_uniquecarrier, test_origin, test_dest, X_test['Distance'].reshape(-1, 1),test_route))

#print('oversampling')
#from imblearn.over_sampling import SMOTE
#smote = SMOTE(kind='svm', random_state=17)
#X_train_preproc, y_train = smote.fit_sample(X_train_preproc, y_train)
#from sklearn.utils import shuffle
#X_train_preproc, y_train = shuffle(X_train_preproc, y_train, random_state=17)









# CROSS VALIDATION AND HOLDOUT
cv_holdout_splitter = int(0.8 * X_train_preproc.shape[0])

X_train_preproc_cv = X_train_preproc[:cv_holdout_splitter, :]
y_train_cv = y_train[:cv_holdout_splitter]

X_train_preproc_holdout = X_train_preproc[cv_holdout_splitter:, :]
y_train_holdout = y_train[cv_holdout_splitter:]

print('start fitting')
gbm = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)


cross_val_score = np.mean(cross_val_score(gbm, X_train_preproc_cv, y_train_cv, cv=skf, scoring='roc_auc'))

gbm = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary')
gbm.fit(X_train_preproc_cv, y_train_cv)
holdout_predictions = gbm.predict_proba(X_train_preproc_holdout)[:, 1]
holdout_score = roc_auc_score(y_train_holdout, holdout_predictions)

print(cross_val_score, holdout_score)



 #SUBMISSION
gbm = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary')
gbm.fit(X_train_preproc, y_train)

test_predictions = gbm.predict_proba(X_test_preproc)[:, 1]
pd.Series(test_predictions, name='dep_delayed_15min').to_csv('gbm2.csv', index_label='id', header=True)  #0.7306


from catboost import CatBoostClassifier
# Initialize data
X_train['route'] = X_train.Origin + ' - ' + X_train.Dest
X_test['route'] = X_test.Origin + ' - ' + X_test.Dest
cat_features = np.where(X_train.dtypes == 'object')[0].tolist()
train_data = X_train.values[:cv_holdout_splitter, :]
train_labels = y_train.values[:cv_holdout_splitter]
test_data = X_train.values[cv_holdout_splitter:]
# Initialize CatBoostClassifier
model = CatBoostClassifier(random_seed=17)
# Fit model
model.fit(train_data, train_labels, cat_features)
# Get predicted classes
preds_class = model.predict(test_data)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(test_data)
# Get predicted RawFormulaVal
preds_raw = model.predict(test_data, prediction_type='RawFormulaVal')

holdout_score = roc_auc_score(y_train.values[cv_holdout_splitter:], preds_proba[:, 1])


#submissiom

model.fit(X_train, y_train, cat_features)
test_predictions2 = model.predict_proba(X_test)[:, 1]
pd.Series(test_predictions2*0.4+test_predictions*0.6, name='dep_delayed_15min').to_csv('union.csv', index_label='id', header=True)  #0.7306















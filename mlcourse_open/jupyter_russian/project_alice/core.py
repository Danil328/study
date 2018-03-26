import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy import io

from tools import write_to_submission_file
from preprocessing import base_preprocessing
from preprocessing import benchmark_1_preprocessing, benchmark_2_preprocessing, benchmark_3_preprocessing
from preprocessing import variant_1_preprocessing
from preprocessing import variant_2_preprocessing
from preprocessing import variant_3_preprocessing
from preprocessing import variant_4_preprocessing
from preprocessing import variant_5_preprocessing
from preprocessing import variant_6_preprocessing
from preprocessing import variant_7_preprocessing



if __name__=='__main__':
    train = pd.read_csv('data/train_sessions.csv', index_col='session_id')
    test = pd.read_csv('data/test_sessions.csv', index_col='session_id')

    X_train, X_test, y_train = base_preprocessing(train, test)
    #X_train, X_test = variant_7_preprocessing(X_train, X_test, y_train)

    X_train, X_test = io.mmread('X_train.mtx'), io.mmread('X_test.mtx')

    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=17)

    X_train, y_train = ros.fit_sample(X_train, y_train)






    # from evolutionary_search import EvolutionaryAlgorithmSearchCV
    # from sklearn.model_selection import ShuffleSplit
    #
    # lr = LogisticRegression(n_jobs=-1)
    # params = {'penalty': ['l2'],
    #           'C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1] + list(range(2, 100, 5)) + [1000],
    #           'class_weight': ['balanced'],
    #           'solver': ['newton-cg', 'lbfgs', 'sag']}
    # cv = ShuffleSplit(test_size=0.30, n_splits=1)
    #
    # evo = EvolutionaryAlgorithmSearchCV(estimator=lr,
    #                                     params=params,
    #                                     scoring='roc_auc',
    #                                     cv=cv,
    #                                     verbose=True,
    #                                     population_size=100,
    #                                     gene_mutation_prob=0.10,
    #                                     gene_crossover_prob=0.5,
    #                                     tournament_size=5,
    #                                     generations_number=10)
    # evo.fit(X_train, y_train)

    #{'penalty': 'l2', 'max_iter': 100, 'solver': 'sag', 'class_weight': 'balanced', 'C': 37} with fitness: 0.9947851116234038




    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

    # #HOLD OUT
    # train_len = int(0.9 * X_train.shape[0])
    # X_for_train = X_train[:train_len, :]
    # X_for_valid = X_train[train_len:, :]
    # y_for_train = y_train[:train_len]
    # y_for_valid = y_train[train_len:]
    #
    # logit = LogisticRegression(n_jobs=-1, random_state=17)
    # logit.fit(X_for_train, y_for_train)
    #
    # valid_pred = logit.predict_proba(X_for_valid)[:, 1]
    #
    # print(roc_auc_score(y_for_valid, valid_pred))






    # # VALIDATION CURVES
    # for i in [10, 20, 50, 100, 200, 500, 700, 1000, 1500, 2000]:
    #     X_train1, X_test1 = variant_5_preprocessing(X_train, X_test, y_train, i)
    #     train_len = int(0.9 * X_train1.shape[0])
    #     X_for_train = X_train1[:train_len, :]
    #     X_for_valid = X_train1[train_len:, :]
    #     y_for_train = y_train[:train_len]
    #     y_for_valid = y_train[train_len:]
    #
    #     logit = LogisticRegression(n_jobs=-1, random_state=17)
    #     logit.fit(X_for_train, y_for_train)
    #
    #     valid_pred = logit.predict_proba(X_for_valid)[:, 1]
    #
    #     print('param={0}, roc_auc={1}'.format(i, roc_auc_score(y_for_valid, valid_pred)))



    # # CROSS-VALIDATION
    # log_regressor = LogisticRegression(random_state=17)
    # skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
    #
    # cross_val_scores = np.mean(cross_val_score(log_regressor, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1))



    #SUBMIT
    from sklearn.linear_model import LogisticRegression
    logit = LogisticRegression(n_jobs=-1)
    logit.fit(X_train, y_train)

    predictions = logit.predict_proba(X_test)[:, 1]

    write_to_submission_file(predictions, 'preprocessing_7.csv')








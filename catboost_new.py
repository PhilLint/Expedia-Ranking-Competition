from catboost import CatBoost, Pool, CatBoostClassifier, CatBoostRegressor
from copy import deepcopy
import matplotlib as plt
import numpy as np
import os
from sklearn.utils import shuffle
import pandas as pd
from feature_engineering import *
from scoring import *
from model_test import *
from catboost.datasets import msrank
from collections import Counter


def fit_model(loss_function, train_pool, test_pool, default_parameters, additional_params=None):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoost(parameters)
    model.fit(train_pool, eval_set=test_pool, plot=False)

    return model


# model = fit_model(loss_function='RMSE')

def get_info_datasets(X_train, queries_train, y_train):
    num_documents = X_train.shape[0]
    num_var = X_train.shape[1]
    num_queries = np.unique(queries_train).shape[0]
    num_target = Counter(y_train).items()
    return [num_documents, num_var, num_queries, num_target]


def train_test(training, n_estimators, max_rank, target='score', save_model=False, sample=False, size=0.5):
    # get target value
    extract_train_features(training, target=target)
    # for testing purposes use smaller dataset
    if sample:
        data = get_sample(training, size=size)
    else:
        data = training

    # split entire dataset
    train, test = split_train_test(data, split=4)
    # oversample training part
    train, _, _, _ = oversample(data=train, max_rank=max_rank)
    # cat boost needs both sorted
    train = train.sort_values('srch_id', ascending=True)
    test = test.sort_values('srch_id', ascending=True)

    X_train = train.drop(columns=["target", "booking_bool", "click_bool", "position", "random_bool", 'srch_id'])
    y_train = train["target"]
    queries_train = train['srch_id'].values

    X_test = test.drop(columns=["target", "booking_bool", "click_bool", "position", "random_bool", 'srch_id'])
    y_test = test["target"]
    y_test_our_eval = test.loc[:, ["srch_id", "prop_id", "booking_bool", "click_bool"]]
    queries_test = test['srch_id'].values

    train_list = get_info_datasets(X_train, queries_train, y_train)
    # test_list = get_info_datasets(X_test, queries_test, y_test)

    print(f"Training data:  number: {train_list[0]}, target info: {train_list[3]}, number queries : {train_list[2]}, number features: {train_list[1]}")
    # print(f"Test data:  number: {test_list[0]}, target info: {test_list[3]}, number queries : {test_list[2]}, number features: {test_list[1]}")

    train_pool = Pool(
        data=X_train,
        label=y_train,
        group_id=queries_train
    )

    test_pool = Pool(
        data=X_test,
        label=y_test,
        group_id=queries_test
    )

    default_parameters = {
        'iterations': n_estimators,
        'verbose': 100,
        'random_seed': 42,
        'eval_metric': 'NDCG:top=5',
        'use_best_model': True
    }

    # fit model
    print(f"Fitting model...")
    if target == 'score' or target == 'score_rank':
        model = fit_model(loss_function='RMSE', train_pool=train_pool, test_pool=test_pool,
                          default_parameters=default_parameters)
    elif target == 'book' or target == 'book_clck':
        model = fit_model(loss_function='Logloss', train_pool=train_pool, test_pool=test_pool,
                          default_parameters=default_parameters)
    print("Done")

    predictions = model.predict(X_test)
    cat_score = model.best_score_
    score = score_prediction(predictions, y_test_our_eval, to_print=False)

    return [score, cat_score]


if __name__ == "__main__":
    # constants
    pd.options.mode.chained_assignment = None
    targets = ["score", 'score_rank', 'book', 'book_click']
    n_estimators = [1000]
    max_ranks = [None, 10, 5]
    k_folds = 1
    scores = []
    # data = test_norm
    # regression: top10 features, 300 trees, max_rank = 10, target=score, prediction: ~ 0.31
    # regression: top10 features, 300 trees, max_rank = 10, target=score_rank, prediction: ~ .312
    # classification: all features, 300 trees, max_rank = 10, target=score, prediction: ~.349
    # training = pd.read_csv(str('./data/') + 'training_norm_data.csv', low_memory=False)
    # test = pd.read_csv(str('./data/') + 'test_norm_data.csv', low_memory=False)

    # data = pd.read_csv()
    # data = impute_na(data)
    # data = get_sample(data=data, size=0.01)

    for target in targets:
        for max_rank in max_ranks:
            print(f"\nCURRENT CONFIGURATION")
            print("########################################################################")
            print(f"Target = {target}")
            print(f"N_trees = {n_estimators}")
            print(f"Max_rank = {max_rank}")
            print("########################################################################")

            scores.append(train_test(training=training,
                       max_rank=max_rank,
                       target=target,
                       n_estimators=n_estimators,
                        save_model = False))

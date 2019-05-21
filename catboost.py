from catboost import CatBoost, Pool
from copy import deepcopy
import matplotlib as plt
import numpy as np
import os
from sklearn.utils import shuffle
import pandas as pd
from feature_engineering import *
from collections import Counter

extract_train_features(training, target="score")
data = training.copy()

training = training.sort_values('srch_id', ascending=True)

X_train = training.drop(['target', 'srch_id'], axis=1).values
y_train = training['target'].values
queries_train = training['srch_id'].values

X_test = test_df.drop([0, 1], axis=1).values
y_test = test_df[0].values
queries_test = test_df[1].values

num_documents = X_train.shape[0]
X_train.shape[1]

from collections import Counter
Counter(y_train).items()

max_relevance = np.max(y_train)
y_train /= max_relevance
y_test /= max_relevance

num_queries = np.unique(queries_train).shape[0]
num_queries

train = Pool(
    data=X_train,
    label=y_train,
    group_id=queries_train
)

test = Pool(
    data=X_test,
    label=y_test,
    group_id=queries_test
)

"""
data_dir = os.path.join('..', 'msrank')
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

description_file = os.path.join(data_dir, 'dataset.cd')
with open(description_file, 'w') as f:
    f.write('0\tLabel\n')
    f.write('1\tQueryId\n')

Pool(data=train_file, column_description=description_file, delimiter=',')
"""

default_parameters = {
    'iterations': 2000,
    'custom_metric': ['NDCG:top=5'],
    'verbose': True,
    'random_seed': 42,
}

parameters = {}

def get_sample(data, size):
    srch_ids = shuffle(data["srch_id"].value_counts().index.tolist())
    bound = int(len(srch_ids) * size)
    sample_ids = srch_ids[:bound]
    sample = data.loc[data["srch_id"].isin(sample_ids)]
    return sample

def fit_model(loss_function, train_pool, test_pool, additional_params=None):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoost(parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)

    return model

# model = fit_model(loss_function='RMSE')

def get_info_datasets(X_train, queries_train, y_train):
    num_documents = X_train.shape[0]
    num_var = X_train.shape[1]
    num_queries = np.unique(queries_train).shape[0]
    num_target = Counter(y_train).items()
    return [num_documents, num_var, num_queries, num_target]

def train_test_submit(estimator, training, max_rank, pred_weight, save_model=False, sample=False):

    if sample:
        data = get_sample(training, size=0.1)
    else:
        data = training

    train, test = split_train_test(data, split=4)
    train, _, _, _ = oversample(data=train, max_rank=max_rank)

    train = train.sort_values('srch_id', ascending=True)
    test = test.sort_values('srch_id', ascending=True)

    X_train = train.drop(columns=["target", "booking_bool", "click_bool", "position", "random_bool", 'srch_id'])
    y_train = train["target"]
    queries_train = train['srch_id'].values

    X_test = test.drop(columns=["target", "booking_bool", "click_bool", "position", "random_bool", 'srch_id'])
    y_test = test["target"]
    queries_test = test['srch_id'].values

    train_list = get_info_datasets(X_train, queries_train, y_train)
    #test_list = get_info_datasets(X_test, queries_test, y_test)

    print(f"Training data:  number: {train_list[0]}, target info: {train_list[3]}, number queries : {train_list[2]}, number features: {train_list[1]}")
    #print(f"Test data:  number: {test_list[0]}, target info: {test_list[3]}, number queries : {test_list[2]}, number features: {test_list[1]}")

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
        'iterations': 200,
        'custom_metric': ['NDCG:top=5'],
        'verbose': True,
        'random_seed': 42,
    }
    # fit model
    print(f"Fitting model...")
    model = fit_model(loss_function='RMSE', train_pool=train_pool, test_pool=test_pool)
    print("Done")

    if save_model:
        dump(estimator, "model3.joblib")
    print("Loading test data...")
    test_data = pd.read_csv("C:/Users/Frede/Dropbox/Master/DM/Assignments/2/DM2/test_norm_data.csv")
    print("Done")
    print("Predicting...")
    test_data = impute_na(test_data)
    predict_array = estimator.predict_proba(test_data)
    predict_array[:, 2] = predict_array[:, 2] * pred_weight
    prediction = predict_array[:, [1, 2]].sum(axis=1)
    print("Formatting to submission...")
    submission = prediction_to_submission(prediction, test_data)
    submission.to_csv("sub2.csv", index=False)
    print("Done")


if __name__ == "__main__":
    # constants
    pd.options.mode.chained_assignment = None
    targets = ["score"]
    n_estimators = [300]
    max_ranks = [10]
    type_est = "classifier"
    pred_weight = 3
    k_folds = 1
    top10_feat = ['prop_location_score1', 'prop_location_score2',
       'orig_destination_distance', 'price_usd', 'srch_average_loc1',
       'srch_diff_price', 'srch_diff_locscore1', 'srch_diff_locscore2',
       'srch_diff_prop_review_score', 'norm_srch_diff_locscore2']

    # data = test_norm
    # regression: top10 features, 300 trees, max_rank = 10, target=score, prediction: ~ 0.31
    # regression: top10 features, 300 trees, max_rank = 10, target=score_rank, prediction: ~ .312
    # classification: all features, 300 trees, max_rank = 10, target=score, prediction: ~.349
    training = pd.read_csv(str('./data/') + 'training_norm_data.csv', low_memory=False)

    data = pd.read_csv()
    data = impute_na(data)
    #data = get_sample(data=data, size=0.01)

    for target in targets:
        for n_estimator in n_estimators:
            for max_rank in max_ranks:

                print(f"\nCURRENT CONFIGURATION")
                print("########################################################################")
                print(f"Target = {target}")
                print(f"N_trees = {n_estimator}")
                print(f"Max_rank = {max_rank}")
                print(f"Type estimator: {type_est}")
                if type_est == "classifier":
                    print(f"Prediction weight: {pred_weight}")
                print("########################################################################")

                extract_train_features(data=data, target=target, max_rank=max_rank)
                if type_est == "classifier":
                    estimator = RandomForestClassifier(n_estimators=n_estimator, n_jobs=-1)
                elif type_est == "regression":
                    estimator = RandomForestRegressor(n_estimators=n_estimator, n_jobs=-1)
                else:
                    print("Invalid estimator specified!")

                train_test_submit(estimator=estimator,
                                  train=data,
                                  max_rank=max_rank,
                                  pred_weight=pred_weight,
                                  save_model=True)





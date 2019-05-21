from catboost import CatBoost, Pool
from copy import deepcopy
import matplotlib as plt
import numpy as np
import os
import pandas as pd


from catboost.datasets import msrank
train_df, test_df = msrank()

X_train = train_df.drop([0, 1], axis=1).values
y_train = train_df[0].values
queries_train = train_df[1].values

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


default_parameters = {
    'iterations': 2000,
    'custom_metric': ['NDCG', 'PFound', 'AverageGain:top=5'],
    'verbose': False,
    'random_seed': 0,
}

parameters = {}

def fit_model(loss_function, additional_params=None, train_pool=train, test_pool=test):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoost(parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)

    return model

model = fit_model(loss_function='RMSE')


def create_weights(queries):
    query_set = np.unique(queries)
    query_weights = np.random.uniform(size=query_set.shape[0])
    weights = np.zeros(shape=queries.shape)

    for i, query_id in enumerate(query_set):
        weights[queries == query_id] = query_weights[i]

    return weights


train_with_weights = Pool(
    data=X_train,
    label=y_train,
    group_weight=create_weights(queries_train),
    group_id=queries_train
)

test_with_weights = Pool(
    data=X_test,
    label=y_test,
    group_weight=create_weights(queries_test),
    group_id=queries_test
)

fit_model(
    'RMSE',
    additional_params={'train_dir': 'RMSE_weigths'},
    train_pool=train_with_weights,
    test_pool=test_with_weights
)

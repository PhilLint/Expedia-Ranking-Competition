import pandas as pd
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from math import log, floor
import seaborn
rnd.seed(42)

training = pd.read_csv('./data/training_set_VU_DM.csv', nrows = 100000)
test = pd.read_csv('./data/test_set_VU_DM.csv', nrows = 100000)

plt.figure(figsize=(12,8))
ax = seaborn.countplot(x="random_bool", data=training)
plt.title('')
plt.xlabel('Number of Axes')
plt.show()

def filter_nothing_instances(data, nothing_ids=None, max_rank=None):
    """
    For neither clicked nor booked either choose randomly a smaller number of instances, or
    only take the x-th first ranked (based on position) nothing instances to decrease size of dataset.
    :param data:
    :param max_rank: at which rank do we stop to consider nothing instances
    :return: smaller dataset
    """
    # not booked and not clicked and position smaller than max_rank
    if max_rank not None:
        data = data[data['booking_bool'] == 0][data['click_bool'] == 0][data['position']<max_rank]
    # if random choice: choose 0.3 of nothing instances
    else:
        n_sample = np.random.choice(nothing_ids.tolist(), floor(0.3*len(nothing_ids)), False)
        data = data.loc[n_sample]

    return data


def oversample(data, max_rank=None):
    """
    Mainly unbooked and unclicked results, therefore to save computational cost and
    weigh relevant instances higher, neither clicked nor booked instances deleted.
    :param data: panda array of training data
    :return: smaller panda array based on training data
    """
    # getting the indices of the instance
    # booked instances
    book_ids = data[data['booking_bool']==1].index.values
    # only clicked (not booked) instances
    click_ids = data[data['booking_bool']==0][data['click_bool']==1].index.values
    # neither clicked nor booked
    nothing_ids = data[data['booking_bool']==0][data['click_bool']==0].index.values
    # number booked
    book_cutoff = book_ids.size
    # number clicked
    number_clicks = click_ids.size
    # filter nothing instances down either randomly or only keep nothing instances
    # of position smaller than max_rank
    data = filter_nothing_instances(data, nothing_ids=nothing_ids, max_rank=max_rank)
    return data, book_cutoff,


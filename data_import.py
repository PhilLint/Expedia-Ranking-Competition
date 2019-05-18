import pandas as pd
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from math import log, floor
import seaborn
rnd.seed(42)

def import_data(filename, nrows=None):
    """
    import either train or test set (onl nrows, if None then all)
    :param path: to csv file
    :param nrows: how many instances should be looked at
    :return: od dataframe
    """
    if nrows is None:
        data = pd.read_csv(str('./data/') + filename)
    else:
        data = pd.read_csv(str('./data/') + filename, nrows=nrows)
    # remove time stamp of query
    data = data.loc[:, data.columns != "date_time"]

    return data


def filter_nothing_instances(data, nothing_ids=None, max_rank=None):
    """
    For neither clicked nor booked either choose randomly a smaller number of instances, or
    only take the x-th first ranked (based on position) nothing instances to decrease size of dataset.
    :param data:
    :param max_rank: at which rank do we stop to consider nothing instances
    :return: smaller dataset
    """
    # not booked and not clicked and position smaller than max_rank
    if max_rank is not None:
        nothing_ids = data[(data['booking_bool'] == 0) & (data['click_bool'] == 0) & (data['position'] < max_rank)].index.values
    # if random choice: choose 0.3 of nothing instances
    else:
        n_sample = np.random.choice(nothing_ids.tolist(), floor(0.3*len(nothing_ids)), False)
        nothing_ids = data.loc[n_sample].index.values

    return nothing_ids


def get_id_list(data, max_rank=None):
    """

    :param data:
    :return:
    """
    # getting the indices of the instance
    # booked instances
    book_ids = data[data['booking_bool'] == 1].index.values
    # only clicked (not booked) instances
    click_ids = data[(data['booking_bool'] == 0) & (data['click_bool'] == 1)].index.values
    # neither clicked nor booked
    nothing_ids = data[(data['booking_bool'] == 0) & (data['click_bool'] == 0)].index.values
    # number booked
    number_book = book_ids.size
    # number clicked
    number_clicks = click_ids.size

    # only keep books and clicks
    # filter nothing instances down either randomly or only keep nothing instances
    # of position smaller than max_rank
    filtered_nothing_ids = filter_nothing_instances(data, nothing_ids=nothing_ids, max_rank=max_rank)

    id_list = [book_ids, click_ids, filtered_nothing_ids]

    return id_list, number_book, number_clicks



def oversample(data, max_rank=None, print_desc=False):
    """
    Mainly unbooked and unclicked results, therefore to save computational cost and
    weigh relevant instances higher, neither clicked nor booked instances deleted.
    :param data: panda array of training data
    :return: smaller panda array based on training data
    """
    id_list, number_book, number_clicks = get_id_list(data, max_rank=max_rank)
    book_ids, click_ids, filtered_nothing_ids = id_list
    # filter out the dataset
    data = pd.concat([data[data.index.isin(book_ids)], data[data.index.isin(click_ids)],  data[data.index.isin(filtered_nothing_ids)]])
    # print descriptives
    if print_desc:
        print("Number of observations: " + str(len(new_training)) + " ||  number of bookings: " + str(number_books) +
              " ||  number of clicks: " + str(number_clicks))
    return data, number_book, number_clicks, id_list


if __name__ == "__main__":
    training = import_data('training_set_VU_DM.csv', nrows=100000)
    test = import_data('test_set_VU_DM.csv', nrows=100000)

    plt.figure(figsize=(12, 8))
    ax = seaborn.countplot(x="random_bool", data=training)
    plt.title('')
    plt.xlabel('Number of Axes')
    plt.show()

    new_training, number_books, number_clicks, id_list = oversample(training)
    # plot position to frequency booked (1) or not booked (0)
    new_training['position'].hist(by=new_training['booking_bool'])
    plt.show()
    # now with max_rank
    new_training, number_books, number_clicks, id_list = oversample(training, 15)
    new_training['position'].hist(by=new_training['booking_bool'])
    plt.show()

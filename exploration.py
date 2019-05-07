import pandas as pd
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from math import log, floor
import seaborn
from data_import import import_data
from feature_engineering import *

training = import_data('training_set_VU_DM.csv', nrows = 100000)
test = import_data('test_set_VU_DM.csv', nrows = 100000)

def ranking_plot(df):
    """
    PLots for expedia and random ranking the position and booking / clicking
    :param data: pd dataframe
    :return: plot so no return
    """
    # new variable book_clicK. 0 is nothing, 1 only clicked, 2 is booked
    df['book_click'] = df['booking_bool'] + df['click_bool']
    # filter nothing out
    df = df.loc[df['book_click'] > 0, :]
    # change to string for plot naming
    df.loc[df['book_click'] == 1, 'book_click'] = 'click'
    df.loc[df['book_click'] == 2, 'book_click'] = 'booking'
    # change to string for plot title
    df.loc[df['random_bool'] == 0, 'random_bool'] = 'expedia'
    df.loc[df['random_bool'] == 1, 'random_bool'] = 'random'
    # change for legend title
    df['ranking'] = df['random_bool']
    g = sns.catplot(x="position", hue="book_click", row="ranking",
                    data=df, kind="count",
                    height=4, aspect=2, legend=False)
    # change legend
    g.fig.get_axes()[0].legend(title='Type of action', loc='upper right')
    plt.show()

# demo
ranking_plot(training)

# Justify log transformation of price per search_id
def price_plot(data, upper=True):
    clip_outliers(data, 'price_usd', upper_quantile=upper)
    data['price_usd'].hist(bins=100)
    plt.title('Price frequencies')
    plt.xlabel('Price in $')
    plt.ylabel('Count')
    plt.show()

#demo
price_plot(training, upper=False)
price_plot(training)




import pandas as pd
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from math import log, floor
import seaborn as sns
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




def scatters(data, feature_1, feature_2):

    """
    check for correlation between two features - prints one plot
    :param feature_1: name of feature one (str)
    :param feature_2: name of feature two (str)
    :return: none
    """

    print(f"Checking for correlations between feature: {feature_1} and feature: {feature_2}...\n")

    mask_1 = data[feature_1].notna()
    mask_2 = data[feature_2].notna()
    subset_1 = data[feature_1].loc[mask_1].loc[mask_2]
    subset_2 = data[feature_2].loc[mask_1].loc[mask_2]
    print(subset_1)
    print(subset_2)

    cor = stats.pearsonr(subset_1, subset_2 )
    print(f"Correlation {feature_1},{feature_2}: {cor}")

    ax=sns.regplot(subset_1, subset_2, marker='o', color='blue', scatter_kws={'s': 2})
    ax.set_ylim([0,2000])
    ax.set_xlim([0,2000])
    plt.show()

def save_corr_mat(data):

    """
    calculate correlation matrix of all features
    saves to corr_mat.csv
    :param data: pandas df
    :return: none
    """
    corr_mat = data.corr()
    corr_mat.to_csv("corr_mat.csv")


def outlier_plot(data, features=[], to_save=False, name=None):
    """
    save/show boxplot for numeric features
    :param data: pandas df
    :param features: list of features to be plotted, if empty all relavant numeric features will be plotted
    :param to_save: save plot, if False plot is shown
    :param name: name of plot to be saved
    :return:
    """

    # clear old plots
    plt.clf()

    if not features:
        cols_exlude = ["srch_id", "date_time", "site_id", "prop_id", "visitor_location_country_id", "prop_country_id",
                       "comp1_rate", "comp2_rate", "comp3_rate", "comp4_rate", "comp5_rate", "comp6_rate", "comp7_rate", "comp8_rate",
                       "comp1_inv", "comp2_inv", "comp3_inv", "comp4_inv", "comp5_inv","comp6_inv","comp7_inv", "comp8_inv"]
        for col in data.columns:
            if data[col].dtype == "float64" and col not in cols_exlude:
                features.append(col)

    num_df = data[features]
    sns.set(style="ticks")
    ax = sns.boxplot(y="variable", x="value", orient="h", data=pd.melt(num_df))
    ax.set_xscale("log")
    fig = plt.gcf()
    fig.set_size_inches(9,6)
    ax.set_position([.25, .15, .70, .75])
    if to_save:
        plt.savefig(name)
    else:
        plt.show()


def competition_plot(data, to_save=False, name=None):
    """
    combine all competitor rate information into one feature
    plot click and bool proportions depending on competition
    :param data: pandas df
    :param to_save: bool
    :param name: name of plot if to_save=True
    :return: none
    """

    competition = data[["comp1_rate", "comp2_rate", "comp3_rate", "comp4_rate", "comp5_rate", "comp6_rate", "comp7_rate", "comp8_rate"]]
    user_beh = data[["booking_bool", "click_bool"]]
    user_beh["comp_rate"] = [1 if row > 0 else row for row in competition.sum(axis=1)]
    user_beh["comp_rate"] = [-1 if row < 0 else user_beh["comp_rate"].loc[idx] for idx,row in enumerate(competition.sum(axis=1))]

    melted = pd.melt(user_beh, id_vars="comp_rate", value_vars=["click_bool", "booking_bool"])

    plt.clf()
    ax = sns.barplot(x="comp_rate", y="value", hue="variable", data=melted, palette="deep")
    ax.set(xticklabels=["More expensive", "Same Price", "Cheaper"])
    ax.set(ylabel="Proportion clicked/booked")
    ax.set(xlabel=[])
    if to_save:
        plt.savefig(name)
    else:
        plt.show()



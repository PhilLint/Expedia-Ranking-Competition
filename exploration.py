import pandas as pd
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from math import log, floor
import seaborn as sns


training = pd.read_csv('./data/training_set_VU_DM.csv', nrows = 10000)
test = pd.read_csv('./data/test_set_VU_DM.csv', nrows = 10000)


# How many values are missing for per feature?
missing=[]
for i in training:
    missing.append({'field':i,'percent':len(training[training[i].isnull()])/training.shape[0]})

miss=pd.DataFrame(sorted(missing, key=lambda k: k['percent']))
plt.rcParams["figure.figsize"]=[30,10]
ax=miss.plot.bar(x='field',y='percent',legend=None)
vals = ax.get_yticks()
ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

plt.figure(figsize=(12,8))
ax = sns.countplot(x="booking_bool", data=training)
plt.title('')
plt.xlabel('Hotel booked (0 = no/1 = yes')
plt.show()

# Describe the starrating history of the customers
print(training["visitor_hist_starrating"].describe())
training["visitor_hist_starrating"].hist()
plt.title("Distribution of visitor_hist_starrating")

sns.distplot(training["visitor_hist_starrating"])


# fraction of missing values for starrating history
null_ratings = pd.isnull(training["visitor_hist_starrating"]).sum()
number_of_ratings = training.shape[0]
print(null_ratings/(1.0*number_of_ratings))
# 94.6 percent missing data

# Does the starrating have an influence on the booking behavior?

training['visitor_hist_starrating'].hist(by=training['booking_bool'])
plt.title("Star rating distribution by booking (1/0)", horizontalalignment ='center')

# Describe the mean price per night that customer previously purchased
print(training["visitor_hist_adr_usd"].describe())
training["visitor_hist_adr_usd"].hist()
plt.title("Distribution of visitor_hist_adr_usd")

# frequencies of boolean features
plt.rcParams["figure.figsize"]=[15,15]
fig, axes = plt.subplots(nrows=4, ncols=2)

training.promotion_flag.value_counts().plot.pie(ax=axes[0,0])
training.srch_room_count.value_counts().plot.pie(ax=axes[0,1])
training.srch_adults_count.value_counts().plot.pie(ax=axes[1,0])
training.srch_children_count.value_counts().plot.pie(ax=axes[1,1])
training.srch_saturday_night_bool.value_counts().plot.pie(ax=axes[2,0])
training.booking_bool.value_counts().plot.pie(ax=axes[2,1])
training.random_bool.value_counts().plot.pie(ax=axes[3,0])
training.click_bool.value_counts().plot.pie(ax=axes[3,1])

# Plot sale price promotion bool
plt.figure(figsize=(12,8))
ax = sns.countplot(x="promotion_flag", data=training)
plt.title('Did offered hotels have a sale price promotion?')
plt.xlabel('Sale Price Promotion (0 = no/1 = yes)')
plt.show()

training['promotion_flag'].bar(by=training['booking_bool'])
plt.title("Sale Price Promotion by booking (1/0)", horizontalalignment ='center')


# Barplots: x-axis prop_brand_bool (+1 hotel belongs to chain, 0 independent hotel),
#           y-axis = number of bookings/clicks

# Scatterplot: prop_location_score1/2 for desirability of hotel’s location, y-axis

# Overview of score values
training["prop_location_score1"].head()
training["prop_location_score1"].head()

# barplot: x-axis = prop_location_score1/2/prop_review_score, y-axis = number of bookings for that block

# Are there strong correlations in the data set?

# scatterplot: x-axis = prop_location_score1, y-axis = prop_location_score2
sns.lmplot('prop_location_score1', # Horizontal axis
           'prop_location_score2', # Vertical axis
           data= training) # Data source


# Set title
plt.title('Correlation of location desirability scores')
# Set x-axis label
plt.xlabel('Location desirability: score 1')
# Set y-axis label
plt.ylabel('Location desirability: score 2')
plt.show()

# slight positive correlation, still both scores shall be kept as features
import seaborn

from data_import import import_data
from feature_engineering import *



training = import_data('training_set_VU_DM.csv', nrows = 100000)
test = import_data('test_set_VU_DM.csv', nrows = 100000)

def ranking_plot(df):
    """
    Plots for expedia and random ranking the position and booking / clicking
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



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

# scatterplot: x-axis = prop_review_score, y-axis = number of bookings
#

# Barplots: x-axis prop_brand_bool (+1 hotel belongs to chain, 0 independent hotel),
#           y-axis = number of bookings/clicks

# Scatterplot: prop_location_score1/2 for desirability of hotel’s location, y-axis

# Overview of score values
training["prop_location_score1"].head()
training["prop_location_score1"].head()

# barplot: x-axis = prop_location_score1/2, y-axis = number of bookings for that block

# Are there strong correllations in the dataset?
#sns.pairplot(training)

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




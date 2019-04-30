import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = pd.read_csv("training_set_VU_DM.csv", nrows=100000)
for col in data.columns:
    print(col)

################### NA HANDLING ###########################


def check_na(feature):
    """
    check correlation of feature with click, bool, position
    :param feature: (str) column name of pandas df
    :return: none
    """

    # PROPORTION NAN
    mask = data[feature].notna()
    len(data[feature].loc[mask]) / len(data[feature])


    # CORRLEATE WITH CLICK, BOOK, POSITION
    # >>>>>>>>>>>>>>>>>>>> CHECK TYPE OF FEATURE ?? or not
    #if data[feature].dtype == "float64"
    book_cor = stats.pointbiserialr(data[feature].loc[mask], data.booking_bool[mask])
    click_cor = stats.pointbiserialr(data[feature].loc[mask], data.click_bool[mask])
    pos_cor = stats.pearsonr(data[feature].loc[mask], data.position[mask])

    print("Correlation with booking_bool: ", book_cor)
    print("Correlation with click_bool: ", click_cor)
    print("Correlation with position: ", pos_cor)


    fig, axs = plt.subplots(1, 3)
    sns.boxplot(data.click_bool[mask], data[feature].loc[mask], ax=axs[0])
    axs[0].set_title('click_bool')
    sns.boxplot(data.booking_bool[mask], data[feature].loc[mask], ax=axs[1])
    axs[1].set_title('booking_bool')
    ### SCATTER
    #sns.scatterplot(data.position[mask], data[feature].loc[mask], s=5, ax=axs[2])
    sns.regplot(data[feature].loc[mask], data.position[mask], marker='o', color='blue', scatter_kws={'s': 2})
    axs[2].set_title('position')


check_na("prop_location_score2")


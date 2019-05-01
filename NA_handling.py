import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime as dt

data = pd.read_csv("training_set_VU_DM.csv", nrows=100000)
for col in data.columns:
    print(col)


def check_na(feature):
    """
    check correlation of feature with click, bool, position
    prints the 3 correlations and plots/crosstabs, depending on feature type
    :param feature: (str) column name of pandas df
    :return: none
    """

    # PROPORTION NAN
    mask = data[feature].notna()
    prop_na = len(data[feature].loc[mask]) / len(data[feature])
    print("Proportion notna values: ", prop_na)

    # CORRELATE WITH CLICK, BOOK, POSITION
    # FOR CONTINUOUS FEATURES
    if data[feature].dtype == "float64":

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
        sns.regplot(data.position[mask], data[feature].loc[mask], ax=axs[2], marker='o', color='blue', scatter_kws={'s': 2})
        axs[2].set_title('position')
        # SCATTER
        # sns.scatterplot(data.position[mask], data[feature].loc[mask], s=5, ax=axs[2])

    # FOR CATEGORICAL FEATURES CODED 0,1
    elif data[feature].dtype == "int64":

        book_cor = stats.pearsonr(data[feature].loc[mask], data.booking_bool[mask])
        click_cor = stats.pearsonr(data[feature].loc[mask], data.click_bool[mask])
        pos_cor = stats.pointbiserialr(data[feature].loc[mask], data.position[mask])

        print("Correlation with booking_bool: ", book_cor)
        print("Correlation with click_bool: ", click_cor)
        print("Correlation with position: ", pos_cor)

        print("CROSSTABS\n")
        print(pd.crosstab(data.click_bool[mask], data[feature].loc[mask]))
        print(pd.crosstab(data.booking_bool[mask], data[feature].loc[mask]))

        sns.boxplot(data[feature].loc[mask], data.position[mask])

    else:
        print("Unknown feature type.")


def try_imputing(feature):
    """
    compare correlations with click_bool, booking_bool and position before imputing and imputing NaN
    if numeric: mean, median, first and third quartile
    if binary: mode
    prints all possible correlations
    :param feature: (str) column name of pandas df
    :return: none
    """
    # CORRLEATE WITH CLICK, BOOK, POSITION
    # FOR CONTINUOUS FEATURES
    if data[feature].dtype == "float64":

        feature_imp_mean = data[feature].fillna(data[feature].mean())
        feature_imp_med = data[feature].fillna(data[feature].median())
        feature_imp_firstq = data[feature].fillna(data[feature].quantile(q=0.25))
        feature_imp_thirdq = data[feature].fillna(data[feature].quantile(q=0.75))

        methods = [(feature_imp_mean, "mean"), (feature_imp_med, "median"),
                   (feature_imp_firstq, "first quartile"), (feature_imp_thirdq, "third quartile")]

        for imp_method, name in methods:
            book_cor = stats.pointbiserialr(imp_method, data.booking_bool)
            click_cor = stats.pointbiserialr(imp_method, data.click_bool)
            pos_cor = stats.pearsonr(imp_method, data.position)

            print("Imputed with %s. Correlation with booking_bool: %s" % (name, book_cor))
            print("Imputed with %s. Correlation with click_bool: %s" % (name, click_cor))
            print("Imputed with %s. Correlation with position: %s" % (name, pos_cor))
            print("\n")

    # FOR CAT FEATURES
    elif data[feature].dtype == "int64":
        imp_mode = data[feature].fillna(data[feature].mode())

        book_cor = stats.pearsonr(imp_mode, data.booking_bool)
        click_cor = stats.pearsonr(imp_mode, data.click_bool)
        pos_cor = stats.pointbiserialr(imp_mode, data.position)

        print("Imputed with mode. Correlation with booking_bool: ", book_cor)
        print("Imputed with mode. Correlation with click_bool: ", click_cor)
        print("Imputed with mode. Correlation with position:", pos_cor)

    else:
        print("Unknown feature type.")


try_imputing("random_bool")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime as dt


def check_na(data, feature):
    """
    check correlation of feature with click, bool, position
    prints the 3 correlations and plots/crosstabs, depending on feature type
    :param feature: (str) column name of pandas df
    :return: none
    """

    print(f"Checking NaN values for feature {feature}...\n")
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
        plt.savefig(feature)
        
        plt.clf()
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

        # does not work
        # sns.boxplot(data[feature].loc[mask], data.position[mask])
        # plt.savefig(feature)

    else:
        print("Unknown feature type.")


def try_imputing(data, feature):
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


def outlier_plot(data, features = [], to_save=False, name=None):
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
        cols_exlude = ["srch_id", "date_time", "site_id", "prop_id", "visitor_location_country_id", "prop_country_id"]
        for col in data.columns:
            if data[col].dtype == "float64" and col not in cols_exlude:
                features.append(col)

    num_df = data[features]
    sns.set(style="ticks")
    ax = sns.boxplot(y="variable", x="value", orient="h", data=pd.melt(num_df))
    ax.set_xscale("log")
    if to_save:
        plt.savefig(name)
    else:
        plt.show()


if __name__ == "__main__":

    data = pd.read_csv("C:/Users/Frede/Dropbox/Master/DM/Assignments/2/DM2/training_set_VU_DM.csv", nrows=1_000_000)


    features = ["gross_bookings_usd", "price_usd", "prop_log_historical_price", "comp1_rate_percent_diff", "comp2_rate_percent_diff",
                "comp3_rate_percent_diff", "comp4_rate_percent_diff", "comp5_rate_percent_diff", "comp6_rate_percent_diff",
                "comp7_rate_percent_diff", "comp8_rate_percent_diff",]

    outlier_plot(data, features, to_save=True, name="outlier_plot.png")

from sklearn import preprocessing as pp
from sklearn import ensemble as en
import numpy as np
import os
from data_import import *
from data_import import oversample
from sklearn.impute import SimpleImputer

training = import_data('training_set_VU_DM.csv', nrows = 100000)
test = import_data('test_set_VU_DM.csv', nrows = 100000)

# get variable types
df = training
dtypeCount =[df.iloc[:,i].apply(type).value_counts() for i in range(df.shape[1])]
dtypeCount

def clip_outliers(data, feature_name, upper_quantile=True, manual_upper=None):
    """
    Handle outlier by discarding bottom and upper 5% quantile. The new value is the
    0.05 or 0.95 quantile value.
    :param train: training pd object
    :param feature_name:
    :return: no return
    """
    # lower bound for both options the same
    lower_bound = data[feature_name].quantile(.025)
    # values above or below those bounds are set to the boundary value
    data.loc[data[feature_name] < lower_bound, feature_name] = lower_bound
    # extract quantiles of feature feature_name
    if upper_quantile:
        upper_bound = data[feature_name].quantile(.975)
        data.loc[data[feature_name] > upper_bound, feature_name] = upper_bound
    else:
        # alternative clip at manual_upper a night (price for isntance 5000)
        data.loc[data[feature_name] > manual_upper, feature_name] = manual_upper


# demo
new_training = training
new_training['price_usd'].describe()
clip_outliers(new_training, "price_usd", upper_quantile=True)

def create_target_score(data, id_list, weight_rank=False):
    """
    Create target variable as numerical value: 5 booked; 1 clicked; 0 nothing
    If weight_rank=True then consider position.
    :param data: pd dataframe
    :param weight_rank boolean if rank is added to score or not
    :return: data with added target score variable
    """
    # get specific ids
    book_ids, click_ids, nothing_ids = id_list
    # add 5 / 1 / 0 values to dataframe
    data.loc[book_ids, 'target_score'] = 5
    data.loc[click_ids, 'target_score'] = 1
    data.loc[nothing_ids, 'target_score'] = 0
    # if weight also considered
    if weight_rank:
        # temporary target scores from before
        tmp_score = data.loc[:, 'target_score']
        # old target_score(5,1,0) + rank_score as described in report
        data.loc[:, 'target_score'] = tmp_score + 2 - (2 / (1 + np.exp(0.1*(-data['position']+1))))
    return data

# demo
#data_with_target = create_target_score(new_training, id_list, weight_rank=True)

def create_label(data, three_classes=False):
    """
    If three_classes==False then we only differentiate between booked and not booked as
    target value. If three classes available, then we differentiate between nothing, clicked and booked.
    :param data: od dataframe
    :param three_classes: boolen
    :return: pd dataframe with new variable label
    """
    if three_classes:
        # nothing=0; clickes not booked = 1; booked and clicked = 2
        data.loc[:,'target_label'] =  data['booking_bool'] + data['click_bool']
        return data
    else:
        # only booked / not booked
        data.loc[:, 'target_label'] =  data['booking_bool']
        return data
#demo
data_with_target = create_label(training)

def transform_price_per_night()

correlations = data.groupby('prop_country_id')[['price_usd', 'srch_length_of_stay']].corr().iloc[0::2,-1]
data.groupby('prop_country_id')[['price_usd','srch_length_of_stay']].corr().iloc[0::2]['srch_length_of_stay']

over_threshold = np.where(arr>0.4)[0]
prop_ids = sorted(np.unique(data["prop_country_id"]))

countries = list(np.array(prop_ids)[over_threshold])

sub = data
sub = sub.loc[sub["prop_country_id"].isin(countries),:]


sub["prop_country_id"] = sub["prop_country_id"].astype(int)
sub.loc[sub["prop_country_id"]== 56].plot.scatter("srch_length_of_stay","price_usd", c = "prop_country_id")
plt.show()

def simple_imputation(data, feature_name, type='mean'):
    """
    Impute NaN values with mean or median of vector
    :param data: np array vector
    :return: same np without missing values
    """
    # find missing ids
    missing_ids = data.loc[data[feature_name].isna(), :].index.values
    if type == 'mean':
        # find mean
        imp_val = data[feature_name].mean()
    elif type == 'median':
        # find median
        imp_val = data[feature_name].median()
    data.loc[missing_ids, feature_name] = imp_val

def numerical_imputation(data, target_name, feature_names, type='mean'):
    """
    Impute a categorial or numerical mssing variable with predictions coming from
    a random Forest model based on the non missing data
    :param data: pd dataframe
    :param feature_name: feature to be predicted
    :return: predicted feature
    """
    if type is 'random_forest':
        missing_ids = data.loc[data[feature_name].isna(),:].index.values
        not_missing_ids = data.loc[~data[feature_name].isna(),:].index.values
        # get target array
        data = data.dropna()
        y = np.array(data.loc[not_missing_ids, feature_name])
        # get feature array
        X = np.array(data.loc[not_missing_ids, data.columns != feature_name])
        # create regressor
        regr = en.RandomForestRegressor(max_depth=None, random_state=0, n_estimators=100)
        # fit regression trees
        regr.fit(X, y)
        # predict all missing values
        # concatenate missing values vector (predictions) to non missing values to end up with
        # full imputations
        # TODO
    else:
        # apply simple imputation of type
        simple_imputation(data, feature_name, type=type)

def standardize_feature(data, feature_name, group_by=None):
    """
    Standardize one feature either by all instances or grouped variable
    :param data: pd dataframe
    :param feature_name: feature to be standardized
    :return:
    """
    if group_by is not None:
        standardized_feature = data[feature_name].groupby(data[group_by]).apply(lambda x: (x - x.mean()) / x.std())
    else:
        # difference of feature the mean and divided by the standard deviation
        standardized_feature = data[feature_name].apply(lambda x: (x - x.mean()) / x.std())
    return standardized_feature

def normalize_feature(data, feature_name, group_by=None):
    """
    Normalize one feature (between 0 and 1)
    :param data: pd dataframe
    :param feature_name: feature to be standardized
    :return:
    """
    if group_by is not None:
        normalized_feature = data[feature_name].groupby(data[group_by]).apply(lambda x: (x - min(x))/(max(x) - min(x))+1)
    else:
        # difference of feature the min and divided by max - min (if the same add 1 to overcome numerical
        # trouble
        normalized_feature = data[feature_name].apply(lambda x: (x - min(x))/(max(x) - min(x))+1)
    return normalized_feature

def log_feature(data, feature_name, group_by=None):
    """
    Log transformation of feature.
    :param data:
    :param feature_name:
    :return: log transformed feature
    """
    if group_by is not None:
        log_feature = data[feature_name].groupby(data[group_by]).apply(lambda x: log(x+1))
    else:
        # for numerical reasons: add 1 to avoid log(0)
        log_feature = data[feature_name].apply(lambda x: log(x+1))
    return log_feature

def mean_med_std_feature(data, feature_name, type, group_by=None):
    """
    Create mean vector for feature feature_mean
    :param data:
    :param feature_name:
    :return: vector with values that are obtained
    """
    if group_by is not None:
        if type == "mean":
            # get mean of feature
            processed_feature = data.groupby([group_by])[feature_name].transform(lambda x: x.mean())
        elif type == "median":
            # get median of feature
            processed_feature = data.groupby([group_by])[feature_name].transform(lambda x: x.median())
        elif type == "std":
            # get standard deviation of feature
            processed_feature = data.groupby([group_by])[feature_name].transform(lambda x: x.std())
    else:
        if type == "mean":
            # get mean of feature
            processed_feature = data[feature_name].mean()
        elif type == "median":
            # get median of feature
            processed_feature = data[feature_name].median()
        elif type == "std":
            # get standard deviation of feature
            processed_feature = data[feature_name].std()
        # large vector for all values
        processed_feature = [processed_feature] * len(data[feature_name])

    return processed_feature


data = training
feature_name = "price_usd"
group_by = "srch_id"

def create_difference_feature(data, feature1, feature2, group_by=None):
    """
    Perform difference between two features as new feature
    :param data:
    :param feature1:
    :param feature2:
    :return:
    """
    if group_by is not None:
        tmp_data = data.groupby(data[group_by])
        difference_feature = tmp_data[feature1] - tmp_data[feature2]
    else:
        difference_feature = data[feature1] - data[feature2]

    return difference_feature

def perform_preprocessing(data, feature_name, type, group_by=None):
    """
    Filter the type of preprocessing and apply to feature
    :param data: pd dataframe
    :param feature_name: feature to be preprocessed
    :param type: norm/lognorm/stand/log
    :return:
    """

    if type == "stand":
        processed_feature = standardize_feature(data, feature_name, group_by)
    elif type == "norm":
        processed_feature = normalize_feature(data, feature_name, group_by)
    elif type == "lognorm":
        tmp_log = log_feature(data, feature_name)
        data.loc[:, "tmp_log"] = tmp_log
        processed_feature = normalize_feature(data, "tmp_log", group_by)
    elif type == "logstand":
        tmp_log = log_feature(data, feature_name)
        data.loc[:, "tmp_log"] = tmp_log
        processed_feature = standardize_feature(data, "tmp_log", group_by)
    elif type == "log":
        processed_feature = log_feature(data, feature_name, group_by)
    elif type == "mean":
        processed_feature = mean_med_std_feature(data, feature_name, "mean", group_by=group_by)
    elif type == "median":
        processed_feature = mean_med_std_feature(data, feature_name, "median", group_by=group_by)
    elif type == "std":
        processed_feature = mean_med_std_feature(data, feature_name, "std", group_by=group_by)

    return processed_feature

# demo: produce lognormalized price_usd per search_id vector
perform_preprocessing(data, "price_usd", group_by="srch_id", type="lognorm")
# mean price per search id
perform_preprocessing(data, "price_usd", group_by="srch_id", type="mean")

# perform feature extraction / generate all features we want to generate
# add them to training and test dataset
# preprocess all variables

def standardize_all_numerical(data, feature_list):
    """
    Standardize all features from feature_list
    :param data:
    :param feature_list:
    :return:
    """
    standardized_features = data[feature_list].astype(float).apply(pp.scale, axis=0, raw=True)
    return standardized_features

def normalize_all_numerical(data, feature_list):
    """
    Normalize all features from feature_list
    :param data:
    :param feature_list:
    :return:
    """
    normalized_features = data[feature_list].astype(float).apply(lambda x: pp.normalize(x)[0], axis=1, raw=True)
    return normalized_features

numeric_feature_list = ['srch_length_of_stay', 'srch_booking_window', \
        'srch_adults_count', 'srch_children_count',  \
        'prop_location_score2' ]

categorial_feature_list = ['srch_id', 'promotion_flag', 'prop_id', 'click_bool', 'booking_bool']

def extract_train_features(data, numeric_feature_list, categorial_feature_list, target="book", max_rank=None):
    """
    Extract the traning features from the data which already incorporates the target
    as either label or score.
    :param data: pd dataframe with target_label or target_score
    :param numeric_feature_list: list of feature names that need to be kept
    :param categorial_feature_list
    :return: new data with only relevant features for training
    """
    # filter by both numerical and categorial features
    data = data.loc[:, numeric_feature_list + categorial_feature_list]

    # create new features
    data.loc[:, "price_diff"] = create_difference_feature(data, "visitor_hist_adr_usd", "price_usd")
    data.loc[:, "star_diff"] = create_difference_feature(data, "visitor_hist_starrating", "prop_starrating")
    data.loc[:, "average_price_country"] = perform_preprocessing(data, feature_name="price_usd", group_by="prop_country_id", type="mean")
    data.loc[:, "average_price_srch"] = perform_preprocessing(data, feature_name="price_usd", group_by="srch_id", type="mean")
    data.loc[:, "price_diff_country"] = create_difference_feature(data, "average_price_country", "price_usd")
    data.loc[:, "price_diff_srch"] = create_difference_feature(data, "average_price_srch", "price_usd")

    data.loc[:, "average_star_country"] = perform_preprocessing(data, feature_name="prop_starrating",
                                                                 group_by="prop_country_id", type="mean")
    data.loc[:, "average_star_srch"] = perform_preprocessing(data, feature_name="prop_starrating", group_by="srch_id",
                                                              type="mean")
    data.loc[:, "star_diff_country"] = create_difference_feature(data, "average_star_country", "visitor_hist_starrating")
    data.loc[:, "star_diff_srch"] = create_difference_feature(data, "average_star_srch", "visitor_hist_starrating")

    #
    data.loc[:, "average_loc1_srch"] = perform_preprocessing(data, feature_name="prop_location_score1",
                                                                group_by="srch_id", type="mean")
    data.loc[:, "average_loc2_srch"] = perform_preprocessing(data, feature_name="prop_location_score2",
                                                                group_by="srch_id", type="mean")
    #
    data.loc[:, "locscore1_diff_srch"] = create_difference_feature(data, "average_loc1_srch", "prop_location_score1")
    data.loc[:, "locscore2_diff_srch"] = create_difference_feature(data, "average_loc2_srch", "prop_location_score2")

    # mean
    data.loc[:, "average_num_country"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                                 group_by="prop_country_id", type="mean").mean(axis=1)
    data.loc[:, "average_num_srch"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                               group_by="srch_id", type="mean").mean(axis=1)
    data.loc[:, "average_num_prop"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                               group_by="prop_id", type="mean").mean(axis=1)
    # median
    data.loc[:, "median_num_country"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                               group_by="prop_country_id", type="median").mean(axis=1)
    data.loc[:, "median_num_srch"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                            group_by="srch_id", type="median").mean(axis=1)
    data.loc[:, "median_num_prop"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                            group_by="prop_id", type="median").mean(axis=1)
    # stds
    data.loc[:, "std_num_country"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                               group_by="prop_country_id", type="std").mean(axis=1)
    data.loc[:, "std_num_srch"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                            group_by="srch_id", type="std").mean(axis=1)
    data.loc[:, "std_num_prop"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                            group_by="prop_id", type="std").mean(axis=1)

    # new customer
    new_ids = data["visitor_hist_starrating"].isnull() & data["visitor_hist_adr_usd"].isnull()
    data.loc[:, "new_customer"] = 0
    data.loc[new_ids, "new_customer"] = 1
    # person per room
    data.loc[:, "guest_count"] = data.loc[:, "srch_adults_count"] + data.loc[:, "srch_children_count"]
    #
    children_ids = data["srch_children_count"]>0
    data.loc[:, "children_bool"] = 0
    data.loc[children_ids, "children_bool"] =  1

    id_list = get_id_list(data, max_rank=max_rank)
    if target == "book":
        data.loc[:, "target"] = create_label(data, three_classes=False)
    elif target == "book_click":
        data.loc[:, "target"] = create_label(data, three_classes=True)
    elif target == "score":
        data.loc[:, "target"] = create_target_score(data, id_list, weight_rank=False)
    elif target == "score_rank":
        data.loc[:, "target"] = create_target_score(data, id_list, weight_rank=True)


    # fill non existing prop review scores with 0s
    df.loc[:, 'prop_review_score'] = data['prop_review_score'].fillna(0)
    ds.loc[:, 'loc_ratio2'] = loc_ratio2(dset).fillna(0)
    ds.loc[:, 'norm_star_rating'] = norm_pcid(dset, 'prop_starrating')
    ds.loc[:, 'nlog_price'] = log_norm_srch_id(dset, 'price_usd').fillna(0)
    ds.loc[:, 'label'] = label(dset)

    return ds.fillna(0)

def test_feature_extraction(dset):

    field_list = ['promotion_flag', 'srch_length_of_stay',
        'srch_booking_window', 'srch_adults_count',
        'srch_children_count', 'prop_id', \
        'prop_location_score2', 'srch_id']

    ds = dset[field_list]

    ds.loc[:, 'prop_review_score'] = dset['prop_review_score'].fillna(0)
    ds.loc[:, 'loc_ratio2'] = loc_ratio2(dset).fillna(0)
    ds.loc[:, 'norm_star_rating'] = norm_pcid(dset, 'prop_starrating')
    ds.loc[:, 'nlog_price'] = log_norm_srch_id(dset, 'price_usd').fillna(0)

    return ds.fillna(0)




### TEST
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)
data = data.loc[:, data.columns != "date_time"]

# def get_taxes(data):
#     """
#     Some countries have taxes on the prices
#     :param data:
#     :return:
#     """
#     # gross booking price
#     corr_data = data[data['gross_bookings_usd'].notnull()]
#     # tax = gross - (nights*price)
#     # add tax as new feature
#     corr_data.loc[:,'tax'] = corr_data['gross_bookings_usd'] - (corr_data['price_usd'] * corr_data['srch_length_of_stay'])
#     #
#     corr_data[corr_data['tax'] > 0].groupby('prop_country_id').mean()['tax'].sort_values(ascending=False)
#
#     return corr_data









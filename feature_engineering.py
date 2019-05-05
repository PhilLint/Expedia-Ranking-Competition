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

def clip_outliers(data, feature_name, upper_quantile=True):
    """
    Handle outlier by discarding bottom and upper 5% quantile. The new value is the
    0.05 or 0.95 quantile value.
    :param train: training pd object
    :param feature_name:
    :return:
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
        # alternative clip at 5000 a night
        data.loc[data[feature_name] > 5000, feature_name] = 5000


# demo
new_training['price_usd'].describe()
clip_outliers(new_training, "price_usd", upper_quantile=True)
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
data_with_target = create_target_score(new_training, id_list, weight_rank=True)

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

## Preprocessing and feature generating functions
def standardize_per_country(data, feature_name):
    """
    Normalize feature_name by the same feature for the instances of the same
    country.
    :param data: pd dataframe
    :param feature_name: feature to be standardized per country
    :return: feature vector with newly standardized values
    """
    # group by country and then apply standardization to that subset
    standardized_feature = data[feature_name].groupby(data['prop_country_id']).apply(lambda x: (x - x.mean()) / x.std())
    return standardized_feature

def log_standardize_per_srch_id(data, feature_name, search_id=True):
    """
    Logarithm of feature as a preprocessing measure.
    :param data: pd dataframe
    :param feature_name: feature to be log transformed.
    :param search_id: boolean if only lognormalization or also on search_id shall be performed
    :return: feature vector with new logarithm values
    """
    # log(x+1) for numerical reasons
    # temporary log vector
    tmp_log = data[feature_name].apply(lambda x: log(x+1))
    if search_id:
        #normalize by each srch_id
        log_norm_feature = tmp_log.groupby(dataset['srch_id']).apply(lambda x: (x-x.mean())/x.std())
    else:
        # without grouping by search_id
        log_norm_feature = tmp_log.apply(lambda x: (x-x.mean())/x.std())

    return log_norm_feature

def loc_score2_log_price(data):
    """
    Create feature: prop_location_score2 over the log normalized price per search id.
    :param data: pd dataframe
    :return: new feature vector
    """
    # get the log standardized adjusted by search id prices
    log_standardized_price = log_standardize_per_srch_id(data, 'price_usd')
    # normalize per search id
    log_normalized_price= log_standardized_price.groupby(data['srch_id']).apply(lambda x: (x - min(x))/(max(x) - min(x))+1)
    # return location score2 / log_normalized prices per search id
    return data['prop_location_score2'].fillna(0) / log_normalized_price

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



def numerical_imputation(data, feature_name, type='mean'):
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

# difference between mean purchase history and price
def diff_mean_history_to_price(data):
    """
    Creates feature which is the difference between the mean purchase history of a user
    and the current price of the instance.
    :param data: pd dataframe
    :return: new feature vector
    """
    # new feature is difference of visitors mean purchase history price and current usd price
    data['mean_price_hist_diff'] = data['visitor_hist_adr_usd'] - data['price_usd']
    return data['mean_price_hist_diff']

def diff_prop_cust_rating(data):
    """
    Create feature which is the difference between the mean str rating a customer gave in the past
    and the current property rating of a hotel.
    :param data: pd dataframe
    :return: new feature vector
    """
    # new feature is difference of visitors mean purchase history price and current usd price
    data['mean_star_rating_diff'] = data['visitor_hist_starrating'] - data['prop_starrating']
    return data['mean_star_rating_diff']




numeric_feature_list = ['srch_length_of_stay', 'srch_booking_window', \
        'srch_adults_count', 'srch_children_count', 'prop_id', 'click_bool', 'booking_bool', \
        'prop_location_score2' ]
categorial_feature_list = ['srch_id', 'promotion_flag']

def extract_train_features(data, numeric_feature_list, categorial_feature_list):
    """
    Extract the traning features from the data which already incorporates the target
    as either label or score.
    :param data: pd dataframe with target_label or target_score
    :param numeric_feature_list: list of feature names that need to be kept
    :param categorial_feature_list
    :return: new data with only relevant features for training
    """
    # filter by both numerical and categorial features
    df = data.loc[:,numeric_feature_list + categorial_feature_list]
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


def scale_features(dset):
    """

    :param dset:
    :return:
    """
    field_list = ['prop_review_score', 'promotion_flag', 'srch_length_of_stay', \
        'srch_booking_window', 'srch_adults_count', 'srch_children_count', \
        'loc_ratio2']

    tmp = dset[field_list].astype(float).apply(pp.scale, axis=0, raw=True)

    tmp.loc[:, 'norm_star_rating'] = dset['norm_star_rating'].astype(float)
    tmp.loc[:, 'nlog_price'] = dset['nlog_price'].astype(float)
    tmp.loc[:, 'prop_id'] = dset['prop_id'].astype(float)
    tmp.loc[:, 'srch_id'] = dset['srch_id'].astype(float)
    tmp.loc[:, 'click_bool'] = dset['click_bool'].astype(float)
    tmp.loc[:, 'booking_bool'] = dset['booking_bool'].astype(float)
    tmp.loc[:, 'label'] = dset['label'].astype(float)

    return tmp

def normalize_samples(dset):
    field_list = ['prop_review_score', 'promotion_flag', 'srch_length_of_stay', \
        'srch_booking_window', 'srch_adults_count', 'srch_children_count', \
        'loc_ratio2', 'norm_star_rating', 'nlog_price']

    tmp = dset[field_list].apply(lambda x: pp.normalize(x)[0], axis=1, raw=True)

    tmp.loc[:, 'prop_id'] = dset['prop_id']
    tmp.loc[:, 'srch_id'] = dset['srch_id']
    tmp.loc[:, 'click_bool'] = dset['click_bool']
    tmp.loc[:, 'booking_bool'] = dset['booking_bool']
    tmp.loc[:, 'label'] = dset['label']

    return tmp



### TEST
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)
data = data.loc[:, data.columns != "date_time"]

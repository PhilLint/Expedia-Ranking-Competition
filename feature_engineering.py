from sklearn import preprocessing as pp
import numpy as np
import os
from data_import import import_data

training = import_data('training_set_VU_DM.csv', nrows = 100000)
test = import_data('test_set_VU_DM.csv', nrows = 100000)

# get variable types
df = training
dtypeCount =[df.iloc[:,i].apply(type).value_counts() for i in range(df.shape[1])]
dtypeCount

def clip_outliers(data, feature_name):
    """
    Handle outlier by discarding bottom and upper 5% quantile. The new value is the
    0.05 or 0.95 quantile value.
    :param train: training pd object
    :param feature_name:
    :return:
    """
    # extract quantiles of feature feature_name
    upper_bound = data[feature_name].quantile(.95)
    lower_bound = data[feature_name].quantile(.05)
    # values above or below those bounds are set to the boundary value
    data.loc[data[feature_name] > upper_bound, feature_name] = upper_bound
    data.loc[data[feature_name] < lower_bound, feature_name] = lower_bound

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

def feature_extraction(inname):

    dset = pd.read_csv(inname)
    # outlier prices are clipped
    dset = dset[dset['price_usd']<5000]
    field_list = ['promotion_flag', 'srch_length_of_stay', 'srch_booking_window', \
        'srch_adults_count', 'srch_children_count', 'prop_id', 'click_bool', 'booking_bool', \
        'prop_location_score2', 'srch_id']

    ds = dset[field_list]

    ds.loc[:, 'prop_review_score'] = dset['prop_review_score'].fillna(0)
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


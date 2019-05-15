from sklearn import preprocessing as pp
from sklearn import ensemble as en
import numpy as np
import os
from data_import import *
from data_import import oversample
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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

# def transform_price_per_night(data, threshold=0.3):
#     correlations = data.groupby('prop_country_id')[['price_usd', 'srch_length_of_stay']].corr().iloc[0::2,-1]
#     arr = data.groupby('prop_country_id')[['price_usd','srch_length_of_stay']].corr().iloc[0::2]['srch_length_of_stay']
#     over_threshold = np.where(arr > threshold)[0]
#     prop_ids = sorted(np.unique(data["prop_country_id"]))
#     countries = list(np.array(prop_ids)[over_threshold])
#     sub = data
#     sub = sub.loc[sub["prop_country_id"].isin(countries),:]


def simple_imputation(data, feature_name, type='mean'):
    """
    Impute NaN values with mean or median of vector
    :param data: np array vector
    :return: imputed vector
    """
    # find missing ids
    missing_ids = data.loc[data[feature_name].isna(), :].index.values
    if type == 'mean':
        # find mean
        imp_val = data[feature_name].mean()
    elif type == 'median':
        # find median
        imp_val = data[feature_name].median()
    imputed_feature = [imp_val] * len(data.loc[missing_ids, feature_name])
    return imputed_feature

def numerical_imputation(data, target_name, feature_names, type='mean'):
    """
    Impute a categorial or numerical mssing variable with predictions coming from
    a random Forest model based on the non missing data
    :param data: pd dataframe
    :param feature_name: feature to be predicted
    :return: predicted feature
    """
    if type == "random_forest" or type == "lm":
        missing_ids = data.loc[data[target_name].isna(), :].index.values
        not_missing_ids = data.loc[~data[target_name].isna(), :].index.values
        # get target array
        imp = IterativeImputer(max_iter=50, random_state=0)
        imp_test = data.loc[:, feature_names]
        imp.fit(imp_test)
        # the model learns that the second feature is double the first
        imputed = pd.DataFrame(np.round(imp.transform(imp_test)))
        imputed.columns = feature_names

    if type is 'random_forest':
        y = np.array(data.loc[not_missing_ids, target_name])
        # get feature array
        X = np.array(imputed.loc[not_missing_ids, :])
        # create regressor
        regr = en.RandomForestRegressor(max_depth=None, random_state=0, n_estimators=100)
        # fit regression trees
        regr.fit(X, y)
        imputed.loc[missing_ids, target_name] = data.loc[missing_ids, target_name]
        imputed_feature = regr.predict(imputed.loc[missing_ids, imputed.columns.isin(feature_names)])

    elif type == "lm":
        y = np.array(data.loc[not_missing_ids, target_name])
        # get feature array
        X = np.array(imputed.loc[not_missing_ids, :])
        # create regressor
        reg = LinearRegression().fit(X, y)
        imputed.loc[missing_ids, target_name] = data.loc[missing_ids, target_name]
        imputed_feature = reg.predict(imputed.loc[missing_ids, imputed.columns.isin(feature_names)])
    else:
        # apply simple imputation of type mean or median
        imputed_feature = simple_imputation(data, target_name, type=type)

    # fill in imputated values
    data.loc[missing_ids, target_name] = imputed_feature


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

def find_predictors_for_imputation(data, target_name, threshold=0.15):
    """
    For numerical imputation: If linear regression imputation find highly correlated features to use as
    the features for the target feature.
    :param data:
    :param target_name:
    :return:
    """
    # get correlations of features on the target variable (to be imputed one to find possible predictors for a
    # linear regression
    correlations = data.corr(method="spearman").iloc[0::2][target_name]
    over_threshold = np.where(correlations > threshold)[0]
    predictor_names = correlations.axes[0][over_threshold].tolist()
    return predictor_names

def add_overall_comp(data):
    """
    Add all competitor variables to come up with one feature to reduce NA values. This feature is the sum of the
    comp rate feattures and indicates whether thie query is cheaper or more expensive then the competitors and sum
    indicates also if many are cheaper or more expensive (-3 -> 3 are cheaper)
    :param data:
    :return:
    """
    competition = data[["comp1_rate", "comp2_rate", "comp3_rate", "comp4_rate", "comp5_rate", "comp6_rate", "comp7_rate", "comp8_rate"]]
    competition = competition.sum(axis=1, skipna=True)
    print("na count:", sum(competition.isnull()))
    print("!= 0: ", len(competition != 0))

    return competition

def return_na_rate(data, feature_name, verbose=False):
    """
    Return NA rate to determine if imputation shall be done or not.
    :param data:
    :param feature_name:
    :return:
    """
    # PROPORTION NAN
    mask = data[feature_name].notna()
    prop_na = len(data[feature_name].loc[mask]) / len(data[feature_name])
    if verbose:
        print(f"Checking NaN values for feature {feature_name}...\n")
        print("Proportion notna values: ", prop_na)
    return prop_na

def return_imputables(data):
    """
    Get features that have na values
    :param data:
    :return: list with names of features with na values
    """
    impute_list = []
    for feature in data.columns:
        prop_na = return_na_rate(data, feature)
        if prop_na < 1:
            impute_list.append(feature)

    return impute_list

imp_list = return_imputables(data)



numeric_feature_list = ['srch_length_of_stay', 'srch_booking_window', \
        'srch_adults_count', 'srch_children_count',  \
        'prop_location_score2' ]

categorial_feature_list = ['srch_id', 'promotion_flag', 'prop_id', 'click_bool', 'booking_bool']



def impute(data, impute_list):
    """

    :param data:
    :param impute_list:
    :return:
    """
    for target in impute_list:
        preds = find_predictors_for_imputation(data, target, threshold=0.15)
        if len(preds) != 0:
            numerical_imputation(data, target, preds, type='lm')
        else:
            imputed_feature = numerical_imputation(data, target, preds, type='random_forest')


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

    ####################################################################################################################
    # IMPUTE MISSING VALUES
    # add overall comp
    data.loc[:, "overall_comp"] = add_overall_comp(data)
    # delete other comp variables
    filter_col = [col for col in data if col.startswith('comp')]
    data = data.drop(columns = filter_col)
    # get features that need to be imputed
    impute_list = return_imputables(data)
    # impute all numerical values of impute_list
    impute(data, impute_list)


    ####################################################################################################################
    # CREATE NEW FEATURES
    # add competitor variable

    data.loc[:, "price_diff"] = create_difference_feature(data, "visitor_hist_adr_usd", "price_usd")
    data.loc[:, "star_diff"] = create_difference_feature(data, "visitor_hist_starrating", "prop_starrating")
    data.loc[:, "average_price_country"] = perform_preprocessing(data, feature_name="price_usd", group_by="prop_country_id", type="mean")
    data.loc[:, "average_price_srch"] = perform_preprocessing(data, feature_name="price_usd", group_by="srch_id", type="mean")
    data.loc[:, "price_diff_country"] = create_difference_feature(data, "average_price_country", "price_usd")
    data.loc[:, "price_diff_srch"] = create_difference_feature(data, "average_price_srch", "price_usd")

    # create average per srch id
    data.loc[:, "average_loc1_srch"] = perform_preprocessing(data, feature_name="prop_location_score1",
                                                             group_by="srch_id", type="mean")
    data.loc[:, "average_loc2_srch"] = perform_preprocessing(data, feature_name="prop_location_score2",
                                                             group_by="srch_id", type="mean")
    data.loc[:, "average_prop_review_score"] = perform_preprocessing(data, feature_name="prop_review_score",
                                                                     group_by="srch_id", type="mean")
    data.loc[:, "average_star_country"] = perform_preprocessing(data, feature_name="prop_starrating",
                                                                 group_by="prop_country_id", type="mean")
    data.loc[:, "average_star_srch"] = perform_preprocessing(data, feature_name="prop_starrating", group_by="srch_id",
                                                              type="mean")
    # create differences from average to features
    data.loc[:, "star_diff_country"] = create_difference_feature(data, "average_star_country", "visitor_hist_starrating")
    data.loc[:, "star_diff_srch"] = create_difference_feature(data, "average_star_srch", "visitor_hist_starrating")
    data.loc[:, "locscore1_diff_srch"] = create_difference_feature(data, "average_loc1_srch", "prop_location_score1")
    data.loc[:, "locscore2_diff_srch"] = create_difference_feature(data, "average_loc2_srch", "prop_location_score2")
    data.loc[:, "prop_review_score_diff_srch"] = create_difference_feature(data, "average_prop_review_score", "prop_review_score")

    # mean of all numerical features per group
    data.loc[:, "average_num_country"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                                 group_by="prop_country_id", type="mean").mean(axis=1)
    data.loc[:, "average_num_srch"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                               group_by="srch_id", type="mean").mean(axis=1)
    data.loc[:, "average_num_prop"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                               group_by="prop_id", type="mean").mean(axis=1)
    # median of all numerical features
    data.loc[:, "median_num_country"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                               group_by="prop_country_id", type="median").mean(axis=1)
    data.loc[:, "median_num_srch"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                            group_by="srch_id", type="median").mean(axis=1)
    data.loc[:, "median_num_prop"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                            group_by="prop_id", type="median").mean(axis=1)
    # stds of all numerical features
    data.loc[:, "std_num_country"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                               group_by="prop_country_id", type="std").mean(axis=1)
    data.loc[:, "std_num_srch"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                            group_by="srch_id", type="std").mean(axis=1)
    data.loc[:, "std_num_prop"] = perform_preprocessing(data, feature_name=numeric_feature_list,
                                                            group_by="prop_id", type="std").mean(axis=1)

    # new customer yes or no
    new_ids = data["visitor_hist_starrating"].isnull() & data["visitor_hist_adr_usd"].isnull()
    data.loc[:, "new_customer"] = 0
    data.loc[new_ids, "new_customer"] = 1
    # person per room
    data.loc[:, "guest_count"] = data.loc[:, "srch_adults_count"] + data.loc[:, "srch_children_count"]
    # children yes or no
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










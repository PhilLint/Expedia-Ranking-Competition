from sklearn import preprocessing as pp
from sklearn import ensemble as en
from data_import import *
from data_import import oversample
from math import log
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#training = pd.read_csv(str('./data/') + 'training_set_VU_DM.csv', low_memory=False)
test = pd.read_csv(str('./data/') + 'test_set_VU_DM.csv', low_memory=False)
test = test.loc[:, test.columns != "date_time"]


#training_sample, _,_ = oversample(training, max_rank=10)
#training_sample = training_sample.loc[:, training_sample.columns != "date_time"]
#training_sample.to_csv("oversampled_training.csv")

# get variable types
#df = training
#dtypeCount =[df.iloc[:,i].apply(type).value_counts() for i in range(df.shape[1])]
#dtypeCount

def clip_outliers(data, feature_name, lower_quantile=False, upper_quantile=True, manual_upper=None):
    """
    Handle outlier by discarding bottom and upper 5% quantile. The new value is the
    0.05 or 0.95 quantile value.
    :param train: training pd object
    :param feature_name:
    :return: no return
    """
    if lower_quantile:
        # lower bound for both options the same
        lower_bound = data[feature_name].quantile(.01)
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
#new_training = training
#new_training['price_usd'].describe()
#clip_outliers(new_training, "price_usd", upper_quantile=True)

def create_target_score(data, id_list, weight_rank=False):
    """
    Create target variable as numerical value: 5 booked; 1 clicked; 0 nothing
    If weight_rank=True then consider position.
    :param data: pd dataframe
    :param weight_rank boolean if rank is added to score or not
    :return: data with added target score variable
    """
    # get specific ids
    book_ids, click_ids, nothing_ids = id_list[
    # add 5 / 1 / 0 values to dataframe
    data.loc[book_ids, 'target'] = 5
    data.loc[click_ids, 'target'] = 1
    data.loc[nothing_ids, 'target'] = 0
    # if weight also considered
    if weight_rank:
        # temporary target scores from before
        tmp_score = data.loc[:, 'target']
        # old target_score(5,1,0) + rank_score as described in report
        data.loc[:, 'target'] = tmp_score + 2 - (2 / (1 + np.exp(0.1*(-data['position']+1))))

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
        data.loc[:,'target'] =  data['booking_bool'] + data['click_bool']
    else:
        # only booked / not booked
        data.loc[:, 'target'] =  data['booking_bool']
#demo
#data_with_target = create_label(training)

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

def numerical_imputation(data, target_name, feature_names, imp_type='mean'):
    """
    Impute a categorial or numerical mssing variable with predictions coming from
    a random Forest model based on the non missing data
    :param data: pd dataframe
    :param feature_name: feature to be predicted
    :return: predicted feature
    """
    if imp_type == "random_forest" or imp_type == "lm":
        missing_ids = data.loc[data[target_name].isna(), :].index.values
        not_missing_ids = data.loc[~data[target_name].isna(), :].index.values

        # get target array
        imp = IterativeImputer(max_iter=500, random_state=0)
        if imp_type == "random_forest":
            feature_names = data.columns[data.columns != target_name]

        imp_test = data.loc[:, feature_names]
        imp.fit(imp_test)
        # the model learns that the second feature is double the first
        imputed = pd.DataFrame(np.round(imp.transform(imp_test)), index=data.index.values)
        imputed.columns = feature_names

    if imp_type is 'random_forest':
        y = np.array(data.loc[not_missing_ids, target_name])
        # get feature array
        X = np.array(imputed.loc[not_missing_ids, :])
        # create regressor
        regr = en.RandomForestRegressor(max_depth=None, random_state=0, n_estimators=100)
        # fit regression trees
        regr.fit(X, y)
        imputed.loc[missing_ids, target_name] = data.loc[missing_ids, target_name]
        imputed_feature = regr.predict(imputed.loc[missing_ids, imputed.columns.isin(feature_names)])

    elif imp_type == "lm":
        y = np.array(data.loc[not_missing_ids, target_name])
        # get feature array
        X = np.array(imputed.loc[not_missing_ids, :])
        # create regressor
        reg = LinearRegression().fit(X, y)
        #imputed.loc[missing_ids, target_name] = data.loc[missing_ids, target_name]
        imputed_feature = reg.predict(imputed.loc[missing_ids, imputed.columns.isin(feature_names)])
    else:
        # apply simple imputation of type mean or median
        imputed_feature = simple_imputation(data, target_name, type=imp_type)

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
        mean_x = data[feature_name].mean()
        x_std = data[feature_name].std()
        standardized_feature = data[feature_name].apply(lambda x: (x - mean_x) / x_std)
    return standardized_feature

def normalize_feature(data, feature_name, group_by=None):
    """
    Normalize one feature (between 0 and 1)
    :param data: pd dataframe
    :param feature_name: feature to be standardized
    :return:
    """
    if group_by is not None:
        normalized_feature = data[feature_name].groupby(data[group_by]).apply(lambda x: (x - min(x))/(max(x) - min(x)+1))
    else:
        # difference of feature the min and divided by max - min (if the same add 1 to overcome numerical
        # trouble
        min_x = data[feature_name].min()
        max_x = data[feature_name].max()
        normalized_feature = data[feature_name].apply(lambda x: (x - min_x)/((max_x - min_x)+1))
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


#data = training
#feature_name = "price_usd"
#group_by = "srch_id"

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
#perform_preprocessing(data, "price_usd", group_by="srch_id", type="lognorm")
# mean price per search id
#perform_preprocessing(data, "price_usd", group_by="srch_id", type="mean")

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
    data.loc[:, feature_list] = data[feature_list].astype(float).apply(pp.scale, axis=0, raw=True)

def normalize_all_numerical(data, feature_list):
    """
    Normalize all features from feature_list
    :param data:
    :param feature_list:
    :return:
    """
    for feature in feature_list:
        data.loc[:, feature] = normalize_feature(data, feature)

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
    itself = np.where(correlations == 1.0)[0]
    over_threshold = np.where(correlations > threshold)[0]
    # if present delete from array
    over_threshold  = over_threshold[over_threshold != itself]
    predictor_names = correlations.axes[0][over_threshold].tolist()
    return predictor_names

def add_overall_comp(data, verbose=False):
    """
    Add all competitor variables to come up with one feature to reduce NA values. This feature is the sum of the
    comp rate feattures and indicates whether thie query is cheaper or more expensive then the competitors and sum
    indicates also if many are cheaper or more expensive (-3 -> 3 are cheaper)
    :param data:
    :return:
    """
    competition = data[["comp1_rate", "comp2_rate", "comp3_rate", "comp4_rate", "comp5_rate", "comp6_rate", "comp7_rate", "comp8_rate"]]
    competition = competition.sum(axis=1, skipna=True)
    if verbose:
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

#imp_list = return_imputables(data)

def impute(data, impute_list):
    """
    Given a list of to be imputed features, extracts the predictors according to spearman correlation. If correlation
    is over threshold, its a predictor. If no predictors identified -> random forest imputation.
    :param data:
    :param impute_list:
    :return: no return.
    """
    for target in impute_list:
        preds = find_predictors_for_imputation(data, target_name=target, threshold=0.15)
        if len(preds) > 2:
            numerical_imputation(data, target_name=target, feature_names=preds, imp_type='lm')
        else:
            numerical_imputation(data, target_name=target, feature_names=preds, imp_type='random_forest')

def add_norm_features(data, name, group_by):
    cols = [col for col in data if col.startswith(name)]
    if name == "price_usd":
        new_srch_names = ["norm_" + s + "_" + str(group_by) for s in cols]
    else:
        new_srch_names = ["norm_" + s for s in cols]
    for i in range(len(new_srch_names)):
        feature = cols[i]
        data.loc[:, new_srch_names[i]] = np.array(data.loc[:, feature])
        data.loc[:, new_srch_names[i]] = standardize_feature(data, feature, group_by=group_by)
        data.loc[:, new_srch_names[i]] = normalize_feature(data, new_srch_names[i], group_by=group_by)


def generate_features(data):
    """
    Function to perform steps for all features to be generated and normalized. Separate function as used for train
    and test features separately.
    :param data:
    :return: nothing
    """
    ####################################################################################################################
    # IMPUTE MISSING VALUES
    # add overall comp
    data.loc[:, "overall_comp"] = add_overall_comp(data).astype(int)
    # delete other comp variables
    filter_col = [col for col in data if col.startswith('comp')]
    data = data.drop(columns=filter_col)
    # get features that need to be imputed
    impute_list = return_imputables(data)
    # impute all numerical values of impute_list
    impute(data, impute_list)
    ####################################################################################################################
    # CLIP OUTLIERS
    # data.dtypes -> only floats interesting
    # Exploration: only prices are skewed in the upper region
    clip_outliers(data, "visitor_hist_adr_usd")
    clip_outliers(data, "price_usd")

    ####################################################################################################################

    # CREATE NEW FEATURES
    data.loc[:, "price_diff"] = create_difference_feature(data, "visitor_hist_adr_usd", "price_usd")
    data.loc[:, "star_diff"] = create_difference_feature(data, "visitor_hist_starrating", "prop_starrating")
    # create averages
    data.loc[:, "country_average_price"] = perform_preprocessing(data, feature_name="price_usd",
                                                                 group_by="prop_country_id", type="mean")
    data.loc[:, "srch_average_price"] = perform_preprocessing(data, feature_name="price_usd", group_by="srch_id",
                                                              type="mean")
    # create average per srch id
    data.loc[:, "srch_average_loc1"] = perform_preprocessing(data, feature_name="prop_location_score1",
                                                             group_by="srch_id", type="mean")
    data.loc[:, "srch_average_loc2"] = perform_preprocessing(data, feature_name="prop_location_score2",
                                                             group_by="srch_id", type="mean")
    data.loc[:, "srch_average_prop_review_score"] = perform_preprocessing(data, feature_name="prop_review_score",
                                                                          group_by="srch_id", type="mean")
    data.loc[:, "country_average_star"] = perform_preprocessing(data, feature_name="prop_starrating",
                                                                group_by="prop_country_id", type="mean")
    data.loc[:, "srch_average_star"] = perform_preprocessing(data, feature_name="prop_starrating", group_by="srch_id",
                                                             type="mean")
    # create differences from average to features
    data.loc[:, "country_diff_price"] = create_difference_feature(data, "country_average_price", "price_usd")
    data.loc[:, "srch_diff_price"] = create_difference_feature(data, "srch_average_price", "price_usd")
    data.loc[:, "country_diff_star"] = create_difference_feature(data, "country_average_star",
                                                                 "visitor_hist_starrating")
    data.loc[:, "srch_diff_star"] = create_difference_feature(data, "srch_average_star", "visitor_hist_starrating")
    data.loc[:, "srch_diff_locscore1"] = create_difference_feature(data, "srch_average_loc1", "prop_location_score1")
    data.loc[:, "srch_diff_locscore2"] = create_difference_feature(data, "srch_average_loc2", "prop_location_score2")
    data.loc[:, "srch_diff_prop_review_score"] = create_difference_feature(data, "srch_average_prop_review_score",
                                                                           "prop_review_score")

    numeric_feature_list = data.select_dtypes(include=['float64']).columns[:-2]

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
    children_ids = data["srch_children_count"] > 0
    data.loc[:, "children_bool"] = 0
    data.loc[children_ids, "children_bool"] = 1
    ####################################################################################################################
    # Normalization after Standardization
    # log transform prices before
    data.loc[:, "price_usd"] = log_feature(data, "price_usd")
    data.loc[:, "visitor_hist_adr_usd"] = log_feature(data, "visitor_hist_adr_usd")
    # get float features that are going to be standardized and normalized
    float_features = data.select_dtypes(include=['float64']).columns
    # normalize per group
    add_norm_features(data, name="srch_diff", group_by="srch_id")
    add_norm_features(data, name="country_diff", group_by="srch_id")
    # price normalized per search id country id and prop id
    add_norm_features(data, name="price_usd", group_by="srch_id")
    add_norm_features(data, name="price_usd", group_by="prop_id")
    add_norm_features(data, name="price_usd", group_by="prop_country_id")
    # normalize all features
    # apply stand and norm to all of them
    standardize_all_numerical(data, feature_list=float_features)
    normalize_all_numerical(data, feature_list=float_features)

def save_final_dataframe_csv(data, name):
    """
    Save training/test data as csv to then use with target of choice if extract features
    :param data:
    :return:
    """
    data.to_csv(path_or_buf= name + "_data.csv", index=False)

def extract_train_features(data, target="book", max_rank=None):
    """
    Extract the traning features from the data which already incorporates the target
    as either label or score.
    :param data: pd dataframe with target_label or target_score
    :param numeric_feature_list: list of feature names that need to be kept
    :param categorial_feature_list
    :param preprocessed_data
    :return: new data with only relevant features for training
    """
    # Create Target
    id_list, _, _ = get_id_list(data, max_rank=max_rank)
    if target == "book":
        create_label(data, three_classes=False)
    elif target == "book_click":
        create_label(data, three_classes=True)
    elif target == "score":
        create_target_score(data, id_list, weight_rank=False)
    elif target == "score_rank":
        create_target_score(data, id_list, weight_rank=True)

# demo
#extract_train_features(data, target="book")

def test_feature_extraction(data):
    """
    Same steps but no target created
    :param data:
    :return:
    """
    generate_features(data)

#generate_features(test)
#generate_features(test)
#save_final_dataframe_csv(training_sample, "final_training")
#save_final_dataframe_csv(test, "test")
# target is most important.
#extract_train_features(training_sample, target="book", max_rank=10)

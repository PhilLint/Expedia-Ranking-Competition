from sklearn.ensemble import RandomForestClassifier


def forest_feat_select(X, y, n_estimators):
    """
    given X and y determine the importance of each feature in X in predicting y
    by applying random a forrest classifier
    :param X: pandas df
    :param y: single column pandas df or 1d numpy array
    :param n_estimators: n_estimators arg passed to rf_classifier instance
    :return:
    """

    rf = RandomForestClassifier(n_estimators=100)

    rf.fit(X, y)
    feature_importances = rf.feature_importances_

    print("feature importance")
    for pair in zip(X.columns, feature_importances):
        print(pair)


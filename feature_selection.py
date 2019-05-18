import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor




def feature_selection(train_data, estimator, n_features=None):

    selector = RFE(estimator=estimator, n_features_to_select=n_features)

    selector.fit(X_train, y_train)

    cols = selector.support_


if __name__ == "main":

    estimator = RandomForestRegressor(n_estimators=100)








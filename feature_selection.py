import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from data_import import oversample



def feature_selection(estimator, n_features=None):

    eleminator = RFE(estimator=estimator, n_features_to_select=n_features)




if __name__ == "main":
    data = pd.read_csv("C:/Users/Frede/Dropbox/Master/DM/Assignments/2/DM2/training_set_VU_DM.csv", nrows=100_000)
    data = data.loc[:, data.columns != "date_time"]
    data_no_na = data.loc[:, data.notnull().all()]

    data_down, _, _, _ = oversample(data_no_na, max_rank=5)

    X_train = data_down.drop(columns=["booking_bool", "click_bool", "position"])
    y_train = data_down.loc[:, "booking_bool"]

    estimator = RandomForestRegressor(n_estimators=100)
    selector = RFE(estimator=estimator)
    selector.fit(X_train, y_train)

    cols = selector.support_







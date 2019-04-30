import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = pd.read_csv("training_set_VU_DM.csv", nrows=100000)
for col in data.columns:
    print(col)

################### NA HANDLING ###########################

### VISITOR_HIST_STARRATING
# PROPORTION NAN
len(data.visitor_hist_starrating.loc[data.visitor_hist_starrating.notna() == True]) / len(data.visitor_hist_starrating)
visit_star_mask = data.visitor_hist_starrating.notna() == True

# CORRLEATE WITH CLICK, BOOK, POSITION
stats.pointbiserialr(data.visitor_hist_starrating.loc[data.visitor_hist_starrating.notna() == True], data.booking_bool[visit_star_mask])
stats.pointbiserialr(data.visitor_hist_starrating.loc[data.visitor_hist_starrating.notna() == True], data.click_bool[visit_star_mask])
stats.pearsonr(data.visitor_hist_starrating.loc[data.visitor_hist_starrating.notna() == True], data.position[visit_star_mask])

# PLOT
sns.scatterplot(data.visitor_hist_starrating.loc[data.visitor_hist_starrating.notna() == True], data.position[visit_star_mask], s=5)
"""
CORRS: no correlations with click, book or position
PLOT: does not look promising
SUGGESTION: exclude
"""

### PROP LOCATION SCORE 2
# PROPORTION NAN
len(data.prop_location_score2.loc[data.prop_location_score2.notna() == True]) / len(data.prop_location_score2)
prop_score2_mask = data.prop_location_score2.notna() == True

# CORRLEATE WITH CLICK, BOOK, POSITION
stats.pointbiserialr(data.prop_location_score2.loc[data.prop_location_score2.notna() == True], data.booking_bool[prop_score2_mask])
stats.pointbiserialr(data.prop_location_score2.loc[data.prop_location_score2.notna() == True], data.click_bool[prop_score2_mask])
stats.pearsonr(data.prop_location_score2.loc[data.prop_location_score2.notna() == True], data.position[prop_score2_mask])


# PLOT
sns.scatterplot(data.prop_location_score2.loc[data.prop_location_score2.notna() == True], data.position[prop_score2_mask], s=5)
sns.regplot(data.prop_location_score2.loc[data.prop_location_score2.notna() == True], data.position[prop_score2_mask],
            marker='o', color='blue', scatter_kws={'s':2})

"""
CORR: 
- booking/click both < .1
- position -.17
SCATTER: confirms negative corr (?)
5th place Kagge comp: impute with first quartile
"""

data.prop_location_score2.fillna(data.prop_location_score2.quantile(0.25), inplace=True)


### GROSS BOOKINGS USD


len(data.prop_location_score2.loc[data.prop_location_score2.notna() == True]) / len(data.prop_location_score2)
prop_score2_mask = data.prop_location_score2.notna() == True

# CORRLEATE WITH CLICK, BOOK, POSITION
stats.pointbiserialr(data.prop_location_score2.loc[data.prop_location_score2.notna() == True], data.booking_bool[prop_score2_mask])
stats.pointbiserialr(data.prop_location_score2.loc[data.prop_location_score2.notna() == True], data.click_bool[prop_score2_mask])
stats.pearsonr(data.prop_location_score2.loc[data.prop_location_score2.notna() == True], data.position[prop_score2_mask])


# PLOT
sns.scatterplot(data.prop_location_score2.loc[data.prop_location_score2.notna() == True], data.position[prop_score2_mask], s=5)
sns.regplot(data.prop_location_score2.loc[data.prop_location_score2.notna() == True], data.position[prop_score2_mask],
            marker='o', color='blue', scatter_kws={'s':2})
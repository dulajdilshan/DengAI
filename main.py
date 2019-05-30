'''
Plan of the work
-----------------

1 - Integrate Data and fill the Null values
2 - Normalization: Scaling the values
3 - Adding new Features (if needs)
4 - Clustering the data
5 - Feature selection
6 - Pattern Discovery
'''


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gc

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
# Inputs
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import Imputer

# Data viz
from mlens.visualization import corr_X_y, corrmat

# Model evaluation
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator

# Ensemble
from mlens.ensemble import SuperLearner

from scipy.stats import uniform, randint
from matplotlib.pyplot import show


%matplotlib inline

# get input dataset
features_train = pd.read_csv('data/dengue_features_train.csv')
labels_train = pd.read_csv('data/dengue_labels_train.csv')
features_test = pd.read_csv('data/dengue_features_test.csv')

# Normalize the week_start_date feature value
features_train['week_start_date'] = pd.to_datetime(features_train['week_start_date'])
features_test['week_start_date'] = pd.to_datetime(features_test['week_start_date'])

# combine the labels and features of the training data-set
features_train['total_cases'] = labels_train['total_cases']

# get shapes
print(f'Feature and label train: {features_train.shape}')

# Split into cities: SJ and IQ
FL_sj = features_train.loc[features_train['city'] == 'sj']
FL_iq = features_train.loc[features_train['city'] == 'iq']

# get the correlation with the total cases
FL_sj_corr = FL_sj.corr().total_cases.sort_values(ascending=False)
FL_iq_corr = FL_iq.corr().total_cases.sort_values(ascending=False)
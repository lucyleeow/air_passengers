import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import sklearn.ensemble


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        path = os.path.dirname(__file__)
        ext_data = pd.read_csv(os.path.join(path, 'external_data.csv'))


        X_encoded = pd.merge(
            X_encoded, ext_data, how='left',
            left_on=['DateOfDeparture', 'Departure','Arrival'],
            right_on=['DateOfDeparture', 'Departure','Arrival'],
            sort=False)

        # remove city names
        X_encoded = X_encoded.drop(['A_City','D_City','MONTH'], axis=1)

        #hot encode arr and dep
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)

        # dates

        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("2010-01-01")).days)

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))

        X_encoded = X_encoded.drop(['year','month','day','weekday','week','DateOfDeparture'], axis=1)

        X_array = X_encoded.values



        return X_array

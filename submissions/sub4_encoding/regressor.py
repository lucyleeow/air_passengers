from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import sklearn.ensemble


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = sklearn.ensemble.GradientBoostingRegressor(
            learning_rate=0.05,
            n_estimators=1000,
            subsample=1,
            criterion='mse',
            max_features=15,
            min_samples_split=0.05,
            max_depth=7)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

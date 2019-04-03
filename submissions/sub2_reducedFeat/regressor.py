from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import sklearn.ensemble


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = sklearn.ensemble.GradientBoostingRegressor(
            learning_rate=0.1,
            n_estimators=800,
            subsample=1,
            criterion='mse',
            max_features=120,
            min_samples_split=0.1,
            max_depth=12)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

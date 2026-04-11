from sklearn.ensemble import HistGradientBoostingClassifier
from .base_model import BaseModel


class HistBoostModel(BaseModel):
    def __init__(self, **params):
        self.model = HistGradientBoostingClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
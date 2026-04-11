from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel


class AdaBoostModel(BaseModel):
    def __init__(self, **params):
        if "estimator" not in params:
            params["estimator"] = DecisionTreeClassifier()

        self.model = AdaBoostClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
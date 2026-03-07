from sklearn.svm import SVC
from .base_model import BaseModel


class SVMModel(BaseModel):
    # Support Vector Machine wrapper
    def __init__(self, **params):
        # probability=True enables probability estimates
        self.model = SVC(probability=True, **params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # Return probability of positive class only
        return self.model.predict_proba(X)[:, 1]
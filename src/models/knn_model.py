from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseModel
class KNNModel(BaseModel):
    # K-Nearest Neighbors wrapper
    def __init__(self, **params):
        self.model = KNeighborsClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # Return probability of positive class only
        return self.model.predict_proba(X)[:, 1]
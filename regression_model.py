
from sklearn.ensemble import RandomForestRegressor

class RegressionModel:
    def __init__(self, model_type="random_forest"):
        self.model_type = model_type
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

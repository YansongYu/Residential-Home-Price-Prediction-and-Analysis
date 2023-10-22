
import pandas as pd
from data_preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from regression_model import RegressionModel

class HousePricePredictor:
    def __init__(self, train_path, test_path):
        self.data_preprocessor = DataPreprocessor(train_path, test_path)
        self.feature_engineer = None
        self.model = RegressionModel()
        self.predictions = None

    def execute(self):
        self.data_preprocessor.preprocess()
        self.feature_engineer = FeatureEngineer(self.data_preprocessor.train_data)
        self.feature_engineer.engineer_features()
        
        # Splitting data into features
        X_train = self.feature_engineer.data
        
        self.model.train(X_train, self.data_preprocessor.y_train)

        # Making predictions on the test set
        self.feature_engineer = FeatureEngineer(self.data_preprocessor.test_data)
        self.feature_engineer.engineer_features()
        X_test = self.feature_engineer.data
        self.predictions = self.model.predict(X_test)

        return self.predictions

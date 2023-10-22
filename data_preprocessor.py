
import pandas as pd

class DataPreprocessor:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        self.y_train = None

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        self.y_train = self.train_data["SalePrice"]
        self.train_data.drop("SalePrice", axis=1, inplace=True)

    def handle_missing_values(self):
        for df in [self.train_data, self.test_data]:
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)

    def encode_categoricals(self):
        combined_data = pd.concat([self.train_data, self.test_data], axis=0)
        combined_data = pd.get_dummies(combined_data)
        
        self.train_data = combined_data.iloc[:self.train_data.shape[0], :]
        self.test_data = combined_data.iloc[self.train_data.shape[0]:, :]

    def preprocess(self):
        self.load_data()
        self.handle_missing_values()
        self.encode_categoricals()

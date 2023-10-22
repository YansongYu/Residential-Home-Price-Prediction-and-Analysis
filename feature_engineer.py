
import numpy as np

class FeatureEngineer:
    def __init__(self, data):
        self.data = data

    def generate_features(self):
        if 'GrLivArea' in self.data.columns and 'TotalBsmtSF' in self.data.columns:
            self.data['TotalArea'] = self.data['GrLivArea'] + self.data['TotalBsmtSF']

    def transform_features(self):
        if 'TotalArea' in self.data.columns:
            self.data['LogTotalArea'] = np.log1p(self.data['TotalArea'])

    def engineer_features(self):
        self.generate_features()
        self.transform_features()

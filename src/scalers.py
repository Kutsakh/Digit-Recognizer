from sklearn.preprocessing import StandardScaler  # set mean=0, std=1
from sklearn.preprocessing import MinMaxScaler  # set values to [0, 1]
from sklearn.base import TransformerMixin


class NoScaler(TransformerMixin):
    """
    Class that do not perform any data manipulations
    """
    def __init__(self):
        pass

    def fit(self, data):
        return self

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

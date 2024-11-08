from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variable:str, mappings:dict):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variable] = X[self.variable].map(self.mappings).astype(int)

        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, variable):
        self.variable = variable

    def fit(self, X, y=None):
        q1, q3 = np.percentile(X[self.variable], q=[25, 75])
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for i in X.index:
            if X.loc[i, self.variable] > self.upper_bound:
                X.loc[i, self.variable] = np.floor(self.upper_bound).astype(X[self.variable].dtype)
            elif X.loc[i, self.variable] < self.lower_bound:
                X.loc[i, self.variable] = np.ceil(self.lower_bound).astype(X[self.variable].dtype)
        return X


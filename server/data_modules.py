
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression




class MissingIndicator(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var+'_nan'] = X[var].isnull().astype(int)
        
        return X
    

class ExtractLetters(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.variable = 'cabin'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.variable] = X[self.variable].apply(lambda x: ''.join(re.findall("[a-zA-Z]+", x)) if type(x)==str else x)
        return X
    

class CategoricalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna('Missing')
        return X

    
class NumericalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X, y=None):
        self.median_dict_ = {}
        for var in self.variables:
            self.median_dict_[var] = X[var].median()
        return self
        

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna(self.median_dict_[var])
        return X

    
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
                    
    def fit(self, X, y=None):
        self.rare_labels_dict = {}
        for var in self.variables:
            t = pd.Series(X[var].value_counts() / np.float(X.shape[0]))
            self.rare_labels_dict[var] = list(t[t<self.tol].index)
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.rare_labels_dict[var]), 'rare', X[var])
        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
       
    def fit(self, X, y=None):
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        return self
    
    def transform(self, X):
        X = X.copy()
        X = pd.concat([X, pd.get_dummies(X[self.variables], drop_first=True)], 1)
        X.drop(self.variables, 1, inplace=True)
        
        # Adding missing dummies, if any
        missing_dummies = [var for var in self.dummies if var not in X.columns]
        if len(missing_dummies) != 0:
            for col in missing_dummies:
                X[col] = 0
        
        return X


class OrderingFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
            
    def fit(self, X, y=None):
        self.ordered_features = X.columns
        return self
    
    def transform(self, X):
        return X[self.ordered_features]


# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test  = scaler.transform(X_test)

# model = LogisticRegression(C=0.0005, class_weight='balanced', random_state=SEED_MODEL)
# model.fit(X_train, y_train)
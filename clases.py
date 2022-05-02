import numpy as np

class columnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        return self 
    
    
class columnZeroToNaNTransformer():
    def __init__(self,columns):
        self.columns=columns
    def transform(self,X,y=None):
        print('HOLA')
        X[self.columns] = X[self.columns].replace({0:np.nan,0.0:np.nan})
        return X
    def fit(self, X, y=None):
        return self
    
class outOfRangeTransformer():
    def __init__(self, columns):
        self.columns=columns
    def transform(self,X,y=None):
        print('MUNDO')
        X["infant deaths"] = X["infant deaths"].apply( lambda x : x if x<=1000 else np.nan)
        X["Measles"] = X["Measles"].apply( lambda x : x if x<=1000 else np.nan )
        X["under-five deaths"] = X["under-five deaths"].apply( lambda x : x if x<=1000 else np.nan )
        X["BMI"] = X["BMI"].apply( lambda x : x if (x<=60 and x>=10) else np.nan )
        X["GDP"] = X["GDP"].apply( lambda x : x if x>=200 else np.nan )
        X["Population"] = X["Population"].apply( lambda x : x if x>=800 else np.nan )
        X["Total expenditure"] = X["Total expenditure"].apply( lambda x : x if x>0 else np.nan )
        X[self.columns] = X[self.columns].apply(lambda x:x.replace({np.nan: (np.sum(x)/len(x))}))
        print(X)
        return X
    def fit(self, X, y=None):
        return self
import pandas as pd 

def calc_returns(x : pd.Series) -> pd.Series:
    ret = (x.diff() / x.shift(1)).dropna()
    return ret

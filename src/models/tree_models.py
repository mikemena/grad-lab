# tree_models.py
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def make_random_forest(tree_params=None, rf_params=None):
    tree_params = tree_params or {}
    rf_params = rf_params or {}
    params = {**tree_params, **rf_params}
    return RandomForestClassifier(**params)

def make_xgboost(tree_params=None, xgb_params=None):
    tree_params = tree_params or {}
    xgb_params = xgb_params or {}
    params = {**tree_params, **xgb_params}
    return XGBClassifier(**params)

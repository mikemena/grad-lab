from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

@dataclass
class SklearnLikeModel:
    model: Any

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        # Trees expose predict_proba; make sure we always return prob for class 1
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:,1]
        # Safety: if sigle column, assume it's positive class already
        return proba.ravel()

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >=threshold).astype(int)

    def get_model_info(self) -> Dict[str, Any]:
            name = self.model.__class__.__name__
            info = {"name": name}
            if name == "RandomForestClassifier":
                info["n_estimators"] = getattr(self.model, "n_estimators", None)
                info["max_depth"] = getattr(self.model, "max_depth", None)
            return info

def make_random_forest(params: Dict[str, Any]) -> SklearnLikeModel:
    # sensible defaults already handled by sklearn; you can inject class_weight etc.
    rf = RandomForestClassifier(**params)
    return SklearnLikeModel(rf)

def make_xgboost(params: Dict[str, Any]) -> SklearnLikeModel:
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed. `pip install xgboost`")
    # Ensure binary:logistic by default (probabilities)
    params = {"objective": "binary:logistic", **params}
    xgb = XGBClassifier(**params)
    return SklearnLikeModel(xgb)

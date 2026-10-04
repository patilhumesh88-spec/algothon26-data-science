"""Model zoo for the baseline ladder: Dummy -> LogReg -> RandomForest -> boosting."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


def make_rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                  min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)


def make_models() -> dict:
    models = {
        "Dummy": DummyClassifier(strategy="prior"),
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, class_weight="balanced")),
        "RandomForest": make_rf(),
        "HistGB": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                 random_state=RANDOM_STATE),
    }
    try:  # optional, only if installed
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                          subsample=0.9, eval_metric="logloss",
                                          random_state=RANDOM_STATE, n_jobs=-1)
    except Exception:
        pass
    return models

"""Late-shipment prediction. XGBoost if installed, otherwise scikit-learn fallback.
Validation uses a TIME-BASED split (train on the past, test on the future)."""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score

from core.db import get_conn

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:  # app still works without it
    HAS_XGB = False

FEATURES = ["base_transit_days", "port_congestion", "event_active",
            "event_duration", "supplier_reliability", "buffer_days"]


def _clean(X):
    """Pick the model columns and force plain numbers (XGBoost is strict about dtypes)."""
    return X[FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)


class DelayModel:
    def __init__(self, clf, kind, metrics):
        self.clf, self.kind, self.metrics = clf, kind, metrics

    def predict_proba(self, X):
        return self.clf.predict_proba(_clean(X))[:, 1]

    def explain(self, row):
        """Top 3 factors. XGBoost: SHAP-style contributions (built in).
        Fallback: global feature importance."""
        X = _clean(pd.DataFrame([row]))
        if self.kind == "xgboost":
            contrib = self.clf.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
        else:
            contrib = self.clf.feature_importances_
        pairs = sorted(zip(FEATURES, contrib), key=lambda p: -abs(p[1]))
        return [(name, float(val)) for name, val in pairs[:3]]


def train_model():
    conn = get_conn()
    hist = pd.read_sql_query("SELECT * FROM history ORDER BY ship_date", conn)
    conn.close()

    split = int(len(hist) * 0.8)                 # oldest 80% train, newest 20% test
    train, test = hist.iloc[:split], hist.iloc[split:]

    if HAS_XGB:
        clf = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1,
                                eval_metric="logloss")
        kind = "xgboost"
    else:
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
        kind = "sklearn"
    clf.fit(_clean(train), train["late"])

    ml_pred = clf.predict(_clean(test))
    rule_pred = (test["event_duration"] > test["buffer_days"]).astype(int)  # simple baseline
    metrics = {
        "kind": kind, "train_rows": len(train), "test_rows": len(test),
        "rule": {"precision": precision_score(test["late"], rule_pred, zero_division=0),
                 "recall": recall_score(test["late"], rule_pred, zero_division=0)},
        "ml": {"precision": precision_score(test["late"], ml_pred, zero_division=0),
               "recall": recall_score(test["late"], ml_pred, zero_division=0)},
    }
    return DelayModel(clf, kind, metrics)

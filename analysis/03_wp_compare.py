import sys, warnings
sys.path.insert(0, sys.argv[2])
import pandas as pd, numpy as np
from common import load
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, brier_score_loss
warnings.filterwarnings('ignore')
b, x = load(sys.argv[1])
print(len(b), b.match_id.nunique(), x.good.sum())
print(x[x.good].groupby(['group', 'year']).par.first().unstack(0).round(1))


def feats(d, inn):
    if inn == 1:
        return np.c_[d.balls_rem, d.runs_before, d.wk_before, d.par]
    return np.c_[d.balls_rem, d.need, 10 - d.wk_before, d.par]


def eng(X, inn):
    br, r, w, par = X.T
    lb = np.log1p(br)
    if inn == 1:
        ov = (120 - br) / 6
        crr = r / np.maximum(ov, 1)
        # crude resource-based projection
        proj = r + br * (par / 120) * (1 - w / 10) ** 0.6
        return np.c_[br, r, w, par, lb, crr, proj - par, (proj - par) * (120 - br) / 120, w * lb, w ** 2]
    rrr = r / np.maximum(br, 1) * 6
    res = br * (par / 120) * (w / 10) ** 0.6
    return np.c_[br, r, w, par, lb, rrr, np.minimum(rrr, 36) * lb, r - res, (r - res) / np.sqrt(br + 1), w * lb, r / np.sqrt(br + 1)]


models = {
    'gbm_mono': lambda inn: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
                                                          monotonic_cst=[1, 1, -1, -1] if inn == 1 else [1, -1, 1, 1]),
    'logit_eng': lambda inn: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
    'mlp': lambda inn: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16, 16), activation='tanh', alpha=1e-2, max_iter=60, early_stopping=True, random_state=0)),
}
for inn in (1, 2):
    d = b[(b.innings == inn)]
    d = d[d.y != 0.5]
    X = feats(d, inn); y = d.y.values.astype(int); grp = d.match_id.values
    res = {k: np.zeros(len(d)) for k in models}
    for tr, te in GroupKFold(5).split(X, y, grp):
        for k, mk in models.items():
            Xtr, Xte = (eng(X[tr], inn), eng(X[te], inn)) if k != 'gbm_mono' else (X[tr], X[te])
            mdl = mk(inn).fit(Xtr, y[tr])
            res[k][te] = mdl.predict_proba(Xte)[:, 1]
    print(f"innings {inn}: base rate {y.mean():.3f}")
    for k, p in res.items():
        print(f"  {k:10s} logloss {log_loss(y, p):.4f} brier {brier_score_loss(y, p):.4f}")
    p = (res['gbm_mono'] + res['mlp']) / 2
    print(f"  {'avg':10s} logloss {log_loss(y, p):.4f} brier {brier_score_loss(y, p):.4f}")
    # by phase
    for ph in ['PP', 'MID', 'DEATH']:
        mk_ = (d.phase == ph).values
        print('   ', ph, {k: round(log_loss(y[mk_], v[mk_]), 4) for k, v in res.items()})

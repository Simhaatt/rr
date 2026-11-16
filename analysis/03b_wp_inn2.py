import sys, warnings
sys.path.insert(0, sys.argv[2])
import pandas as pd, numpy as np
from common import load
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
b, x = load(sys.argv[1])
d = b[(b.innings == 2) & (b.y != 0.5)].copy()
br, r, w, par = d.balls_rem.values.astype(float), d.need.values.astype(float), (10 - d.wk_before).values.astype(float), d.par.values
y = d.y.values.astype(int); grp = d.match_id.values


def v1(br, r, w, par):
    lb = np.log1p(br); rrr = r / np.maximum(br, 1) * 6
    res = br * (par / 120) * (w / 10) ** 0.6
    return np.c_[br, r, w, par, lb, rrr, np.minimum(rrr, 36) * lb, r - res, (r - res) / np.sqrt(br + 1), w * lb, r / np.sqrt(br + 1)]


def v2(br, r, w, par):
    lb = np.log1p(br); sb = np.sqrt(br)
    rpb = np.minimum(r / np.maximum(br, 1), 6.0)          # required runs per ball, capped
    ew = np.minimum(w, br / 3.0)                           # wickets that can realistically be used
    env = par / 180.0
    return np.c_[rpb, rpb * sb, rpb * lb, rpb ** 2, r / (sb + 1), sb, lb, w, ew, ew * rpb, w * lb, env, env * sb,
                 (rpb > 1) * (rpb - 1), (rpb > 2) * (rpb - 2), (rpb > 3) * (rpb - 3), (br <= 6) * rpb, (br <= 12) * rpb, (br <= 6), (br <= 12)]


cands = {
    'v1': (v1, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))),
    'v2': (v2, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))),
    'gbm': (lambda *a: np.c_[a], lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, monotonic_cst=[1, -1, 1, 1])),
}
res = {k: np.zeros(len(d)) for k in cands}
for tr, te in GroupKFold(5).split(br, y, grp):
    for k, (f, mk) in cands.items():
        mdl = mk().fit(f(br[tr], r[tr], w[tr], par[tr]), y[tr])
        res[k][te] = mdl.predict_proba(f(br[te], r[te], w[te], par[te]))[:, 1]
res['v2+gbm'] = (res['v2'] + res['gbm']) / 2
for k, p in res.items():
    p = np.clip(p, 1e-6, 1 - 1e-6)
    print(f"{k:8s} all {log_loss(y, p):.4f} | br<=12 {log_loss(y[br <= 12], p[br <= 12]):.4f} | br<=6 {log_loss(y[br <= 6], p[br <= 6]):.4f} | 13-36 {log_loss(y[(br > 12) & (br <= 36)], p[(br > 12) & (br <= 36)]):.4f} | >36 {log_loss(y[br > 36], p[br > 36]):.4f}")
rrr = r / br * 6
for k in ['v1', 'v2', 'gbm']:
    t = pd.DataFrame({'p': res[k], 'y': y, 'rrr': pd.cut(rrr, [0, 6, 8, 10, 12, 14, 17, 20, 30, 200])})[br <= 6]
    print(k, "last over:\n", t.groupby('rrr', observed=True).agg(pred=('p', 'mean'), actual=('y', 'mean'), n=('y', 'size')).round(3).T)
mdl = cands['v2'][1]().fit(v2(br, r, w, par), y)
for s in [(6, 12, 5), (6, 6, 5), (1, 6, 1), (1, 2, 1), (1, 4, 3), (12, 30, 7), (12, 20, 5), (60, 60, 8), (60, 120, 4), (120, 180, 10), (120, 220, 10), (30, 30, 2), (3, 10, 4), (2, 3, 6)]:
    a = [np.array([float(v)]) for v in s] + [np.array([180.])]
    print(s, round(float(mdl.predict_proba(v2(*a))[0, 1]), 3))

"""Win probability (out-of-fold) -> leverage index -> per-ball WPA. Saves balls.pkl"""
import sys, warnings, pickle
sys.path.insert(0, sys.argv[2])
import pandas as pd, numpy as np
from common import load
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
OUT = sys.argv[3]
b, x = load(sys.argv[1])


def eng(br, r, w, par, inn):
    """Feature map. inn1: r=runs scored, w=wickets lost. inn2: r=runs needed, w=wickets in hand."""
    br, r, w, par = [np.asarray(a, float) for a in (br, r, w, par)]
    lb = np.log1p(br)
    if inn == 1:
        ov = (120 - br) / 6
        crr = r / np.maximum(ov, 1)
        proj = r + br * (par / 120) * (1 - w / 10) ** 0.6
        return np.c_[br, r, w, par, lb, crr, proj - par, (proj - par) * (120 - br) / 120, w * lb, w ** 2]
    rrr = r / np.maximum(br, 1) * 6
    res = br * (par / 120) * (w / 10) ** 0.6
    return np.c_[br, r, w, par, lb, rrr, np.minimum(rrr, 36) * lb, r - res, (r - res) / np.sqrt(br + 1), w * lb, r / np.sqrt(br + 1)]


class Blend:
    """Innings 1: smooth logistic. Innings 2: mean of smooth logistic and a monotone-constrained GBM
    (the GBM guards the end-game, where a purely parametric curve extrapolates badly)."""

    def __init__(self, inn):
        self.inn = inn
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
        self.mode = __import__('os').environ.get('WPMODE', 'blend')     # 'blend' (main), 'lr' (logistic only), 'gbm' (boosted trees only, both innings)
        cst = [1, -1, 1, 1] if inn == 2 else [1, 1, -1, -1]             # inn2: balls left, runs needed, wickets in hand, par; inn1: balls left, runs, wickets lost, par
        use_gb = (self.mode == 'gbm') or (inn == 2 and self.mode == 'blend')
        self.gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                                 l2_regularization=1.0, monotonic_cst=cst, random_state=0) if use_gb else None

    def fit(self, X, y):
        if self.mode != 'gbm':
            self.lr.fit(X, y)
        if self.gb is not None:
            self.gb.fit(X[:, :4], y)
        return self

    def predict_proba(self, X):
        if self.mode == 'gbm':
            p = self.gb.predict_proba(X[:, :4])[:, 1]
            return np.c_[1 - p, p]
        p = self.lr.predict_proba(X)[:, 1]
        if self.gb is not None:
            p = 0.5 * p + 0.5 * self.gb.predict_proba(X[:, :4])[:, 1]
        return np.c_[1 - p, p]


def fit(d, inn):
    d = d[(d.innings == inn) & (d.y != 0.5)]
    X = eng(d.balls_rem, d.runs_before if inn == 1 else d.need, d.wk_before if inn == 1 else 10 - d.wk_before, d.par, inn)
    return Blend(inn).fit(X, d.y.astype(int).values)


def wp(models, inn, br, r, wl, par):
    """Batting-side win prob. r = runs scored (inn1) or runs still needed (inn2); wl = wickets lost."""
    m1, m2 = models
    br, r, wl, par = [np.asarray(a, float) for a in (br, r, wl, par)]
    out = np.empty(len(br))
    term = (br <= 0) | (wl >= 10)
    if inn == 1:
        if term.any():   # innings over: hand over to the chase model
            out[term] = 1 - m2.predict_proba(eng(np.full(term.sum(), 120.), r[term] + 1, np.full(term.sum(), 10.), par[term], 2))[:, 1]
        if (~term).any():
            out[~term] = m1.predict_proba(eng(br[~term], r[~term], wl[~term], par[~term], 1))[:, 1]
    else:
        won = r <= 0
        lost = term & ~won
        live = ~won & ~term
        out[won] = 1.0
        out[lost] = np.where(r[lost] == 1, 0.5, 0.0)
        if live.any():
            out[live] = m2.predict_proba(eng(br[live], r[live], 10 - wl[live], par[live], 2))[:, 1]
    return out


# outcome menu for leverage: (runs, wicket, legal)
OUTC = [(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1), (4, 0, 1), (6, 0, 1), (0, 1, 1), (1, 0, 0)]
cat = np.select([b.wk == 1, b.legal == 0, b.tot == 0, b.tot == 1, b.tot == 2, b.tot == 3, b.tot.isin([4, 5])], [6, 7, 0, 1, 2, 3, 4], default=5)
b['cat'] = cat
pout = b.groupby(['innings', 'phase']).cat.value_counts(normalize=True).unstack(fill_value=0)
print("Outcome probabilities by innings/phase (dot,1,2,3,4,6,W,extra):\n", pout.round(3))

b['fold'] = -1
mids = b.match_id.drop_duplicates().values.copy()
rng = np.random.RandomState(7); rng.shuffle(mids)
foldmap = {mid: i % 5 for i, mid in enumerate(mids)}
b['fold'] = b.match_id.map(foldmap)
for c in ['wp_pre', 'wp_post', 'wp_post_nowk', 'li_raw'] + [f'h{j}' for j in range(8)]:
    b[c] = np.nan

for k in range(5):
    tr = b[b.fold != k]
    models = (fit(tr, 1), fit(tr, 2))
    for inn in (1, 2):
        idx = b.index[(b.fold == k) & (b.innings == inn)]
        d = b.loc[idx]
        sgn = 1 if inn == 1 else -1
        r0 = (d.runs_before if inn == 1 else d.need).values
        pre = wp(models, inn, d.balls_rem, r0, d.wk_before, d.par)
        post = wp(models, inn, d.balls_rem - d.legal, r0 + sgn * d.tot, d.wk_before + d.wk, d.par)
        post_nw = wp(models, inn, d.balls_rem - d.legal, r0 + sgn * d.tot, d.wk_before, d.par)
        P = pout.loc[inn].loc[d.phase.values].values
        li = np.zeros(len(d))
        for j, (rr, ww, ll) in enumerate(OUTC):
            h = wp(models, inn, d.balls_rem - ll, r0 + sgn * rr, d.wk_before + ww, d.par)
            li += P[:, j] * np.abs(h - pre)
            b.loc[idx, f'h{j}'] = h - pre      # what each possible outcome of this ball would have been worth (used by the Pressure Exam)
        b.loc[idx, 'wp_pre'] = pre; b.loc[idx, 'wp_post'] = post; b.loc[idx, 'wp_post_nowk'] = post_nw; b.loc[idx, 'li_raw'] = li

b['li'] = b.li_raw / b.li_raw.mean()
b['dwp'] = b.wp_post - b.wp_pre
b['dwp_nowk'] = b.wp_post_nowk - b.wp_pre

# ---- diagnostics
d = b[b.y != 0.5]
for inn in (1, 2):
    dd = d[d.innings == inn]
    print(f"OOF logloss inn{inn}: {log_loss(dd.y.astype(int), dd.wp_pre.clip(1e-6, 1 - 1e-6)):.4f}")
d = d.assign(bin=pd.cut(d.wp_pre, np.linspace(0, 1, 11)))
print("Calibration (OOF):\n", d.groupby(['innings', 'bin'], observed=True).agg(pred=('wp_pre', 'mean'), actual=('y', 'mean'), n=('y', 'size')).round(3))
print("mean dwp per ball (should be ~0):", b.groupby('innings').dwp.mean().round(5).to_dict())
print("LI describe:\n", b.li.describe(percentiles=[.1, .25, .5, .6, .7, .75, .8, .9, .95, .99]).round(3))
print("LI by innings/phase:\n", b.groupby(['innings', 'phase']).li.agg(['mean', 'median']).round(2))
print("LI by innings/wickets lost:\n", b.groupby(['innings', pd.cut(b.wk_before, [-1, 1, 3, 5, 7, 10])], observed=True).li.mean().round(2).unstack(0))
c = b[b.innings == 2].copy()
c['rrr'] = c.need / c.balls_rem * 6
print("Chase: LI by balls remaining x RRR:\n", c.groupby([pd.cut(c.balls_rem, [0, 12, 24, 36, 60, 90, 120]), pd.cut(c.rrr, [0, 6, 8, 10, 12, 15, 20, 100])], observed=True).li.mean().round(2).unstack())
print("LI by wp_pre:\n", b.groupby(pd.cut(b.wp_pre, [0, .05, .15, .3, .45, .55, .7, .85, .95, 1]), observed=True).li.agg(['mean', 'size']).round(2))
# sanity states (final fold models)
st = [(2, 6, 12, 5), (2, 6, 6, 5), (2, 1, 6, 9), (2, 1, 2, 9), (2, 12, 30, 3), (2, 60, 60, 2), (2, 60, 120, 6), (2, 120, 180, 0), (2, 120, 220, 0), (2, 30, 30, 8)]
print("chase sanity (balls_rem, need, wk lost) -> WP")
for inn, br, r, w in st:
    print("  ", br, r, w, round(float(wp(models, 2, [br], [r], [w], [180.])[0]), 3))
print("inn1 sanity (balls_rem, runs, wk lost) -> WP")
for br, r, w in [(120, 0, 0), (84, 30, 4), (84, 70, 0), (30, 150, 3), (30, 110, 7), (6, 200, 5), (6, 140, 8)]:
    print("  ", br, r, w, round(float(wp(models, 1, [br], [r], [w], [180.])[0]), 3))

b.to_pickle(f"{OUT}/balls.pkl")
x.to_pickle(f"{OUT}/matches.pkl")
print("saved", len(b))


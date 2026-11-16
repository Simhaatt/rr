"""How should the shrunk components be combined to best predict pressure value in unseen matches?"""
import os as _os
ROOT = _os.environ.get('RR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # repository root
import sys, importlib.util, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
spec = importlib.util.spec_from_file_location('r5', _os.path.join(ROOT, 'analysis', '05_ratings.py')); r5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r5)
b = r5.b
rng = np.random.RandomState(21)
mids = b.match_id.unique()
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
for role in ('bat', 'bowl'):
    a = r5.match_agg(role)
    s0 = r5.rates(a)
    el = list(s0.index[(s0.n_P >= r5.MIN_PBALLS) & (s0.pm >= r5.MIN_PMATCH)])
    ae = a[a.pid.isin(el)]
    pidc = 'striker_id' if role == 'bat' else 'bowler_id'
    d = b[b.bat_row] if role == 'bat' else b
    cv = d.groupby([pidc, 'match_id']).agg(r=('runs' if role == 'bat' else 'bowl_runs', 'sum'), n=('legal', 'sum'), w=('bat_out' if role == 'bat' else 'bowl_wkt', 'sum')).reset_index().rename(columns={pidc: 'pid'})
    cv = cv[cv.pid.isin(el)]
    X = []; Y = []
    feats = ['R_base_eb', 'K_base_eb', 'R_up_eb', 'K_up_eb', 'cons_eb', 'level_eb', 'base_eb', 'uplift_eb', 'score', 'conv', 'convw', 'rP', 'R_rP', 'K_rP', 'R_rN', 'K_rN', 'T_rP', 'T_rN']
    for i in range(NS):
        hh = pd.Series(rng.randint(0, 2, len(mids)), index=mids)
        q = [r5.full_pipeline(ae[ae.match_id.map(hh).values == h], el) for h in (0, 1)]
        c = [cv[cv.match_id.map(hh).values == h].groupby('pid')[['r', 'n', 'w']].sum() for h in (0, 1)]
        common = q[0][1].index.intersection(q[1][1].index)
        fr = []
        for h in (0, 1):
            x = q[h][1].loc[common].join(q[h][2].score).join(q[h][0][['rP', 'R_rP', 'K_rP', 'R_rN', 'K_rN', 'cons', 'T_rP', 'T_rN']])
            x['conv'] = (c[h].r / c[h].n).reindex(common) * (1 if role == 'bat' else -1)
            x['convw'] = (c[h].w / c[h].n).reindex(common) * (-1 if role == 'bat' else 1)
            fr.append(x)
        for h in (0, 1):
            xa = fr[h][feats]; yb = fr[1 - h][['T_rP', 'rP', 'R_rP', 'K_rP', 'cons']].add_prefix('y_')
            X.append((xa - xa.mean()).assign(split=i)); Y.append(yb - yb.mean())
    X = pd.concat(X); Y = pd.concat(Y)
    print(f"\n======== {role}: rows {len(X)}")
    print("corr of half-A feature with half-B target:")
    print(pd.DataFrame({t: [np.corrcoef(X[f], Y[t])[0, 1] for f in feats] for t in Y.columns}, index=feats).round(3).to_string())
    for nm, fs in {'4 comps': ['R_base_eb', 'K_base_eb', 'R_up_eb', 'K_up_eb'], '4 comps+cons': ['R_base_eb', 'K_base_eb', 'R_up_eb', 'K_up_eb', 'cons_eb'], 'base R,K only': ['R_base_eb', 'K_base_eb'],
                   '4 comps+conv': ['R_base_eb', 'K_base_eb', 'R_up_eb', 'K_up_eb', 'conv', 'convw'], 'conv only': ['conv', 'convw']}.items():
        A = X[fs].values; y = Y.y_T_rP.values
        # cross-fitted over splits (leave-split-groups-out) to get honest r
        grp = X.split.values % 5; pred = np.zeros(len(y))
        for k in range(5):
            co, *_ = np.linalg.lstsq(A[grp != k], y[grp != k], rcond=None); pred[grp == k] = A[grp == k] @ co
        co, *_ = np.linalg.lstsq(A, y, rcond=None)
        print(f"  {nm:14s} r={np.corrcoef(pred, y)[0, 1]:.3f}  coefs:", dict(zip(fs, np.round(co, 3))))

print('prices', r5.PRICES)

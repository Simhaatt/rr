"""Do pillar weights learned from unseen matches beat the fixed 50/25/25? Weights are fitted on some splits and judged on others
(cross-fitted by split), always predicting actual pressure value in the other half of matches."""
import os as _os
ROOT = _os.environ.get('RR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # repository root
import sys, json, importlib.util, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
spec = importlib.util.spec_from_file_location('r5', _os.path.join(ROOT, 'analysis', '05_ratings.py')); r5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r5)
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rng = np.random.RandomState(21); mids = r5.b.match_id.unique(); OUT = {}
for role in ('bat', 'bowl'):
    a = r5.match_agg(role); s0 = r5.rates(a)
    el = list(s0.index[(s0.nS >= r5.MIN_PBALLS) & (s0.pmS >= r5.MIN_PMATCH)]); ae = a[a.pid.isin(el)]
    rows = []
    for i in range(NS):
        hh = pd.Series(rng.randint(0, 2, len(mids)), index=mids)
        q = [r5.full_pipeline(ae[ae.match_id.map(hh).values == h], el) for h in (0, 1)]
        cm = q[0][2].index.intersection(q[1][2].index)
        for h in (0, 1):
            z = q[h][2].loc[cm, ['z_level', 'z_uplift', 'z_cons']]; y = q[1 - h][0].T0_rP.reindex(cm)
            rows.append(z.assign(y=y.values, split=i))
    D = pd.concat(rows).dropna()
    def r_of(w, d):
        return np.mean([np.corrcoef(g[['z_level', 'z_uplift', 'z_cons']].values @ w, g.y)[0, 1] for _, g in d.groupby('split')])
    fixed = {'Brief 50/25/25': (0.5, 0.25, 0.25), 'Level only': (1, 0, 0), 'Level + consistency 50/0/50': (0.5, 0, 0.5), 'Equal thirds': (1 / 3, 1 / 3, 1 / 3)}
    out = {k: r_of(np.array(w), D) for k, w in fixed.items()}
    # learned: non-negative least squares fitted on other splits, judged on held-out splits (5 groups)
    from scipy.optimize import nnls
    grp = D.split % 5; preds = []; ws = []
    for k in range(5):
        tr = D[grp != k]; te = D[grp == k]
        w, _ = nnls(tr[['z_level', 'z_uplift', 'z_cons']].values, (tr.y - tr.y.mean()).values); w = w / w.sum(); ws.append(w)
        preds.append(r_of(w, te))
    out['Learned on unseen matches (cross-fitted)'] = float(np.mean(preds))
    wl = np.mean(ws, axis=0)
    OUT[role] = dict(results={k: float(v) for k, v in out.items()}, learned=[float(x) for x in wl], n=NS)
    print(f'==== {role}: rating predicts unseen pressure value ->', {k: round(v, 3) for k, v in out.items()}, '| learned weights level/uplift/cons =', np.round(wl, 2))
json.dump(OUT, open(sys.argv[1] + '/weights_summary.json', 'w'), indent=1)

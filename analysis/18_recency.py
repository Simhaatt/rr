"""Recency weighting: does weighting recent matches more predict the most recent period better?
Train on matches before the cut (with or without recency weights), predict actual pressure value after the cut."""
import os as _os
ROOT = _os.environ.get('RR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # repository root
import sys, json, importlib.util, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
spec = importlib.util.spec_from_file_location('r5', _os.path.join(ROOT, 'analysis', '05_ratings.py')); r5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r5)
OUT = {}
md = pd.to_datetime(r5.mdate)
for cutq, lab in ((0.6, 'last 40% of matches'), (0.75, 'last 25% of matches')):
    cut = md.sort_values().iloc[int(len(md) * cutq)]
    for role in ('bat', 'bowl'):
        a = r5.match_agg(role); s0 = r5.rates(a)
        el = list(s0.index[(s0.nS >= r5.MIN_PBALLS) & (s0.pmS >= r5.MIN_PMATCH)])
        ae = a[a.pid.isin(el)]
        early = md.reindex(ae.match_id).values < np.datetime64(cut)
        tr, te = ae[early], ae[~early]
        ste = r5.rates(te)
        res = {}
        for hl in (None, 3.0, 2.0, 1.0, 0.5):
            if hl is None:
                w = None
            else:
                age = (cut - md).dt.days / 365.25
                w = (0.5 ** (age / hl)).clip(upper=1.0)
            s = r5.rates(tr, w)
            s = s.join(r5.cluster_se(tr, s))
            ok = [p for p in el if p in s.index and np.isfinite(s.loc[p, ['rP', 'rN']]).all() and s.loc[p, 'app'] > 0]
            prm = r5.eb_params(s, ok); s = r5.add_rel(s, prm); p = r5.pillars(s.loc[ok], s, prm); comp = r5.composite(p, p, prm)
            cm = [q for q in ok if q in ste.index and s.loc[q, 'nS'] >= 50 and ste.loc[q, 'nS'] >= 50 and np.isfinite(ste.loc[q, 'T0_rP'])]
            res['none' if hl is None else f'{hl}y'] = dict(n=len(cm), level=float(np.corrcoef(p.loc[cm, 'level_eb'], ste.loc[cm, 'T0_rP'])[0, 1]), score=float(np.corrcoef(comp.loc[cm, 'score'], ste.loc[cm, 'T0_rP'])[0, 1]))
        OUT[f'{role}|{lab}'] = res
        print(role, lab, {k: (v['n'], round(v['level'], 3), round(v['score'], 3)) for k, v in res.items()})
json.dump(OUT, open(sys.argv[1] + '/recency_summary.json', 'w'), indent=1)

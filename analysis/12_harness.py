"""Head-to-head test of rating designs on matches the rating never saw. Config comes from environment variables."""
import os as _os
ROOT = _os.environ.get('RR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # repository root
import sys, importlib.util, warnings, os
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
spec = importlib.util.spec_from_file_location('r5', _os.path.join(ROOT, 'analysis', '05_ratings.py')); r5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r5)
rng = np.random.RandomState(21)
mids = r5.b.match_id.unique()
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
cfg = f"VALUE={r5.VALUE} CENTRE={r5.CENTRE} PRESS={r5.PRESSMODE} UP={r5.UPMODE} KOW={r5.KOW} UPTOT={r5.UPTOT} AR={r5.AR_W} MU={r5.MATCHUP} VEN={r5.VENUE} CONS={r5.CONSMODE}"
res = {}
for role in ('bat', 'bowl'):
    a = r5.match_agg(role); s0 = r5.rates(a)
    el = list(s0.index[(s0.nS >= r5.MIN_PBALLS) & (s0.pmS >= r5.MIN_PMATCH)])
    ae = a[a.pid.isin(el)]
    rows = []
    for i in range(NS):
        hh = pd.Series(rng.randint(0, 2, len(mids)), index=mids)
        q = [r5.full_pipeline(ae[ae.match_id.map(hh).values == h], el) for h in (0, 1)]
        cm = q[0][1].index.intersection(q[1][1].index)
        cc = lambda x, y: np.corrcoef(x.loc[cm], y.loc[cm])[0, 1]
        sym = lambda fa, fb: 0.5 * (cc(fa(q[0]), fb(q[1])) + cc(fa(q[1]), fb(q[0])))
        T0 = lambda z: z[0].T0_rP; U0 = lambda z: z[0].T0_rP - z[0].T0_rN
        rows.append({'rel score': cc(q[0][2].score, q[1][2].score), 'rel level': cc(q[0][1].level_eb, q[1][1].level_eb), 'rel uplift': cc(q[0][1].uplift_eb, q[1][1].uplift_eb),
                     'rel cons': cc(q[0][1].cons_eb, q[1][1].cons_eb), 'level->actual': sym(lambda z: z[1].level_eb, T0), 'score->actual': sym(lambda z: z[2].score, T0),
                     'base->actual': sym(lambda z: z[1].base_eb, T0), 'own record->actual': sym(T0, T0), 'uplift->actual uplift': sym(lambda z: z[1].uplift_eb, U0),
                     'raw uplift->actual uplift': sym(U0, U0), 'n': len(cm)})
    res[role] = pd.DataFrame(rows).mean()
print(cfg); print(pd.DataFrame(res).round(3).T.to_string())


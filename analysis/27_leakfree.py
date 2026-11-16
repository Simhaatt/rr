"""Leakage-free split-half test.

In 05_ratings.py the split-half tests refit shrinkage, standardisation and the rating on each half, but the ball-level steps
(opponent quality, the average for the same kind of ball, the exam marks) are estimated once on all matches. Here every step is
refitted on one half only: each half is loaded on its own (HALF=seed,h) and the whole module runs on it. We then compare the halves:
  repeatable : rating in half A v rating in half B
  predicts   : rating (and strike rate / economy, and the player's own raw record) in half A v actual pressure value in half B,
               where the actual value in half B is itself computed from half B alone.
Players are those eligible on the full data (eligibility uses ball counts only, never outcomes).

Usage: python 27_leakfree.py OUT_DIR NSPLIT          (driver)
       python 27_leakfree.py OUT_DIR --half SEED H   (worker; called by the driver)
"""
import os, sys, json, pickle, subprocess, importlib.util
import numpy as np, pandas as pd
OUT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 3 and sys.argv[2] == '--half':
    seed, h = int(sys.argv[3]), int(sys.argv[4])
    os.environ['HALF'] = f'{seed},{h}'
    spec = importlib.util.spec_from_file_location('r5', os.path.join(HERE, '05_ratings.py')); r5 = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0], OUT]; spec.loader.exec_module(r5)
    full = pickle.load(open(f'{OUT}/results.pkl', 'rb'))
    out = {}
    for role in ('bat', 'bowl'):
        el_full = list(full['res'][role]['table'].index)
        a = r5.match_agg(role); el = [p for p in el_full if p in set(a.pid)]
        s, p, comp, prm = r5.full_pipeline(a[a.pid.isin(el)], el)
        conv_A = (s.cr_P + s.cr_N) / (s.cb_P + s.cb_N); conv_P = s.cr_P / s.cb_P.clip(lower=1)
        out[role] = pd.DataFrame({'score': comp.score, 'T0': s.T0_rP.reindex(comp.index), 'convA': conv_A.reindex(comp.index), 'convP': conv_P.reindex(comp.index)})
    pickle.dump(out, open(f'{OUT}/leakfree_{seed}_{h}.pkl', 'wb'))
    sys.exit(0)

NS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
env = dict(os.environ, WEIGHTS='learned')
rows = {'bat': [], 'bowl': []}
for seed in range(1000, 1000 + NS):
    for h in (0, 1):
        f = f'{OUT}/leakfree_{seed}_{h}.pkl'
        if not os.path.exists(f):
            r = subprocess.run([sys.executable, '-I', __file__, OUT, '--half', str(seed), str(h)], env=env, capture_output=True, text=True)
            if r.returncode:
                sys.exit(r.stderr[-2000:])
    q = [pickle.load(open(f'{OUT}/leakfree_{seed}_{h}.pkl', 'rb')) for h in (0, 1)]
    for role in ('bat', 'bowl'):
        A, B = q[0][role], q[1][role]; c = A.index.intersection(B.index)
        cc = lambda x, y: float(np.corrcoef(x.loc[c], y.loc[c])[0, 1])
        sym = lambda col: 0.5 * (cc(A[col], B.T0) + cc(B[col], A.T0))
        rows[role].append({'repeatable': cc(A.score, B.score), 'rating': sym('score'), 'own record': sym('T0'), 'overall SR/econ': sym('convA'), 'pressure SR/econ': sym('convP'), 'n': len(c)})
    print('split', seed, 'done', flush=True)
res = {}
for role in ('bat', 'bowl'):
    d = pd.DataFrame(rows[role])
    diff = lambda col: d['rating'] - d[col]
    res[role] = {'mean': d.mean().round(4).to_dict(), 'sd': d.std().round(4).to_dict(), 'nsplit': len(d),
                 'diff': {col: {'mean': float(diff(col).mean()), 'lo': float(diff(col).quantile(0.05)), 'hi': float(diff(col).quantile(0.95)), 'share_rating_ahead': float((diff(col) > 0).mean())}
                          for col in ('own record', 'overall SR/econ', 'pressure SR/econ')}}
    print(role, json.dumps(res[role], indent=1))
json.dump(res, open(f'{OUT}/leakfree_summary.json', 'w'), indent=1)

"""Head-to-head odds inside the contested group: in what share of the 500 matched resamples (averaged over the three
win-probability models) does player A rate above player B? Also: how often each contested player lands in places 7-10."""
import os as _os
ROOT = _os.environ.get('RR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # repository root
import sys, json, pickle
import pandas as pd, numpy as np
OUT = sys.argv[1]
DIRS = [_os.path.join(ROOT, 'output'), _os.path.join(ROOT, 'output_alt'), _os.path.join(ROOT, 'output_gbm')]   # rewritten to the Kaggle working folders by the notebook
CON = pickle.load(open(f'{OUT}/consensus.pkl', 'rb'))
res = {}
for role in ('bat', 'bowl'):
    C = CON[role]; pool = C.index
    boots = [pickle.load(open(f'{d}/results.pkl', 'rb'))['res'][role]['boot'].reindex(columns=pool) for d in DIRS]
    nb = min(len(x) for x in boots)
    B = pd.DataFrame(sum(x.iloc[:nb].fillna(0).values for x in boots) / sum(x.iloc[:nb].notna().values for x in boots), columns=pool)
    grp = list(C[C.tier == 'Contested'].sort_values('rank_c').index)
    firm = list(C[C.tier == 'Firm'].index)
    H = pd.DataFrame({b: [(B[a] > B[b]).mean() for a in grp] for b in grp}, index=grp)
    rk = B.rank(axis=1, ascending=False)
    nf = len(firm)
    land = {p: float(((rk[p] > nf) & (rk[p] <= 10)).mean()) for p in grp}          # lands in a contested top-10 place
    ahead = {p: float(H.loc[p].drop(p).mean()) for p in grp}                        # average head-to-head win share
    res[role] = dict(group=[C.name[p] for p in grp], H=[[float(v) for v in row] for row in H.values], land=[land[p] for p in grp], ahead=[ahead[p] for p in grp],
                     score=[float(C.score_c[p]) for p in grp], firm=len(firm))
    names = [C.name[p].split()[-1] for p in grp]
    print(f'\n==== {role}: {len(firm)} firm, {len(grp)} contested for {10 - nf} places')
    print(pd.DataFrame(np.round(H.values * 100).astype(int), index=names, columns=names).to_string())
    print(pd.DataFrame({'name': [C.name[p] for p in grp], 'rank': [int(C.rank_c[p]) for p in grp], 'lands in places 7-10': [round(land[p], 2) for p in grp], 'avg head-to-head': [round(ahead[p], 2) for p in grp]}).to_string(index=False))
json.dump(res, open(f'{OUT}/contest_summary.json', 'w'), indent=1)

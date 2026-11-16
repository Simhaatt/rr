"""Agreement between the main and the logistic-only win-probability model (read by 06_figures.py)."""
import sys, json, pickle
import numpy as np
O = sys.argv[1]; ALT = sys.argv[2]
A = pickle.load(open(f'{O}/results.pkl', 'rb')); B = pickle.load(open(f'{ALT}/results.pkl', 'rb')); aw = {}
for r in ('bat', 'bowl'):
    a = A['res'][r]['table']; b = B['res'][r]['table']
    j = a[['name', 'score']].join(b[['score']].rename(columns={'score': 'score_alt'}), how='inner')
    ta = list(a.name.head(10)); tb = list(b.name.head(10))
    aw[r] = dict(corr=float(np.corrcoef(j.score, j.score_alt)[0, 1]), overlap=len(set(ta) & set(tb)), out=[x for x in ta if x not in tb], inn=[x for x in tb if x not in ta])
json.dump(aw, open(f'{O}/altwp_summary.json', 'w'), indent=1)
print(aw)

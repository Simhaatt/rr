"""Uncertainty in the final ratings and in the gaps between neighbouring ranks.

Uses the 500 matched match-resamples of each win-probability model, averaged across the three models (as in 22_contest.py).
For each of the top 15: the rating with its 90% resampling interval. For each pair of neighbours (rank k and k+1): the
rating gap, its 90% interval, and the share of resamples in which the higher-ranked player stays ahead.
Usage: python 25_intervals.py OUT_DIR
"""
import sys, json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = sys.argv[1]
DIRS = [OUT, OUT + '_alt', OUT + '_gbm']
NAVY, PINK, GREY = '#0B1F4B', '#E6007E', '#9AA3B2'
CON = pickle.load(open(f'{OUT}/consensus.pkl', 'rb'))
res = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, role, title in zip(axes, ('bat', 'bowl'), ('Batters', 'Bowlers')):
    C = CON[role].sort_values('rank_c'); pool = C.index
    boots = [pickle.load(open(f'{d}/results.pkl', 'rb'))['res'][role]['boot'].reindex(columns=pool) for d in DIRS]
    nb = min(len(x) for x in boots)
    B = pd.DataFrame(sum(x.iloc[:nb].fillna(0).values for x in boots) / sum(x.iloc[:nb].notna().values for x in boots), columns=pool)
    top = list(pool[:15])
    # centre each resampled rating on the final rating, so the interval describes spread, not resampling bias
    shift = B.median() - C.score_c
    rows = []
    for i, p in enumerate(top):
        lo, hi = np.percentile(B[p] - shift[p], [5, 95])
        row = dict(rank=i + 1, name=C.name[p], rating=float(C.score_c[p]), lo=float(lo), hi=float(hi))
        if i + 1 < len(top):
            q = top[i + 1]; g = B[p] - B[q]
            row.update(gap=float(C.score_c[p] - C.score_c[q]), gap_lo=float(np.percentile(g - (g.median() - (C.score_c[p] - C.score_c[q])), 5)),
                       gap_hi=float(np.percentile(g - (g.median() - (C.score_c[p] - C.score_c[q])), 95)), ahead=float((g > 0).mean()))
        rows.append(row)
    # gap from each top-10 player to the 11th
    p11 = top[10]
    for r_, p in zip(rows[:10], top[:10]):
        r_['beats_11th'] = float((B[p] > B[p11]).mean())
    res[role] = rows
    print(f'\n== {title}')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    y = np.arange(len(top))[::-1]
    r = pd.DataFrame(rows)
    ax.errorbar(r.rating, y, xerr=[r.rating - r.lo, r.hi - r.rating], fmt='none', ecolor=GREY, elinewidth=2.2, capsize=0)
    ax.scatter(r.rating, y, color=[PINK if k < 10 else NAVY for k in range(len(top))], zorder=3, s=28)
    ax.set_yticks(y); ax.set_yticklabels([f"{k + 1}. {n}" for k, n in enumerate(r.name)], fontsize=8)
    ax.axhline(y[9] - 0.5, color='black', lw=0.7, ls='--')
    ax.set_xlabel('Final rating with 90% interval over 500 resamples'); ax.set_title(title, fontweight='bold')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/figs/f12_rating_intervals.png', dpi=200); plt.close()
json.dump(res, open(f'{OUT}/intervals_summary.json', 'w'), indent=1)

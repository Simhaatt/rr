"""Method flowchart, the rating-versus-reliability chart, and one player traced from raw balls to final rating.
Usage: python 29_charts.py OUT_DIR
"""
import sys, json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT = sys.argv[1]
NAVY, PINK, GREY, LIGHT = '#0B1F4B', '#E6007E', '#9AA3B2', '#EEF0F6'
CON = pickle.load(open(f'{OUT}/consensus.pkl', 'rb'))
R = pickle.load(open(f'{OUT}/results.pkl', 'rb'))

# ---- 1. flowchart of the implemented pipeline
steps = [('Supplied data', '883 matches; 803 analysed after removing rain-revised, abandoned and incomplete games'),
         ('Match state before every ball', 'balls left, runs, wickets, runs needed, par for the competition and year'),
         ('Win probability', 'three models, each fitted in five groups of matches so no match predicts itself'),
         ('Leverage and pressure', 'how far the next ball can move the result; top quarter = pressure balls; light knockout weight'),
         ('Value of each ball', 'half actual win probability added, half the Pressure Exam mark'),
         ('Fair comparison', 'opponent quality and type removed; compared with the same kind of ball and squeeze state'),
         ('Shrinkage', 'every rate pulled towards the average in proportion to how little evidence there is'),
         ('Rating', 'Level and Consistency, weights fitted on unseen matches; averaged over the three models'),
         ('Checks', '500 resamples, 100 split halves, leakage-free halves, held-out season and competition'),
         ('Top 10 lists and decision support', 'firm and contested places, intervals, role fit')]
fig, ax = plt.subplots(figsize=(9, 9.6)); ax.set_xlim(0, 10); ax.set_ylim(0, len(steps) * 1.0 + 0.2); ax.axis('off')
for i, (h, d) in enumerate(steps):
    y = len(steps) - i - 0.55
    ax.add_patch(FancyBboxPatch((0.4, y - 0.36), 9.2, 0.72, boxstyle='round,pad=0.02,rounding_size=0.12', fc=PINK if i in (3, 7) else LIGHT, ec=NAVY, lw=1))
    col = 'white' if i in (3, 7) else NAVY
    ax.text(0.7, y + 0.1, f'{i + 1}. {h}', fontsize=11, fontweight='bold', color=col, va='center')
    ax.text(0.7, y - 0.17, d, fontsize=8.6, color=col, va='center')
    if i < len(steps) - 1:
        ax.annotate('', xy=(5, y - 0.44), xytext=(5, y - 0.36), arrowprops=dict(arrowstyle='-|>', color=NAVY, lw=1.2))
plt.tight_layout(); plt.savefig(f'{OUT}/figs/f13_flowchart.png', dpi=200); plt.close()

# ---- 2. rating against reliability (share of resamples in the top 10), every eligible player
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, role, title in zip(axes, ('bat', 'bowl'), ('Batters', 'Bowlers')):
    C = CON[role].sort_values('rank_c'); cut = C.score_c.iloc[9]
    top = C.rank_c <= 10
    ax.scatter(C.p_top10_c[~top], C.score_c[~top], s=14, color=GREY, zorder=2)
    ax.scatter(C.p_top10_c[top], C.score_c[top], s=26, color=PINK, zorder=3)
    for p in C.index[:12]:
        ax.annotate(C.name[p].split()[-1], (C.p_top10_c[p], C.score_c[p]), fontsize=7, xytext=(3, 2), textcoords='offset points')
    ax.axvline(0.6, color=NAVY, lw=0.8, ls='--'); ax.axhline(cut, color=NAVY, lw=0.8, ls=':')
    ax.text(0.99, ax.get_ylim()[1] * 0.98, 'Strong evidence', ha='right', va='top', fontsize=8, color=NAVY)
    ax.text(0.01, ax.get_ylim()[1] * 0.98, 'High rating, less certain', ha='left', va='top', fontsize=8, color=NAVY)
    ax.set_xlabel('Share of 500 resamples in which the player is in the top 10'); ax.set_ylabel('Final rating'); ax.set_title(title, fontweight='bold')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/figs/f14_rating_reliability.png', dpi=200); plt.close()

# ---- 3. one player from start to finish (Sunil Narine, bowler)
b = pd.read_pickle(f'{OUT}/balls_scored.pkl')
t = R['res']['bowl']['table']; C = CON['bowl']; pid = int(C.index[C.name == 'Sunil Narine'][0]); x = t.loc[pid]
d = b[(b.bowler_id == pid)]
dp = d[d.press_bowl]
IV = {r['name']: r for r in json.load(open(f'{OUT}/intervals_summary.json'))['bowl']}
walk = {'balls': int(len(d)), 'matches': int(d.match_id.nunique()), 'press_balls': int(len(dp)), 'press_matches': int(dp.match_id.nunique()),
        'econ_P': float(x.Econ_P), 'econ_N': float(x.Econ_N), 'sr_P': float(x.SR_P), 'wkts_P': int(x.wkts_P),
        'raw_wpa_P': float((-dp.dwp * dp.L / dp.li.clip(lower=1e-9)).sum() / dp.L.sum() * 1e4),   # as it happened: no centring, no opponent adjustment
        'actual_P': float(x.T_rP * 1e4), 'exam_P': float(x.exam_rP * 1e4), 'blend_P': float(x.rP * 1e4), 'blend_N': float(x.rN * 1e4),
        'base_eb': float(x.base_eb * 1e4), 'uplift_raw': float(x.upB * 1e4), 'uplift_eb': float(x.uplift_eb * 1e4), 'level_eb': float(x.level_eb * 1e4),
        'cons': float(x.cball * 100), 'z_level': float(x.z_level), 'z_cons': float(x.z_cons), 'score_main': float(x.score),
        'score_models': {m: float(C[f'score|{m}'][pid]) for m in CON['models']}, 'score_c': float(C.score_c[pid]), 'rank_c': int(C.rank_c[pid]),
        'lo': IV['Sunil Narine']['lo'], 'hi': IV['Sunil Narine']['hi'], 'p_top10': float(C.p_top10_c[pid]), 'tier': C.tier[pid]}
json.dump(walk, open(f'{OUT}/walkthrough_summary.json', 'w'), indent=1)
print(json.dumps(walk, indent=1))

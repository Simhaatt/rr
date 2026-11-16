"""Charts + summary stats for the report."""
import sys, pickle, json, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
OUT = sys.argv[1]; FIG = OUT + "/figs"
import os; os.makedirs(FIG, exist_ok=True)
b = pd.read_pickle(f"{OUT}/balls_scored.pkl")
x = pd.read_pickle(f"{OUT}/matches.pkl")
R = pickle.load(open(f"{OUT}/results.pkl", 'rb'))
# ---- consensus view: the final rating is the average over three win-probability models (see 20_consensus.py)
CON = pickle.load(open(f"{OUT}/consensus.pkl", 'rb'))
_W = {'50/25/25 (base)': (0.5, 0.25, 0.25), '100/0/0': (1, 0, 0), '40/40/20': (0.4, 0.4, 0.2), '60/20/20': (0.6, 0.2, 0.2), '34/33/33': (0.34, 0.33, 0.33),
      '50/0/50': (0.5, 0, 0.5), '50/50/0': (0.5, 0.5, 0), '70/15/15': (.7, .15, .15)}
for _role in ('bat', 'bowl'):
    _t = R['res'][_role]['table']; _C = CON[_role]
    _t = _t.assign(score_main=_t.score, rank_main=_t['rank'], p_top10_main=_t.p_top10)
    for _a, _b in (('score', 'score_c'), ('rank', 'rank_c'), ('p_top10', 'p_top10_c'), ('rank_lo', 'rank_lo_c'), ('rank_hi', 'rank_hi_c'), ('sens_top10_share', 'sens_c'),
                   ('z_level', 'z_level_c'), ('z_uplift', 'z_uplift_c'), ('z_cons', 'z_cons_c'), ('tier', 'tier'), ('n_models_top10', 'n_models_top10')):
        _t[_a] = _C[_b].reindex(_t.index)
    for _k in CON['models']:
        _t['rank|' + _k] = _C['rank|' + _k].reindex(_t.index)
    _t = _t.sort_values('rank'); R['res'][_role]['table'] = _t
    R['res'][_role]['sens'] = pd.DataFrame({nm: (-(w[0] * _t.z_level + w[1] * _t.z_uplift + w[2] * _t.z_cons)).rank() for nm, w in _W.items()})
    _s = CON['summary'][_role]; _r = R['res'][_role]['rel']
    _r['score_main'] = _r['score']; _r['pred_score_main'] = _r['pred_score_to_rawP']
    _r['score'] = _s['rel']; _r['pred_score_to_rawP'] = _s['pred']; _r['chrono_score'] = _s['chrono']
    if _s['yy'] is not None:
        _r['yy_score'] = _s['yy']

PINK, NAVY, GREY, TEAL = '#E6007E', '#0B1F4B', '#9AA3B2', '#1A9E8F'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.titlesize': 10, 'figure.dpi': 200})
S = {}

# ---------- summary stats
S['matches_total'] = int(len(x)); S['matches_with_bbb'] = int(x.runs_1.notna().sum()); S['matches_used'] = int(x.good.sum())
S['balls_used'] = int(len(b)); S['dls_excluded'] = int((x.dls & x.runs_1.notna()).sum())
S['ipl_matches'] = int((x.good & (x.group == 'IPL')).sum()); S['intl_matches'] = int((x.good & (x.group == 'INTL')).sum())
S['thr'] = float(R['thr']); S['press_balls'] = int(b.press.sum()); S['lev_share'] = float(b.li[b.press].sum() / b.li.sum())
d = b[b.y != 0.5]
for inn in (1, 2):
    dd = d[d.innings == inn]; p = dd.wp_pre.clip(1e-6, 1 - 1e-6)
    S[f'wp{inn}'] = dict(logloss=float(log_loss(dd.y.astype(int), p)), brier=float(brier_score_loss(dd.y.astype(int), p)), auc=float(roc_auc_score(dd.y.astype(int), p)))
b['absd'] = b.dwp.abs()
g = b.groupby('press')
cmpt = pd.DataFrame({'Balls': g.size(), 'Runs per over': 6 * g.tot.sum() / g.legal.sum(), 'Dot ball %': 100 * g.apply(lambda q: (q.tot == 0).mean()),
                     'Boundary %': 100 * g.apply(lambda q: (q.is_four | q.is_six).mean()), 'Wicket %': 100 * g.wk.mean(),
                     'Mean leverage (x avg ball)': g.li.mean(), 'Mean |win-prob swing| per ball (pts)': 100 * g.absd.mean(),
                     'Batting side win prob (mean)': 100 * g.wp_pre.mean()}).T
cmpt.columns = ['Ordinary' if not c else 'Pressure' for c in cmpt.columns]
S['press_vs_ord'] = cmpt.round(2).to_dict()
comp_ = (b[b.press].groupby(['innings', 'phase']).size() / b.press.sum()).round(3)
S['composition'] = {f"inn{i}_{p}": float(v) for (i, p), v in comp_.items()}
# share of balls that are pressure, by how close the match finished
fin = b[b.innings == 2].groupby('match_id').agg(r2=('tot', 'sum'), w2=('wk', 'sum'), lb=('legal', 'sum'), r1=('runs_1', 'first'))
fin['chase_won'] = fin.r2 > fin.r1
fin['close'] = np.where(fin.chase_won, np.where(120 - fin.lb <= 6, 'Chase won in final over', np.where(120 - fin.lb <= 18, 'Chase won with 7-18 balls left', 'Chase won with 19+ balls left')),
                        np.where(fin.r1 - fin.r2 <= 10, 'Defended by 10 or fewer', np.where(fin.r1 - fin.r2 <= 30, 'Defended by 11-30', 'Defended by 31+')))
b['close'] = b.match_id.map(fin.close)
S['press_by_close'] = (100 * b.groupby('close').press.mean()).round(1).to_dict()
S['press_by_stage'] = (100 * b.groupby(b.stage.isin(['Final', 'Qualifier', 'Eliminator', 'Semi Final'])).press.mean()).round(1).to_dict()

# ---------- Fig 1: leverage maps
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
c = b[b.innings == 2].copy(); c['rrr'] = c.need / c.balls_rem * 6
bb = [0, 6, 12, 18, 24, 36, 48, 60, 90, 120]; rr = [0, 4, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 36]
t = c.groupby([pd.cut(c.balls_rem, bb), pd.cut(c.rrr, rr)], observed=False).li.agg(['mean', 'size']).reset_index()
t.loc[t['size'] < 25, 'mean'] = np.nan
piv = t.pivot(index='balls_rem', columns='rrr', values='mean')
im = ax[0].imshow(piv.values, cmap='magma_r', aspect='auto', vmin=0, vmax=6)
ax[0].set_xticks(range(len(piv.columns))); ax[0].set_xticklabels([f"{int(i.left)}-{int(i.right)}" for i in piv.columns], rotation=45, ha='right')
ax[0].set_yticks(range(len(piv.index))); ax[0].set_yticklabels([f"{int(i.left) + 1}-{int(i.right)}" for i in piv.index])
ax[0].set_xlabel('Required run rate'); ax[0].set_ylabel('Balls remaining'); ax[0].set_title('Chase: leverage by balls left and required rate')
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if np.isfinite(v):
            ax[0].text(j, i, f"{v:.1f}", ha='center', va='center', fontsize=6.5, color='white' if v > 3 else 'black')
plt.colorbar(im, ax=ax[0], fraction=0.04, label='Leverage (1.0 = average ball)')
c1 = b[b.innings == 1]
t = c1.groupby([pd.cut(c1.overid, [0, 3, 6, 9, 12, 15, 18, 20]), pd.cut(c1.wk_before, [-1, 0, 1, 2, 3, 4, 5, 6, 9])], observed=False).li.agg(['mean', 'size']).reset_index()
t.loc[t['size'] < 25, 'mean'] = np.nan
piv = t.pivot(index='overid', columns='wk_before', values='mean')
im = ax[1].imshow(piv.values, cmap='magma_r', aspect='auto', vmin=0.4, vmax=1.6)
ax[1].set_xticks(range(len(piv.columns))); ax[1].set_xticklabels(['0', '1', '2', '3', '4', '5', '6', '7+'])
ax[1].set_yticks(range(len(piv.index))); ax[1].set_yticklabels([f"{int(i.left) + 1}-{int(i.right)}" for i in piv.index])
ax[1].set_xlabel('Wickets lost'); ax[1].set_ylabel('Over'); ax[1].set_title('First innings: leverage by over and wickets lost')
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if np.isfinite(v):
            ax[1].text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=6.5, color='white' if v > 1.15 else 'black')
plt.colorbar(im, ax=ax[1], fraction=0.04, label='Leverage (1.0 = average ball)')
plt.tight_layout(); plt.savefig(f"{FIG}/f1_leverage_maps.png"); plt.close()

# ---------- Fig 2: calibration
fig, axs = plt.subplots(2, 2, figsize=(8, 4.6), gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
for i, inn in enumerate((1, 2)):
    dd = d[d.innings == inn]; ax0, ax1 = axs[0, i], axs[1, i]
    q = dd.groupby(pd.cut(dd.wp_pre, np.linspace(0, 1, 21)), observed=True).agg(p=('wp_pre', 'mean'), a=('y', 'mean'), n=('y', 'size'))
    ax0.plot([0, 1], [0, 1], color=GREY, lw=1, ls='--')
    ax0.scatter(q.p, q.a, s=10 + q.n / q.n.max() * 70, color=PINK, zorder=3, edgecolor='white', lw=0.5)
    ax0.set_ylabel('Actual win rate')
    ax0.set_title(f"Innings {inn}: log-loss {S[f'wp{inn}']['logloss']:.3f}, AUC {S[f'wp{inn}']['auc']:.3f}")
    ax1.bar(q.p, q.n / 1000, width=0.04, color=NAVY)
    ax1.set_ylabel('Deliveries\n(thousands)', fontsize=7); ax1.set_xlabel('Predicted win probability (out of sample)')
    for a in (ax0, ax1):
        for sp_ in ('top', 'right'):
            a.spines[sp_].set_visible(False)
plt.tight_layout(); plt.savefig(f"{FIG}/f2_calibration.png"); plt.close()

# ---------- Fig 3: example match
tot_li = b[b.innings == 2].groupby('match_id').li_raw.sum()
cand = fin[(fin.close == 'Chase won in final over') | (fin.close == 'Defended by 10 or fewer')].index
mid = tot_li.reindex(cand).idxmax()
mm = b[b.match_id == mid].copy(); mi = x.loc[mid]
mm['seq'] = np.arange(len(mm))
bf_wp = np.where(mm.innings == 1, mm.wp_pre, 1 - mm.wp_pre)
fig, ax = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={'height_ratios': [1.3, 1]})
ax[0].plot(mm.seq, 100 * bf_wp, color=NAVY, lw=1.4)
ax[0].axhline(50, color=GREY, lw=0.7, ls='--'); ax[0].set_ylim(0, 100)
ax[0].set_ylabel('Win prob. of side\nbatting first (%)')
ax[0].set_title(f"Worked example: {mi.home_team} v {mi.away_team}, {mi.match_date}\n({mi.result})", fontsize=9)
cols = np.where(mm.press, PINK, GREY)
ax[1].bar(mm.seq, mm.li.clip(upper=12), color=cols, width=1.0)
ax[1].set_ylim(0, 12.5); ax[1].text(mm.seq.max() - 1, 11.2, 'bars clipped at 12', fontsize=6.5, ha='right', color=NAVY)
ax[1].axhline(R['thr'], color=NAVY, lw=0.8, ls='--'); ax[1].text(2, R['thr'] + 0.15, 'pressure threshold', fontsize=7, color=NAVY)
ax[1].set_ylabel('Leverage\n(1.0 = average ball)'); ax[1].set_xlabel('Delivery number in match (pink = pressure ball)')
sw = mm.seq[mm.innings == 2].min()
for a in ax:
    a.axvline(sw, color='black', lw=0.7)
ax[0].text(sw + 2, 92, 'chase begins', fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/f3_example_match.png"); plt.close()
S['example'] = dict(match=f"{mi.home_team} v {mi.away_team}", date=str(mi.match_date), result=str(mi.result), press_share=float(mm.press.mean()))

# ---------- Fig 4/5: top-10 evidence (pressure vs usual)
for role, fn, title in (('bat', 'f4_top10_batters', 'batters'), ('bowl', 'f5_top10_bowlers', 'bowlers')):
    t = R['res'][role]['table'].head(10).iloc[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.8), gridspec_kw={'width_ratios': [1.5, 1, 1]})
    y = np.arange(len(t))
    ax[0].hlines(y, t.T_rN * 1e4, t.T_rP * 1e4, color=GREY, lw=1.5, zorder=1)
    ax[0].scatter(t.T_rN * 1e4, y, color=NAVY, s=28, label='Ordinary balls', zorder=2)
    ax[0].scatter(t.T_rP * 1e4, y, color=PINK, s=28, label='Pressure balls', zorder=3)
    ax[0].scatter(t.exam_rP * 1e4, y, marker='^', color=TEAL, s=26, label='Pressure Exam mark', zorder=3)
    ax[0].scatter(t.level_eb * 1e4, y, marker='D', facecolor='white', edgecolor=PINK, s=26, label='Final pressure level', zorder=4)
    ax[0].axvline(0, color='black', lw=0.6)
    ax[0].set_yticks(y); ax[0].set_yticklabels([f"{r}. {n}" + ('' if tr == 'Firm' else '  (contested)') for r, n, tr in zip(t['rank'], t.name, t.tier)])
    ax[0].set_xlabel('Win-probability points added per 100 balls\n(opponent-adjusted, 0 = average player)')
    ax[0].set_title(f'Top-10 {title}: pressure v usual'); ax[0].legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.4, -0.24), ncol=4, frameon=False, columnspacing=0.6, handletextpad=0.1)
    ax[1].barh(y, 100 * t.cball, color=TEAL); ax[1].axvline(0, color='black', lw=0.8)
    ax[1].set_yticks([]); ax[1].set_xlabel('Pressure balls not lost\n(points above average)'); ax[1].set_title('Consistency')
    ax[2].barh(y, 100 * t.p_top10, color=[NAVY if tr == 'Firm' else GREY for tr in t.tier]); ax[2].set_yticks([]); ax[2].set_xlim(0, 100)
    ax[2].axvline(60, color=PINK, lw=1, ls='--')
    ax[2].set_xlabel('% of resamples in the top 10\n(navy = firm, grey = contested)'); ax[2].set_title('Robustness')
    plt.tight_layout(); plt.savefig(f"{FIG}/{fn}.png"); plt.close()

# ---------- Fig 6: rank intervals
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
for i, (role, title) in enumerate((('bat', 'Batters'), ('bowl', 'Bowlers'))):
    t = R['res'][role]['table'].head(20).iloc[::-1]
    y = np.arange(len(t))
    ax[i].hlines(y, t.rank_lo, t.rank_hi, color=GREY, lw=3)
    ax[i].scatter(t['rank'], y, color=PINK, zorder=3, s=22)
    ax[i].axvline(10.5, color=NAVY, lw=0.8, ls='--')
    ax[i].set_yticks(y); ax[i].set_yticklabels(t.name, fontsize=7.5); ax[i].set_xlabel('Rank (dot) and middle-80% range over 500 resamples (bar)')
    ax[i].set_title(f'{title}: how firmly is each rank held?'); ax[i].set_xlim(0, max(60, t.rank_hi.max() + 2))
plt.tight_layout(); plt.savefig(f"{FIG}/f6_rank_intervals.png"); plt.close()

# ---------- Fig 7: baseline vs pressure level for the pool
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for i, (role, title) in enumerate((('bat', 'Batters'), ('bowl', 'Bowlers'))):
    t = R['res'][role]['table']
    top = t['rank'] <= 10
    ax[i].scatter(t.T_rN[~top] * 1e4, t.T_rP[~top] * 1e4, color=GREY, s=14, alpha=0.7)
    ax[i].scatter(t.T_rN[top] * 1e4, t.T_rP[top] * 1e4, color=PINK, s=26, zorder=3)
    lim = [min(t.T_rN.min(), t.T_rP.min()) * 1e4 - 5, max(t.T_rN.max(), t.T_rP.max()) * 1e4 + 5]
    ax[i].plot(lim, lim, color=NAVY, lw=0.8, ls='--'); ax[i].text(lim[1] - 2, lim[1] - 8, 'pressure = usual', fontsize=7, color=NAVY, ha='right')
    for _, r in t[top].iterrows():
        ax[i].annotate(r['name'].split()[-1], (r.T_rN * 1e4, r.T_rP * 1e4), fontsize=6.5, xytext=(3, 3), textcoords='offset points')
    ax[i].set_xlabel('Usual level: value per 100 ordinary balls'); ax[i].set_ylabel('Value per 100 pressure balls')
    ax[i].set_title(f'{title}: above the line = better under pressure')
plt.tight_layout(); plt.savefig(f"{FIG}/f7_pressure_vs_usual.png"); plt.close()

# ---------- Fig 8: reliability
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
lab = {'raw_R_rN': 'Run value, ordinary', 'raw_K_rN': 'Wicket value, ordinary', 'raw_R_rP': 'Run value, pressure', 'raw_K_rP': 'Wicket value, pressure', 'raw_up': 'Raw uplift (pressure - usual)',
       'level_eb': 'Level (rating part 1)', 'uplift_eb': 'Uplift, shrunk (no separate weight)', 'cons_eb': 'Consistency (rating part 2)', 'score': 'Final rating'}
for i, (role, title) in enumerate((('bat', 'Batters'), ('bowl', 'Bowlers'))):
    r = R['res'][role]['rel']
    v = [r[k] for k in lab]
    ax[i].barh(range(len(lab)), v, color=[PINK if k in ('score',) else NAVY if k.endswith('_eb') else GREY for k in lab])
    ax[i].set_yticks(range(len(lab))); ax[i].set_yticklabels(list(lab.values())); ax[i].invert_yaxis()
    ax[i].axvline(0, color='black', lw=0.6); ax[i].set_xlabel('Agreement between random halves (mean of 100)'); ax[i].set_title(title)
    for j, vv in enumerate(v):
        ax[i].text(max(vv, 0) + 0.01, j, f"{vv:.2f}", va='center', fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/f8_reliability.png"); plt.close()

# ---------- Fig 9: out-of-sample test - what predicts pressure value in matches the predictor never saw?
fig, ax = plt.subplots(1, 2, figsize=(10, 2.9), sharex=True)
for i, (role, title, cv) in enumerate((('bat', 'Batters', 'strike rate'), ('bowl', 'Bowlers', 'economy'))):
    r = R['res'][role]['rel']
    items = [('Own past pressure record (raw)', r['pred_rawP_to_rawP'], GREY), (f'Pressure {cv}', r['pred_convP_to_rawP'], GREY), (f'Overall {cv}', r['pred_convA_to_rawP'], GREY),
             ('Our Level (rating part 1)', r['pred_level_to_rawP'], NAVY), ('Our final Rating', r['pred_score_to_rawP'], PINK)]
    ax[i].barh(range(len(items)), [v for _, v, _ in items], color=[c for _, _, c in items])
    ax[i].set_yticks(range(len(items))); ax[i].set_yticklabels([n for n, _, _ in items]); ax[i].invert_yaxis()
    for j, (_, v, _) in enumerate(items):
        ax[i].text(v + 0.005, j, f"{v:.2f}", va='center', fontsize=7)
    ax[i].set_xlim(0, 0.36); ax[i].set_title(title)
fig.supxlabel('Correlation with win probability added under pressure in the other half of matches (mean of 100 random splits)', fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/f9_out_of_sample.png"); plt.close()

# ---------- Fig 10: squeeze - pressure that builds
SQ = json.load(open(f"{OUT}/squeeze_summary.json")); DT = json.load(open(f"{OUT}/design_tests.json", encoding="utf-8-sig"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={'width_ratios': [1.25, 1]})
ax[0].bar(range(6), SQ['d6'], color=[NAVY if v >= 0 else PINK for v in SQ['d6']])
ax[0].axhline(0, color='black', lw=0.6); ax[0].set_xticks(range(6)); ax[0].set_xticklabels(['0', '1', '2', '3', '4', '5+'])
ax[0].set_xlabel('Dot balls in the previous six legal balls of the innings'); ax[0].set_ylabel('Runs on the next ball,\nrelative to the same match situation')
ax[0].set_title('The more dots just bowled, the fewer runs next ball')
for j, v in enumerate(SQ['d6']):
    ax[0].text(j, v + (0.008 if v >= 0 else -0.022), f"{v:+.2f}", ha='center', fontsize=7)
eff = [SQ['bat']['effect'], SQ['bowl']['effect']]; err = [1.96 * SQ['bat']['se'], 1.96 * SQ['bowl']['se']]
ax[1].barh([0, 1], eff, xerr=err, color=[PINK, NAVY], error_kw=dict(lw=1, capsize=3)); ax[1].axvline(0, color='black', lw=0.6)
ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(['Batter, when squeezed\n(4+ dots or a wicket\nin the last 6 balls)', 'Bowler, when being hit\n(2+ boundaries\nin the last 6 balls)'], fontsize=7.5); ax[1].invert_yaxis()
ax[1].set_xlabel('Change in the SAME player\'s value, win-prob points per 100 balls\n(opponent-adjusted; bars = 95% interval)', fontsize=7.5); ax[1].set_title('Same player, squeezed v not squeezed')
plt.tight_layout(); plt.savefig(f"{FIG}/f10_squeeze.png"); plt.close()

# ---------- Fig 11: which ideas survived the test on unseen matches
rows = DT['rows']
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
yy = np.arange(len(rows))
for i, (title, k) in enumerate((('Batters', 1), ('Bowlers', 4))):
    ax[i].barh(yy - 0.2, [r[k] for r in rows], height=0.38, color=[PINK if r[7] == 'adopted' else GREY for r in rows], label='Rating agrees between random halves of matches')
    ax[i].barh(yy + 0.2, [r[k + 1] for r in rows], height=0.38, color=[NAVY if r[7] == 'adopted' else '#c9cedd' for r in rows], label='Level predicts actual pressure value in unseen matches')
    ax[i].axvline(rows[0][k], color=GREY, lw=0.8, ls='--'); ax[i].axvline(rows[0][k + 1], color='#c9cedd', lw=0.8, ls='--')
    for j, r in enumerate(rows):
        ax[i].text(r[k] + 0.004, j - 0.2, f"{r[k]:.2f}", va='center', fontsize=6.5); ax[i].text(r[k + 1] + 0.004, j + 0.2, f"{r[k + 1]:.2f}", va='center', fontsize=6.5)
    ax[i].set_title(title); ax[i].set_xlim(0, 0.42)
ax[0].set_yticks(yy); ax[0].set_yticklabels([r[0] for r in rows], fontsize=7.5); ax[0].invert_yaxis()
fig.supxlabel('Correlation (upper bar: rating v rating across halves; lower bar: Level v actual pressure value in the other half). Dashed = previous version.', fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{FIG}/f11_design_tests.png"); plt.close()
S['squeeze'] = SQ; S['design_tests'] = DT
S['knockout'] = json.load(open(f"{OUT}/knockout_summary.json")); S['KOW'] = float(R['KOW'])
rob = []
for lab, sfx in (('IPL matches only', '_ipl'), ('Internationals only', '_intl'), ('Without 2022', '_drop2022'), ('Without 2023', '_drop2023'), ('Without 2024', '_drop2024'), ('Without 2025', '_drop2025'),
                 ('Without 2026', '_drop2026'), ('Pressure = top 33% of balls', '_pct0.67'), ('Pressure = top 20% of balls', '_pct0.80'), ('Eligibility: 100+ pressure balls', '_min100'), ('Eligibility: 200+ pressure balls', '_min200')):
    try:
        V = pickle.load(open(f"{OUT}/results{sfx}.pkl", 'rb'))
    except Exception:
        continue
    row = [lab]
    for role in ('bat', 'bowl'):
        m = R['res'][role]['table']; v = V['res'][role]['table']; top = list(m.name.head(10)); vt = set(v.name.head(10)); elig = [n for n in top if n in set(v.name)]
        j = m[['score']].join(v[['score']], rsuffix='_v', how='inner')
        row += [int(len(v)), int(len(elig)), int(sum(n in vt for n in elig)), float(j.score.corr(j.score_v)), [n for n in elig if n not in vt]]
    rob.append(row)
S['robust_main'] = rob; S['robust'] = CON['robust']; S['consensus'] = CON['summary']
S['more'] = json.load(open(f"{OUT}/more_summary.json")); S['recency'] = json.load(open(f"{OUT}/recency_summary.json"))
try:
    V = pickle.load(open(f"{OUT}/results_min100.pkl", 'rb')); S['missed'] = {}
    for role in ('bat', 'bowl'):
        v = V['res'][role]['table']; m = set(R['res'][role]['table'].index)
        S['missed'][role] = [[r['name'], int(r['rank']), int(r['nS'])] for i, r in v.head(20).iterrows() if i not in m]
except Exception:
    S['missed'] = {'bat': [], 'bowl': []}
S['checks'] = json.load(open(f"{OUT}/checks_summary.json")); S['altwp'] = json.load(open(f"{OUT}/altwp_summary.json"))
S['allround'] = json.load(open(f"{OUT}/allround_summary.json")); S['insights'] = json.load(open(f"{OUT}/insights_summary.json")); S['AR_W'] = float(R['AR_W'])
S['ko_new_press'] = int((b.press & ~b.press0).sum()); S['ko_press'] = int((b.press & (b.mw > 1)).sum()); S['mw_counts'] = {str(k): int(v) for k, v in b.drop_duplicates('match_id').mw.value_counts().items()}
et = R['EXAM_TABLE']
S['exam_examples'] = {k: {c: float(et.loc[ix, c]) * 100 for c in ['dot', '1', '2', '4', '6', 'wicket']} for k, ix in {'Chase, death overs, 3-5 wickets down': (True, 2, 'DEATH', 1), 'Chase, powerplay, 0-2 wickets down': (True, 2, 'PP', 0), 'Chase, middle overs, 3-5 wickets down': (True, 2, 'MID', 1)}.items()}
S['exam_strata'] = int(len(et))

for role in ('bat', 'bowl'):
    S[f'{role}_KC'] = float(R['res'][role]['prm']['KC']); S[f'{role}_KCB'] = float(R['res'][role]['prm']['KCB'])
    t_ = R['res'][role]['table']; w_ = R.get('W_ROLE', {role: R['W']})[role]
    sdv = np.array([w_['level'] * t_.z_level.std(), w_['uplift'] * t_.z_uplift.std(), w_['cons'] * t_.z_cons.std()])
    S[f'{role}_influence'] = [float(v) for v in sdv / sdv.sum()]
    S[f'{role}_rel'] = {k: float(v) for k, v in R['res'][role]['rel'].items()}
    S[f'{role}_n'] = int(len(R['res'][role]['table']))
    S[f'{role}_prm_cons'] = float(R['res'][role]['prm']['cons'])
    t = R['res'][role]['table']
    S[f'{role}_mean_rel'] = t[[c for c in t.columns if c.endswith('_rel')]].mean().round(2).to_dict()
    S[f'{role}_sens'] = {c: [t.loc[i, 'name'] for i in R['res'][role]['sens'][c].sort_values().index[:10]] for c in R['res'][role]['sens'].columns}
    S[f'{role}_top10_noadj'] = t.sort_values('rank_noadj').name.head(10).tolist()
json.dump(S, open(f"{OUT}/summary.json", 'w'), indent=1, default=str)
print(json.dumps({k: S[k] for k in ['matches_total', 'matches_with_bbb', 'matches_used', 'balls_used', 'dls_excluded', 'ipl_matches', 'intl_matches', 'thr', 'press_balls', 'lev_share', 'wp1', 'wp2', 'composition', 'press_by_close', 'press_by_stage', 'example', 'bat_mean_rel', 'bowl_mean_rel']}, indent=1, default=str))
print(cmpt.round(2))
for role in ('bat', 'bowl'):
    print(role, 'sensitivity top10s:')
    for k, v in S[f'{role}_sens'].items():
        print('  ', k, v)
    print('   no opp adj:', S[f'{role}_top10_noadj'])




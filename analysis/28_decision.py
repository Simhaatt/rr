"""Decision support built only from existing outputs: no new modelling.

1. Pressure profiles (phase, innings, opponent type) for the top 15 in each list and for Rajasthan Royals' 2026 squad, using the
   same calculation as 19_more.py (final ball values on pressure balls, per unit of leverage, x 1e4), with ball counts.
2. Role fit for the top 10: the split with the highest value among splits holding at least 100 pressure balls.
3. Two illustrative choices between near-equal players: their numbers side by side and how often one rates above the other
   across the 500 resamples (averaged over the three win-probability models, as in 22_contest.py).
4. Three worked deliveries (chase boundary, first-innings wicket, defended dot ball).
Usage: python 28_decision.py OUT_DIR DATA_DIR
"""
import os, sys, json, pickle
import numpy as np, pandas as pd
OUT, DATA = sys.argv[1], sys.argv[2]
MIN_ROLE = 100
b = pd.read_pickle(f'{OUT}/balls_scored.pkl')
CON = pickle.load(open(f'{OUT}/consensus.pkl', 'rb'))
R = pickle.load(open(f'{OUT}/results.pkl', 'rb'))
teams = pd.read_csv(f'{DATA}/teams.csv').drop_duplicates('team_id').set_index('team_id').team_name
mp = pd.read_csv(f'{DATA}/competition_team_player_mapping.csv')
rr = set(mp[(mp.team_name.str.contains('Rajasthan', na=False)) & (mp.comp_name == 'IPL 2026')].player_id)
res = {}


def profile(role, pids):
    pid = 'striker_id' if role == 'bat' else 'bowler_id'
    mask = b.bat_row.values if role == 'bat' else np.ones(len(b), bool)
    dd = b[mask & b[f'press_{role}'].values & b[pid].isin(pids).values]
    dd = dd.assign(val=dd[f'{role}_R'] + dd[f'{role}_K'],
                   opp=np.where(dd.pace_or_spin == 'Spin', 'v spin', 'v pace') if role == 'bat' else np.where(dd.batting_type == 'LHB', 'v left-handers', 'v right-handers'),
                   inn=np.where(dd.innings == 1, 'Setting' if role == 'bat' else 'Bowling first', 'Chasing' if role == 'bat' else 'Defending'),
                   ph=dd.phase.map({'PP': 'Powerplay', 'MID': 'Middle', 'DEATH': 'Death'}))
    out = {}
    for p in pids:
        d = dd[dd[pid] == p]; rec = {}
        for col in ('ph', 'inn', 'opp'):
            for k, r in d.groupby(col).agg(v=('val', 'sum'), L=('L', 'sum'), n=('val', 'size')).iterrows():
                rec[k] = [float(r.v / r.L * 1e4) if r.L > 0 else None, int(r.n)]
        out[p] = rec
    return out


def boots(role):
    C = CON[role]; pool = C.index
    bs = [pickle.load(open(f'{d}/results.pkl', 'rb'))['res'][role]['boot'].reindex(columns=pool) for d in (OUT, OUT + '_alt', OUT + '_gbm')]
    nb = min(len(x) for x in bs)
    return pd.DataFrame(sum(x.iloc[:nb].fillna(0).values for x in bs) / sum(x.iloc[:nb].notna().values for x in bs), columns=pool)


for role in ('bat', 'bowl'):
    C = CON[role].sort_values('rank_c'); t = R['res'][role]['table']
    keep = list(C.index[:15]) + [p for p in C.index if p in rr and p not in C.index[:15]]
    pr = profile(role, keep)
    B = boots(role)
    rows = []
    for p in keep:
        r = pr[p]
        splits = {k: v for k, v in r.items() if v[0] is not None}
        fit = max(((k, v) for k, v in splits.items() if v[1] >= MIN_ROLE and k in ('Powerplay', 'Middle', 'Death')), key=lambda kv: kv[1][0], default=(None, None))
        rows.append({'pid': int(p), 'name': C.name[p], 'rank': int(C.rank_c[p]), 'tier': C.tier[p] or 'Outside the contest', 'p_top10': float(C.p_top10_c[p]),
                     'rating': float(C.score_c[p]), 'lo': float(np.percentile(B[p] - (B[p].median() - C.score_c[p]), 5)), 'hi': float(np.percentile(B[p] - (B[p].median() - C.score_c[p]), 95)),
                     'n_P': int(t.n_P.get(p, 0)), 'cons': float(t.cball.get(p, np.nan) * 100), 'level': float(t.level_eb.get(p, np.nan) * 1e4),
                     'rr_2026': p in rr, 'splits': splits, 'role_fit': fit[0], 'role_fit_value': fit[1]})
    res[role] = rows
    res[role + '_ahead'] = lambda a, c, B=B: float((B[a] > B[c]).mean())

name_to = {r['name']: (role, r['pid']) for role in ('bat', 'bowl') for r in res[role]}
pairs = [('Suryakumar Yadav', 'Shreyas Iyer', 'Middle'), ('Shubman Gill', 'Yashasvi Jaiswal', 'Powerplay')]
dec = []
for a_, c_, phase in pairs:
    role, pa = name_to[a_]; _, pc = name_to[c_]
    ra = next(r for r in res[role] if r['pid'] == pa); rc = next(r for r in res[role] if r['pid'] == pc)
    dec.append({'a': ra, 'c': rc, 'phase': phase, 'a_ahead': res[role + '_ahead'](pa, pc)})
for role in ('bat', 'bowl'):
    del res[role + '_ahead']
res['decisions'] = dec

# worked deliveries
w = b.assign(bat_team=b.batting_team_id.map(teams), bowl_team=b.bowling_team_id.map(teams))
cand = {
    'A boundary in a tight chase': w[(w.innings == 2) & (w.is_six | w.is_four) & (w.balls_rem <= 12) & (w.wp_pre.between(0.2, 0.8))],
    'A wicket late in the first innings': w[(w.innings == 1) & w.is_wicket & (w.dismissed_player_id == w.striker_id) & (w.phase == 'DEATH') & (w.balls_rem >= 6)],
    'A dot ball while defending': w[(w.innings == 2) & (w.tot == 0) & (w.legal == 1) & ~w.is_wicket & (w.balls_rem.between(2, 12)) & (w.wp_pre.between(0.2, 0.8))],
}
ex = []
for lab, d in cand.items():
    r = d.loc[(-d.dwp).idxmax()] if 'boundary' not in lab else d.loc[(d.dwp * d.li).idxmax()]
    ex.append({'label': lab, 'match': f"{r.bat_team} v {r.bowl_team}", 'date': str(r.match_date)[:10], 'comp': r.comp_name, 'innings': int(r.innings),
               'striker': r.striker_name, 'bowler': r.bowler_name, 'balls_left': int(r.balls_rem), 'runs_needed': (None if r.innings == 1 else int(r.need)),
               'score': f"{int(r.runs_before)}/{int(r.wk_before)}", 'outcome': ('six' if r.is_six else 'four' if r.is_four else 'wicket' if r.is_wicket else 'dot ball' if r.tot == 0 else f"{int(r.tot)} runs"),
               'wp_pre': float(r.wp_pre * 100), 'wp_post': float(r.wp_post * 100), 'leverage': float(r.li), 'pressure': bool(r.press),
               'batter_value': float(r.dwp * 100), 'bowler_value': float(-r.dwp * 100)})
res['examples'] = ex
json.dump(res, open(f'{OUT}/decision_summary.json', 'w'), indent=1, default=str)
for role in ('bat', 'bowl'):
    print(role); print(pd.DataFrame([{k: v for k, v in r.items() if k not in ('splits',)} for r in res[role]]).round(2).to_string(index=False))
for d in dec:
    print(d['a']['name'], 'v', d['c']['name'], d['phase'], 'ahead', round(d['a_ahead'], 2), d['a']['splits'].get(d['phase']), d['c']['splits'].get(d['phase']))
for e in ex:
    print(e)

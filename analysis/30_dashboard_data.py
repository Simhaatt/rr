"""Data for the dashboard (docs/index.html): every rated player's numbers, splits, intervals and head-to-head odds.
Writes docs/data.js (window.DATA = {...}) so the page works from a file or from GitHub Pages without a server.
Usage: python 30_dashboard_data.py OUT_DIR DATA_DIR DOCS_DIR
"""
import sys, json, pickle
import numpy as np, pandas as pd
OUT, DATA, DOCS = sys.argv[1], sys.argv[2], sys.argv[3]
b = pd.read_pickle(f'{OUT}/balls_scored.pkl')
CON = pickle.load(open(f'{OUT}/consensus.pkl', 'rb'))
R = pickle.load(open(f'{OUT}/results.pkl', 'rb'))
S = json.load(open(f'{OUT}/summary.json')); LF = json.load(open(f'{OUT}/leakfree_summary.json')); EX = json.load(open(f'{OUT}/external_summary.json'))
DS = json.load(open(f'{OUT}/decision_summary.json'))
TY = json.load(open(f'{OUT}/types_summary.json'))
mp = pd.read_csv(f'{DATA}/competition_team_player_mapping.csv')
rr = set(mp[(mp.team_name.str.contains('Rajasthan', na=False)) & (mp.comp_name == 'IPL 2026')].player_id)
r2 = lambda v: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 2)


def splits(role):
    pid = 'striker_id' if role == 'bat' else 'bowler_id'
    mask = b.bat_row.values if role == 'bat' else np.ones(len(b), bool)
    dd = b[mask & b[f'press_{role}'].values]
    dd = dd.assign(val=dd[f'{role}_R'] + dd[f'{role}_K'],
                   opp=np.where(dd.pace_or_spin == 'Spin', 'v spin', 'v pace') if role == 'bat' else np.where(dd.batting_type == 'LHB', 'v left-handers', 'v right-handers'),
                   inn=np.where(dd.innings == 1, 'Setting' if role == 'bat' else 'Bowling first', 'Chasing' if role == 'bat' else 'Defending'),
                   ph=dd.phase.map({'PP': 'Powerplay', 'MID': 'Middle', 'DEATH': 'Death'}))
    out = {}
    for col in ('ph', 'inn', 'opp'):
        g = dd.groupby([pid, col]).agg(v=('val', 'sum'), L=('L', 'sum'), n=('val', 'size'))
        for (p, k), r in g.iterrows():
            out.setdefault(int(p), {})[k] = [r2(r.v / r.L * 1e4) if r.L > 0 else None, int(r.n)]
    return out


def boots(role):
    C = CON[role]; pool = C.index
    bs = [pickle.load(open(f'{d}/results.pkl', 'rb'))['res'][role]['boot'].reindex(columns=pool) for d in (OUT, OUT + '_alt', OUT + '_gbm')]
    nb = min(len(x) for x in bs)
    return pd.DataFrame(sum(x.iloc[:nb].fillna(0).values for x in bs) / sum(x.iloc[:nb].notna().values for x in bs), columns=pool)


D = {'meta': {'matches': S['matches_used'], 'ipl': S['ipl_matches'], 'intl': S['intl_matches'], 'balls': S['balls_used'], 'pressure_balls': S['press_balls'],
              'bat_n': S['bat_n'], 'bowl_n': S['bowl_n'], 'weights': R['W_ROLE'],
              'rel': {r: {'standard': r2(CON['summary'][r]['rel']), 'leakfree': r2(LF[r]['mean']['repeatable']), 'pred': r2(CON['summary'][r]['pred'])} for r in ('bat', 'bowl')},
              'clutch_p': {'bat': r2(S['checks']['bat_perm']['p_spread']), 'bowl': r2(S['checks']['bowl_perm']['p_spread'])},
              'vs_conv': {r: {k: {kk: r2(vv) for kk, vv in v.items()} for k, v in LF[r]['diff'].items()} for r in ('bat', 'bowl')},
              'external': {k: {r: {'rating': r2(EX[k][r]['rows']['Our rating']['r']), 'n': EX[k][r]['n_players']} for r in ('bat', 'bowl')} for k in ('season_2026', 'ipl_to_intl', 'intl_to_ipl')},
              'types': {k: v for k, v in TY.items() if k != 'types'}, 'examples': DS['examples'], 'decisions': [{'a': d['a']['name'], 'c': d['c']['name'], 'phase': d['phase'], 'a_ahead': r2(d['a_ahead'])} for d in DS['decisions']]}}
for role in ('bat', 'bowl'):
    C = CON[role].sort_values('rank_c'); t = R['res'][role]['table']; B = boots(role); sp = splits(role)
    players = []
    for p in C.index:
        x = t.loc[p]; sh = B[p].median() - C.score_c[p]
        conv = ({'sr_p': r2(x.SR_P), 'sr_n': r2(x.SR_N), 'avg_p': r2(x.Avg_P), 'avg_n': r2(x.Avg_N), 'runs_p': int(x.runs_P)} if role == 'bat' else
                {'econ_p': r2(x.Econ_P), 'econ_n': r2(x.Econ_N), 'bpw_p': r2(x.SR_P), 'bpw_n': r2(x.SR_N), 'wkts_p': int(x.wkts_P)})
        players.append({'id': int(p), 'name': C.name[p], 'rank': int(C.rank_c[p]), 'rating': r2(C.score_c[p]), 'lo': r2(np.percentile(B[p] - sh, 5)), 'hi': r2(np.percentile(B[p] - sh, 95)),
                        'tier': C.tier[p] or '', 'p_top10': r2(C.p_top10_c[p]), 'models_top10': int(C.n_models_top10[p]), 'n_p': int(x.n_P),
                        'usual': r2(x.T_rN * 1e4), 'pressure': r2(x.T_rP * 1e4), 'exam': r2(x.exam_rP * 1e4), 'level': r2(x.level_eb * 1e4), 'uplift': r2(x.uplift_eb * 1e4),
                        'cons': r2(x.cball * 100), 'ipl_share': r2(x.ipl_share_P), 'rr2026': bool(p in rr), 'btype': (TY['types'].get(str(int(p))) if role == 'bowl' else None), 'splits': sp.get(int(p), {}), **conv})
    top = list(C.index[:20])
    D[role] = {'players': players, 'h2h_ids': [int(p) for p in top], 'h2h': [[r2((B[a] > B[c]).mean()) for c in top] for a in top]}
open(f'{DOCS}/data.js', 'w', encoding='utf-8').write('window.DATA = ' + json.dumps(D, separators=(',', ':')) + ';\n')
print('players', len(D['bat']['players']), len(D['bowl']['players']))

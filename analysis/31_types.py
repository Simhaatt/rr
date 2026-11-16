"""Why so many spinners and fast scorers? Figures for report Section 6.4 and the dashboard.

Bowler type = the type (pace or spin) of most of the bowler's deliveries in the data.
Economy and balls per wicket on pressure balls, by phase and competition, for every bowler in the data.
Batters: strike rate of the top 10 against the eligible pool, and how closely the rating follows strike rate.
Usage: python 31_types.py OUT_DIR
"""
import sys, json, pickle
import numpy as np, pandas as pd
OUT = sys.argv[1]
b = pd.read_pickle(f'{OUT}/balls_scored.pkl')
CON = pickle.load(open(f'{OUT}/consensus.pkl', 'rb')); R = pickle.load(open(f'{OUT}/results.pkl', 'rb'))
typ = b.groupby('bowler_id').pace_or_spin.agg(lambda s: s.value_counts().idxmax())
C = CON['bowl'].sort_values('rank_c').assign(type=lambda d: typ.reindex(d.index))
out = {'types': {str(int(p)): t for p, t in typ.items()}}
out['bowl'] = {'pool': C.type.value_counts().to_dict(), 'top10': C.head(10).type.value_counts().to_dict(), 'top20': C.head(20).type.value_counts().to_dict(),
               'mean_rating': C.groupby('type').score_c.mean().round(3).to_dict(),
               'top_pace': [(C.name[p], int(C.rank_c[p])) for p in C[C.type == 'Pace'].index[:5]],
               'top_spin': [(C.name[p], int(C.rank_c[p])) for p in C[C.type == 'Spin'].index[:5]]}
p = b[b.press]


def econ(d, keys):
    g = d.groupby(keys).agg(runs=('bowl_runs', 'sum'), balls=('legal', 'sum'), wk=('bowl_wkt', 'sum'))
    return {('|'.join(k) if isinstance(k, tuple) else k): {'econ': round(float(r.runs / r.balls * 6), 2), 'bpw': round(float(r.balls / r.wk), 1), 'balls': int(r.balls)} for k, r in g.iterrows()}


out['econ_phase'] = econ(p, ['phase', 'pace_or_spin'])
out['econ_comp'] = econ(p, ['group', 'pace_or_spin'])
out['spin_share_press'] = round(float((p.pace_or_spin == 'Spin').mean()), 3)
out['press_chase_share'] = round(float((p.innings == 2).mean()), 3)
out['press_chase_mid_share'] = round(float(((p.innings == 2) & (p.phase == 'MID')).mean()), 3)
tb = R['res']['bat']['table']; Cb = CON['bat'].sort_values('rank_c')
sr = (tb.cr_P + tb.cr_N) / (tb.cb_P + tb.cb_N) * 100
out['bat'] = {'corr_rating_sr': round(float(np.corrcoef(Cb.score_c.reindex(tb.index), sr)[0, 1]), 2),
              'sr_top10': round(float(sr.reindex(Cb.index[:10]).mean()), 1), 'sr_pool': round(float(sr.mean()), 1),
              'sr_by_player': {Cb.name[p]: round(float(sr[p]), 1) for p in Cb.index[:10]}}
json.dump(out, open(f'{OUT}/types_summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'types'}, indent=1))

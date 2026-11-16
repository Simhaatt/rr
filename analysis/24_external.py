"""External validation: ratings built WITHOUT a season or competition, scored against what players did in it.

Season test     : rating built on 2022-2025 only (results_drop2026.pkl) -> actual pressure value in 2026.
Competition test: rating built on IPL only -> actual pressure value in internationals, and the reverse.
Target (same fixed yardstick as every held-out test in the report): opponent-adjusted win probability added on
pressure balls per unit of leverage (bat_T0 / bowl_T0 over L0, stakes-only pressure flag press0).
Note: the win-probability model itself was fitted on all seasons, but it uses only the match state, never player identity.
Usage: python 24_external.py OUT_DIR
"""
import sys, json, pickle
import numpy as np, pandas as pd
OUT = sys.argv[1]
MINB = 30                                       # pressure balls needed in the held-out season/competition to be scored
rng = np.random.default_rng(7)
b = pd.read_pickle(f'{OUT}/balls_scored.pkl')
IPL = b.comp_name.str.contains('IPL|Indian Premier', case=False, na=False)


def target(mask, role):
    d = b[mask & (b.bat_row if role == 'bat' else True)]
    pid = 'striker_id' if role == 'bat' else 'bowler_id'
    sp = d.press0.astype(float)
    g = pd.DataFrame({'pid': d[pid], 'v': d[f'{role}_T0'] * sp, 'l': d.L0 * sp, 'n': sp}).groupby('pid').sum()
    g = g[g.n >= MINB]
    return g.v / g.l, g.n


def conventional(mask, role):
    """Plain strike rate (batters) or economy, sign-flipped so higher is better (bowlers), from the training data."""
    if role == 'bat':
        d = b[mask & b.bat_row]
        g = d.groupby('striker_id').agg(r=('runs', 'sum'), n=('legal', 'sum'))
        return 100 * g.r / g.n.clip(lower=1)
    g = b[mask].groupby('bowler_id').agg(r=('bowl_runs', 'sum'), n=('legal', 'sum'))
    return -6 * g.r / g.n.clip(lower=1)


def corr_ci(x, y, nb=2000):
    x, y = np.asarray(x, float), np.asarray(y, float); n = len(x)
    bs = [np.corrcoef(x[i], y[i])[0, 1] for i in (rng.integers(0, n, n) for _ in range(nb))]
    return float(np.corrcoef(x, y)[0, 1]), float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def run(train_file, train_mask, test_mask, label):
    R = pickle.load(open(f'{OUT}/{train_file}', 'rb'))
    out = {}
    for role in ('bat', 'bowl'):
        t = R['res'][role]['table']
        y, ny = target(test_mask, role)
        conv = conventional(train_mask, role)
        common = t.index.intersection(y.index)
        rows = {}
        for nm, s in (('Our rating', t.score), ('Level only', t.level_eb), ('Own raw pressure record', t.T0_rP),
                      ('Strike rate' if role == 'bat' else 'Economy (lower is better)', conv)):
            c = common.intersection(s.dropna().index)
            rows[nm] = dict(zip(('r', 'lo', 'hi'), corr_ci(s.loc[c], y.loc[c]))) | {'n': int(len(c))}
        top = list(t.sort_values('score', ascending=False).index[:10])
        top_in = [p for p in top if p in y.index]
        rest = [p for p in common if p not in top]
        out[role] = dict(rows=rows, n_players=int(len(common)), top10_scored=len(top_in),
                         top10_above_avg=int((y.loc[top_in] > y.loc[common].mean()).sum()),
                         top10_mean=float(y.loc[top_in].mean() * 1e4), rest_mean=float(y.loc[rest].mean() * 1e4),
                         top10_names=[(t.name[p], round(float(y[p] * 1e4), 1), int(ny[p])) for p in top_in])
        print(f'\n== {label} | {role}: {len(common)} players with {MINB}+ pressure balls in the held-out data')
        for nm, v in rows.items():
            print(f'   {nm:32s} r = {v["r"]:+.2f}  (90% interval {v["lo"]:+.2f} to {v["hi"]:+.2f}, n = {v["n"]})')
        print(f'   our top 10 scored: {len(top_in)}; above the average player: {out[role]["top10_above_avg"]}; mean value {out[role]["top10_mean"]:+.1f} v rest {out[role]["rest_mean"]:+.1f}')
    return out


res = {'season_2026': run('results_drop2026.pkl', b.year < 2026, b.year == 2026, 'Built on 2022-2025, tested on 2026'),
       'ipl_to_intl': run('results_ipl.pkl', IPL, ~IPL, 'Built on IPL, tested on internationals'),
       'intl_to_ipl': run('results_intl.pkl', ~IPL, IPL, 'Built on internationals, tested on IPL'),
       'MINB': MINB}
json.dump(res, open(f'{OUT}/external_summary.json', 'w'), indent=1)

"""Design test: which ball weighting gives the most repeatable and most predictive pressure rating?
rate_a = sum(L^a * u) / sum(L^a), u = value per unit leverage.  a=1 is plain WPA/leverage; a=0 weights every ball equally."""
import sys, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
b = pd.read_pickle(sys.argv[1] + "/balls_scored.pkl")
rng = np.random.RandomState(5)
mids = b.match_id.unique()
splits = [pd.Series(rng.randint(0, 2, len(mids)), index=mids) for _ in range(80)]
FLOOR = 0.25
for role, pid, mask in (('bat', 'striker_id', b.bat_row), ('bowl', 'bowler_id', pd.Series(True, index=b.index))):
    d = b[mask].copy()
    v = d[f'{role}_R'] + d[f'{role}_K']
    Lf = d.L.clip(lower=FLOOR)
    d['u'] = v / Lf
    d['uR'] = d[f'{role}_R'] / Lf; d['uK'] = d[f'{role}_K'] / Lf
    cols = {}
    for a in (0.0, 0.5, 1.0):
        w = Lf ** a
        d[f'w{a}'] = w; d[f'v{a}'] = w * d.u; d[f'vR{a}'] = w * d.uR; d[f'vK{a}'] = w * d.uK
    if role == 'bat':
        d['cr'] = d.runs; d['cb'] = d.legal
    else:
        d['cr'] = -d.bowl_runs; d['cb'] = d.legal
    agg = {c: (c, 'sum') for c in d.columns if c[:1] in 'vw' and c[1:2] in '01RK' or c in ('cr', 'cb')}
    agg['n'] = ('u', 'size')
    g = d.groupby([pid, 'match_id', 'press']).agg(**agg).unstack('press', fill_value=0)
    g.columns = [f"{x}_{'P' if p else 'N'}" for x, p in g.columns]
    g = g.reset_index()
    tot = g.groupby(pid).agg(n=('n_P', 'sum'), m=('n_P', lambda x: (x > 0).sum()))
    el = tot.index[(tot.n >= 150) & (tot.m >= 15)]
    ge = g[g[pid].isin(el)]
    out = []
    for hh in splits:
        h = ge.match_id.map(hh).values
        q = [ge[h == k].groupby(pid).sum(numeric_only=True).reindex(el) for k in (0, 1)]
        r = []
        for x in q:
            o = pd.DataFrame(index=el)
            for a in (0.0, 0.5, 1.0):
                o[f'P{a}'] = x[f'v{a}_P'] / x[f'w{a}_P']; o[f'N{a}'] = x[f'v{a}_N'] / x[f'w{a}_N']
                o[f'A{a}'] = (x[f'v{a}_P'] + x[f'v{a}_N']) / (x[f'w{a}_P'] + x[f'w{a}_N'])
                o[f'RP{a}'] = x[f'vR{a}_P'] / x[f'w{a}_P']; o[f'KP{a}'] = x[f'vK{a}_P'] / x[f'w{a}_P']
                o[f'RA{a}'] = (x[f'vR{a}_P'] + x[f'vR{a}_N']) / (x[f'w{a}_P'] + x[f'w{a}_N'])
                o[f'KA{a}'] = (x[f'vK{a}_P'] + x[f'vK{a}_N']) / (x[f'w{a}_P'] + x[f'w{a}_N'])
            o['convP'] = x.cr_P / x.cb_P; o['convA'] = (x.cr_P + x.cr_N) / (x.cb_P + x.cb_N)
            r.append(o)
        ok = r[0].notna().all(axis=1) & r[1].notna().all(axis=1)
        row = {}
        for c in r[0].columns:
            row['rel_' + c] = np.corrcoef(r[0].loc[ok, c], r[1].loc[ok, c])[0, 1]
            # prediction of the other half's pressure WPA (a=1, the "what actually moved matches" target) and equal-weight target
            row['toP1_' + c] = 0.5 * (np.corrcoef(r[0].loc[ok, c], r[1].loc[ok, 'P1.0'])[0, 1] + np.corrcoef(r[1].loc[ok, c], r[0].loc[ok, 'P1.0'])[0, 1])
            row['toP0_' + c] = 0.5 * (np.corrcoef(r[0].loc[ok, c], r[1].loc[ok, 'P0.0'])[0, 1] + np.corrcoef(r[1].loc[ok, c], r[0].loc[ok, 'P0.0'])[0, 1])
        out.append(row)
    m = pd.DataFrame(out).mean()
    t = pd.DataFrame({k: m[[i for i in m.index if i.startswith(k + '_')]].set_axis([i.split('_', 1)[1] for i in m.index if i.startswith(k + '_')]) for k in ('rel', 'toP1', 'toP0')}).round(3)
    print(f"\n==== {role}  n={len(el)}\n", t.to_string())

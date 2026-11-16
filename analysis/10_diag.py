"""Why does plain scoring rate predict unseen pressure value better than the priced run component? Step-by-step variants."""
import os as _os
ROOT = _os.environ.get('RR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # repository root
import sys, importlib.util, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
spec = importlib.util.spec_from_file_location('r5', _os.path.join(ROOT, 'analysis', '05_ratings.py')); r5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r5)
b = r5.b
rng = np.random.RandomState(4)
mids = b.match_id.unique()
splits = [pd.Series(rng.randint(0, 2, len(mids)), index=mids) for _ in range(80)]
cell = [b.innings, (b.legal_before // 6).clip(upper=19), np.minimum(b.wk_before // 3, 2), b.press]
cell2 = [b.innings, (b.legal_before // 6).clip(upper=19), b.press]
for role, pid, mask, rc, wc in (('bat', 'striker_id', b.bat_row, 'runs', 'bat_out'), ('bowl', 'bowler_id', pd.Series(True, index=b.index), 'bowl_runs', 'bowl_wkt')):
    sg = 1 if role == 'bat' else -1
    d = b[mask].copy()
    r = d[rc].astype(float); w = d[wc].astype(float)
    V = {}   # name -> (numerator, denominator)
    one = pd.Series(1.0, index=d.index)
    V['runs/ball raw'] = (sg * r, d.legal.astype(float))
    for nm, c in (('cell', cell), ('cell no-wk', cell2), ('press only', [b.press])):
        cc = [x[mask] for x in c]
        rc_ = r - r.groupby(cc).transform('mean')
        V[f'runs centred {nm}'] = (sg * rc_, one)
        V[f'runs centred {nm} xL'] = (sg * rc_ * d.L, d.L)
    wc_ = w - w.groupby([x[mask] for x in cell]).transform('mean')
    V['wkts centred cell'] = (-sg * wc_, one)
    V['wkts raw'] = (-sg * w, d.legal.astype(float))
    V['R priced raw'] = (d[f'{role}_R_raw'], d.L); V['R priced adj'] = (d[f'{role}_R'], d.L)
    V['K priced raw'] = (d[f'{role}_K_raw'], d.L); V['K priced adj'] = (d[f'{role}_K'], d.L)
    V['T actual adj'] = (d[f'{role}_T'], d.L); V['T actual raw'] = (d[f'{role}_T_raw'], d.L)
    cols = {}
    for k, (n_, d_) in V.items():
        cols[k + '|n'] = n_; cols[k + '|d'] = d_
        cols[k + '|nP'] = n_ * d.press; cols[k + '|dP'] = d_ * d.press
    X = pd.DataFrame(cols); X[pid] = d[pid]; X['match_id'] = d.match_id; X['nP'] = d.press.astype(int)
    g = X.groupby([pid, 'match_id']).sum().reset_index()
    tot = g.groupby(pid).agg(n=('nP', 'sum'), m=('nP', lambda x: (x > 0).sum()))
    el = tot.index[(tot.n >= 150) & (tot.m >= 15)]
    ge = g[g[pid].isin(el)]
    out = []
    for hh in splits:
        h = ge.match_id.map(hh).values
        q = [ge[h == k].groupby(pid).sum(numeric_only=True).reindex(el) for k in (0, 1)]
        f = []
        for x in q:
            o = pd.DataFrame(index=el)
            for k in V:
                o[k + ' [all]'] = x[k + '|n'] / x[k + '|d']; o[k + ' [press]'] = x[k + '|nP'] / x[k + '|dP']
            f.append(o)
        ok = f[0].notna().all(axis=1) & f[1].notna().all(axis=1)
        row = {}
        for c in f[0].columns:
            row[('rel', c)] = np.corrcoef(f[0].loc[ok, c], f[1].loc[ok, c])[0, 1]
            for tn, t in (('->T adj press', 'T actual adj [press]'), ('->T raw press', 'T actual raw [press]')):
                row[(tn, c)] = 0.5 * (np.corrcoef(f[0].loc[ok, c], f[1].loc[ok, t])[0, 1] + np.corrcoef(f[1].loc[ok, c], f[0].loc[ok, t])[0, 1])
        out.append(row)
    m = pd.DataFrame(out).mean()
    m.index = pd.MultiIndex.from_tuples(m.index)
    print(f"\n==== {role}\n", m.unstack(0).round(3).to_string())

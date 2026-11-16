"""Design search: which pressure threshold / leverage cap gives the most repeatable player ratings?"""
import sys
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
b = pd.read_pickle(sys.argv[1] + "/balls_scored.pkl")
rng = np.random.RandomState(3)
mids = b.match_id.unique()
splits = [pd.Series(rng.randint(0, 2, len(mids)), index=mids) for _ in range(60)]
rows = []
for role, pid, vcol, mask in (('bat', 'striker_id', 'v_bat_adj', b.bat_row), ('bowl', 'bowler_id', 'v_bowl_adj', pd.Series(True, index=b.index))):
    d0 = b[mask]
    for cap in (None, 6, 4, 3, 2):
        f = 1.0 if cap is None else np.minimum(1.0, cap / d0.li.clip(lower=1e-9))
        d = d0.assign(V=d0[vcol] * f, L=d0.li * f)
        for pct in (0.67, 0.75, 0.80, 0.85, 0.90):
            thr = b.li.quantile(pct)
            d = d.assign(P=d.li >= thr)
            g = d.groupby([pid, 'match_id', 'P']).agg(V=('V', 'sum'), L=('L', 'sum'), n=('L', 'size')).unstack('P', fill_value=0)
            g.columns = [f"{a}_{'P' if p else 'N'}" for a, p in g.columns]
            g = g.reset_index()
            tot = g.groupby(pid).n_P.sum()
            for minb in (120, 200):
                el = tot.index[tot >= minb]
                ge = g[g[pid].isin(el)]
                cs = []
                for hh in splits:
                    h = ge.match_id.map(hh).values
                    q = [ge[h == k].groupby(pid)[['V_P', 'L_P', 'V_N', 'L_N']].sum().reindex(el) for k in (0, 1)]
                    r = [pd.DataFrame({'rP': x.V_P / x.L_P, 'rN': x.V_N / x.L_N, 'rA': (x.V_P + x.V_N) / (x.L_P + x.L_N)}) for x in q]
                    for x in r:
                        x['up'] = x.rP - x.rN
                    ok = r[0].notna().all(1) & r[1].notna().all(1)
                    cs.append([np.corrcoef(r[0].loc[ok, c], r[1].loc[ok, c])[0, 1] for c in ('rP', 'rN', 'rA', 'up')] + [np.corrcoef(r[0].loc[ok, 'rN'], r[1].loc[ok, 'rP'])[0, 1]])
                m = np.mean(cs, axis=0)
                rows.append(dict(role=role, cap=cap or 0, pct=pct, minb=minb, n=len(el), rP=m[0], rN=m[1], rAll=m[2], up=m[3], rN_to_rP=m[4]))
t = pd.DataFrame(rows).round(3)
print(t.to_string())
t.to_csv(sys.argv[1] + "/design_search.csv", index=False)

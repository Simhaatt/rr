"""Split-half reliability of value components (runs part vs wicket part of WPA), raw vs opponent-adjusted."""
import sys, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
b = pd.read_pickle(sys.argv[1] + "/balls_scored.pkl")
rng = np.random.RandomState(3)
mids = b.match_id.unique()
splits = [pd.Series(rng.randint(0, 2, len(mids)), index=mids) for _ in range(80)]
CAP = 3.0
f = np.minimum(1.0, CAP / b.li.clip(lower=1e-9))
b['L'] = b.li * f
# components, batting-side perspective
b['runs_part'] = b.dwp_nowk * f
b['wkt_part_bat'] = (b.v_bat - np.where(True, b.dwp_nowk, 0)) * f           # v_bat was centred; fine for correlation purposes
b['wkt_part_bowl'] = (b.c_bowl - b.dwp_nowk) * f
b['full_bat'] = b.v_bat * f; b['full_bat_adj'] = b.v_bat_adj * f
b['full_bowl'] = -b.c_bowl * f; b['full_bowl_adj'] = b.v_bowl_adj * f
b['runs_bowl'] = -b.runs_part; b['wk_bowl'] = -b.wkt_part_bowl
b['sr'] = b.runs * 1.0; b['one'] = 1.0


def rel(d, pid, col, den, pct, minb):
    thr = b.li.quantile(pct)
    d = d.assign(P=d.li >= thr, V=d[col], D=d[den])
    g = d.groupby([pid, 'match_id', 'P']).agg(V=('V', 'sum'), D=('D', 'sum'), n=('D', 'size')).unstack('P', fill_value=0)
    g.columns = [f"{a}_{'P' if p else 'N'}" for a, p in g.columns]
    g = g.reset_index()
    tot = g.groupby(pid).n_P.sum(); el = tot.index[tot >= minb]
    ge = g[g[pid].isin(el)]
    cs = []
    for hh in splits:
        h = ge.match_id.map(hh).values
        q = [ge[h == k].groupby(pid)[['V_P', 'D_P', 'V_N', 'D_N']].sum().reindex(el) for k in (0, 1)]
        r = [pd.DataFrame({'P': x.V_P / x.D_P, 'N': x.V_N / x.D_N, 'A': (x.V_P + x.V_N) / (x.D_P + x.D_N)}) for x in q]
        for x in r:
            x['U'] = x.P - x.N
        ok = r[0].notna().all(axis=1) & r[1].notna().all(axis=1)
        cs.append([np.corrcoef(r[0].loc[ok, c], r[1].loc[ok, c])[0, 1] for c in 'PNAU'] + [np.corrcoef(r[0].loc[ok, 'N'], r[1].loc[ok, 'P'])[0, 1]])
    return len(el), np.round(np.mean(cs, axis=0), 3)


for pct in (0.75, 0.80):
    for minb in (120, 200):
        print(f"--- pct {pct} minb {minb}   [n, (press, normal, all, uplift, normal->press)]")
        br = b[b.bat_row]
        for nm, col, den in [('bat full WPA raw', 'full_bat', 'L'), ('bat full WPA adj', 'full_bat_adj', 'L'), ('bat runs part', 'runs_part', 'L'), ('bat wicket part', 'wkt_part_bat', 'L'), ('bat strike rate', 'sr', 'one')]:
            print(f"  {nm:22s}", *rel(br, 'striker_id', col, den, pct, minb))
        for nm, col, den in [('bowl full WPA raw', 'full_bowl', 'L'), ('bowl full WPA adj', 'full_bowl_adj', 'L'), ('bowl runs part', 'runs_bowl', 'L'), ('bowl wicket part', 'wk_bowl', 'L'), ('bowl runs conceded', 'tot', 'one')]:
            print(f"  {nm:22s}", *rel(b, 'bowler_id', col, den, pct, minb))

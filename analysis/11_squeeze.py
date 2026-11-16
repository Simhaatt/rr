"""Idea 2 - Squeeze: does pressure that BUILDS (recent dots, a wicket just fallen, bowler being hit) change outcomes,
after removing who is batting/bowling and the match state? And is a player's response to it repeatable?"""
import sys, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
b = pd.read_pickle(sys.argv[1] + "/balls_scored.pkl")
g = b.groupby(['match_id', 'innings'], sort=False)
leg = b.legal.astype(float)
dot = ((b.tot == 0) & (b.legal == 1)).astype(float)
bdry = (b.is_four | b.is_six).astype(float)
wkt = b.is_wicket.astype(float)


def last_n_legal(x, n):
    """sum of x over the previous n legal balls of the innings (not counting this ball)."""
    cx = x.groupby([b.match_id, b.innings]).cumsum() - x            # total before this ball
    lb = b.legal_before
    key = pd.DataFrame({'m': b.match_id, 'i': b.innings, 'lb': lb, 'cx': cx})
    first = key.drop_duplicates(['m', 'i', 'lb']).set_index(['m', 'i', 'lb']).cx   # value at the moment legal_before == lb
    prev = pd.MultiIndex.from_arrays([b.match_id, b.innings, (lb - n).clip(lower=0)])
    return cx.values - first.reindex(prev).values


b['d6'] = last_n_legal(dot, 6); b['w6'] = last_n_legal(wkt, 6); b['b6'] = last_n_legal(bdry, 6)
b['faced'] = b.groupby(['match_id', 'innings', 'striker_id']).cumcount()
cell = [b.innings, (b.legal_before // 6).clip(upper=19), np.minimum(b.wk_before // 3, 2), b.press]
cc = lambda x: x - x.groupby(cell).transform('mean')
b['runs_c'] = cc(b.tot.astype(float)); b['wk_c'] = cc(wkt)
b['SQ_bat'] = ((b.d6 >= 4) | (b.w6 >= 1)) & (b.legal_before >= 6)      # batting side squeezed
b['SQ_bowl'] = (b.b6 >= 2) & (b.legal_before >= 6)                     # bowler under the pump
print("share of balls: squeezed batting side %.3f, bowler under the pump %.3f" % (b.SQ_bat.mean(), b.SQ_bowl.mean()))
print("share that are also stakes-pressure balls:", b[b.SQ_bat].press.mean().round(3), b[b.SQ_bowl].press.mean().round(3))
print("\nLeague, raw (context-centred) by dots in previous 6 legal balls:")
print(b[b.legal_before >= 6].groupby(b.d6.clip(upper=5)).agg(n=('tot', 'size'), runs_vs_ctx=('runs_c', 'mean'), wkt_vs_ctx=('wk_c', 'mean')).round(4))
print("after a wicket in previous 6:\n", b[b.legal_before >= 6].groupby('w6').agg(n=('tot', 'size'), runs_vs_ctx=('runs_c', 'mean'), wkt_vs_ctx=('wk_c', 'mean')).round(4).head(3))

OUTJ = {'share_bat': float(b.SQ_bat.mean()), 'share_bowl': float(b.SQ_bowl.mean()), 'd6': b[b.legal_before >= 6].groupby(b.d6.clip(upper=5)).runs_c.mean().round(4).tolist(), 'd6_n': b[b.legal_before >= 6].groupby(b.d6.clip(upper=5)).size().tolist()}
rng = np.random.RandomState(8)
mids = b.match_id.unique()
splits = [pd.Series(rng.randint(0, 2, len(mids)), index=mids) for _ in range(80)]
for role, pid, mask, flag in (('bat', 'striker_id', b.bat_row & (b.legal_before >= 6), 'SQ_bat'), ('bowl', 'bowler_id', b.legal_before >= 6, 'SQ_bowl')):
    d = b[mask]
    v = d[f'{role}_T0']                              # opponent-adjusted win probability added (not squeeze-adjusted), player's perspective
    sg = 1 if role == 'bat' else -1
    for scope, dm in (('all balls', np.ones(len(d), bool)), ('stakes-pressure balls only', d.press.values), ('ordinary balls only', ~d.press.values)):
        dd = d[dm]; vv = v[dm]
        a = pd.DataFrame({'pid': dd[pid].values, 'm': dd.match_id.values, 'S': dd[flag].values, 'v': vv.values, 'L': dd.L0.values,
                          'r': sg * dd.runs_c.values, 'w': -sg * dd.wk_c.values, 'one': 1.0})
        gm = a.groupby(['pid', 'm', 'S'])[['v', 'L', 'r', 'w', 'one']].sum().unstack('S', fill_value=0)
        gm.columns = [f"{x}_{'S' if s else 'N'}" for x, s in gm.columns]; gm = gm.reset_index()
        tot = gm.groupby('pid')[['one_S', 'one_N']].sum()
        el = tot.index[(tot.one_S >= 100) & (tot.one_N >= 100)]
        ge = gm[gm.pid.isin(el)]
        s = ge.groupby('pid').sum(numeric_only=True)
        diff = s.v_S / s.L_S - s.v_N / s.L_N
        wts = 1 / (1 / s.L_S + 1 / s.L_N)
        mean = np.average(diff, weights=wts)
        # match-clustered SE of the league-wide within-player effect
        gj = ge.join(s[['v_S', 'L_S', 'v_N', 'L_N']].rename(columns=lambda c: 'T' + c), on='pid').join(wts.rename('wt'), on='pid')
        contrib = gj.wt * ((gj.v_S - gj.Tv_S / gj.TL_S * gj.L_S) / gj.TL_S - (gj.v_N - gj.Tv_N / gj.TL_N * gj.L_N) / gj.TL_N) / wts.sum()
        se = np.sqrt((contrib.groupby(gj.m).sum() ** 2).sum())
        dr = np.average(s.r_S / s.one_S - s.r_N / s.one_N, weights=wts); dw = np.average(s.w_S / s.one_S - s.w_N / s.one_N, weights=wts)
        cs = []
        for hh in splits:
            h = ge.m.map(hh).values
            q = [ge[h == k].groupby('pid').sum(numeric_only=True).reindex(el) for k in (0, 1)]
            dq = [x.v_S / x.L_S - x.v_N / x.L_N for x in q]
            ok = dq[0].notna() & dq[1].notna() & np.isfinite(dq[0]) & np.isfinite(dq[1])
            cs.append(np.corrcoef(dq[0][ok], dq[1][ok])[0, 1])
        if scope == 'all balls':
            OUTJ[role] = dict(effect=float(mean * 1e4), se=float(se * 1e4), z=float(mean / se), runs=float(dr), players=int(len(el)), rel=float(np.mean(cs)))
        print(f"\n[{role}] {scope}: players {len(el)}; within-player effect of squeeze on value = {mean * 1e4:+.2f} pts/100 balls (SE {se * 1e4:.2f}, z={mean / se:+.1f});"
              f" runs/ball {dr:+.3f}, wicket-side {dw:+.4f}; split-half r of player response = {np.mean(cs):.3f}")
import json; json.dump(OUTJ, open(sys.argv[1] + '/squeeze_summary.json', 'w'), indent=1)



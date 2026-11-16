"""Checks called for by published practice on clutch measurement:
A. permutation null - is there any player-level pressure uplift beyond chance?
B. win-probability drift by WP bin (Mishra-style de-drifting check)
C. FanGraphs-style Clutch = WPA/pLI - sum(WPA_i/LI_i), as a cross-check on our uplift
E. how many pressure balls are needed for a reliability of 0.5"""
import sys, json, pickle, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
OUTD = sys.argv[1]; NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 400
b = pd.read_pickle(OUTD + "/balls_scored.pkl"); R = pickle.load(open(OUTD + "/results.pkl", 'rb'))
OUT = {}
rng = np.random.RandomState(12)

# ---------- B. drift
f0 = b.L0 / b.li.clip(lower=1e-9)
d = pd.DataFrame({'bin': (b.wp_pre // 0.02).clip(upper=49), 'v': b.dwp * f0, 'a': (b.dwp * f0).abs()})
g = d.groupby('bin').agg(m=('v', 'mean'), n=('v', 'size'), a=('a', 'mean'))
cont = g[(g.index >= 10) & (g.index < 40)]
OUT['drift'] = dict(max_abs=float(g.m.abs().max()), max_abs_contested=float(cont.m.abs().max()), mean_abs_weighted=float((g.m.abs() * g.n).sum() / g.n.sum()), mean_abs_swing=float(d.a.mean()),
                    overall=float(d.v.mean()))
print('B drift: mean WP change per ball overall %+.5f; largest bin drift %.4f (contested 0.2-0.8: %.4f); ball-weighted mean |drift| %.5f; for scale, mean |swing| per ball %.4f'
      % (OUT['drift']['overall'], OUT['drift']['max_abs'], OUT['drift']['max_abs_contested'], OUT['drift']['mean_abs_weighted'], OUT['drift']['mean_abs_swing']))

for role, pid, mask in (('bat', 'striker_id', b.bat_row.values), ('bowl', 'bowler_id', np.ones(len(b), bool))):
    t = R['res'][role]['table']; el = t.index
    dd = b[mask]; dd = dd[dd[pid].isin(el)]
    v = (dd[f'{role}_R'] + dd[f'{role}_K']).values; L = dd.L.values; P = dd[f'press_{role}'].values; ids = dd[pid].values
    order = np.argsort(ids, kind='stable'); v, L, P, ids = v[order], L[order], P[order], ids[order]
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    obs = []; null = []
    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        vv, ll, pp = v[s:e], L[s:e], P[s:e]
        up = lambda q: vv[q].sum() / ll[q].sum() - vv[~q].sum() / ll[~q].sum()
        obs.append(up(pp))
        null.append([up(rng.permutation(pp)) for _ in range(NPERM)])
    obs = np.array(obs) * 1e4; null = np.array(null) * 1e4          # players x perms
    z = (obs - null.mean(1)) / null.std(1)
    pv = ((np.abs(null - null.mean(1, keepdims=True)) >= np.abs(obs - null.mean(1))[:, None]).mean(1))
    sd_obs = obs.std(); sd_null = null.std(0)                        # spread between players, observed v each permutation
    p_var = float((sd_null >= sd_obs).mean())
    names = t.name.reindex(pd.unique(ids)).values
    sig = [(names[i], round(float(obs[i]), 1), round(float(pv[i]), 3)) for i in np.argsort(pv) if pv[i] < 0.05]
    OUT[f'{role}_perm'] = dict(players=int(len(obs)), sd_obs=float(sd_obs), sd_null_mean=float(sd_null.mean()), sd_null_hi=float(np.percentile(sd_null, 95)), p_spread=p_var,
                               n_sig=int((pv < 0.05).sum()), expected_by_chance=float(0.05 * len(obs)), n_sig_pos=int(((pv < 0.05) & (obs > null.mean(1))).sum()), sig=sig, z_sd=float(z.std()))
    print(f"A [{role}] spread of raw uplift between players: observed {sd_obs:.1f} v {sd_null.mean():.1f} under shuffled pressure labels (95th pct {np.percentile(sd_null, 95):.1f}); p = {p_var:.3f}."
          f" Players individually beyond their own null at 5%: {(pv < 0.05).sum()} of {len(obs)} (chance alone gives about {0.05 * len(obs):.0f}); z-score spread {z.std():.2f} (1.00 = pure chance)")
    print('   ', sig[:12])

    # ---------- C. FanGraphs-style clutch on actual win probability added (opponent-adjusted, in-match leverage)
    w = dd[f'{role}_T0'].values[order]; L0 = dd.L0.values[order]; mid = dd.match_id.values[order]
    df = pd.DataFrame({'pid': ids, 'm': mid, 'w': w, 'L0': L0, 'wl': w / np.clip(L0, 0.05, None), 'n': 1.0})
    def fg(x):
        s = x.groupby('pid')[['w', 'L0', 'wl', 'n']].sum()
        return (s.w / (s.L0 / s.n) - s.wl) / s.n * 1e4          # per 100 balls
    full = fg(df)
    upr = (t.T_rP - t.T_rN) * 1e4
    cs = []
    mids = pd.unique(mid)
    for _ in range(60):
        hh = pd.Series(rng.randint(0, 2, len(mids)), index=mids); h = df.m.map(hh).values
        a_, b_ = fg(df[h == 0]), fg(df[h == 1]); ok = a_.index.intersection(b_.index)
        cs.append(np.corrcoef(a_[ok], b_[ok])[0, 1])
    OUT[f'{role}_fg'] = dict(corr_raw_uplift=float(np.corrcoef(full.reindex(t.index), upr)[0, 1]), corr_shrunk_uplift=float(np.corrcoef(full.reindex(t.index), t.uplift_eb)[0, 1]),
                             corr_rating=float(np.corrcoef(full.reindex(t.index), t.score)[0, 1]), rel=float(np.mean(cs)),
                             top5=[(t.name[i], round(float(full[i]), 1), int(t['rank'][i])) for i in full.reindex(t.index).sort_values(ascending=False).index[:5]])
    print(f"C [{role}] FanGraphs-style clutch: correlation with our raw uplift {OUT[f'{role}_fg']['corr_raw_uplift']:.2f}, with shrunk uplift {OUT[f'{role}_fg']['corr_shrunk_uplift']:.2f}, with final rating {OUT[f'{role}_fg']['corr_rating']:.2f}; its own split-half r {np.mean(cs):.3f}")
    print('   top 5 on that formula (name, clutch per 100 balls, our rank):', OUT[f'{role}_fg']['top5'])

    # ---------- E. balls needed for reliability 0.5 (Spearman-Brown from the split-half numbers already computed)
    rel = R['res'][role]['rel']; half_balls = float(t.n_P.median() / 2)
    need = {k: (half_balls * (1 - rel[k]) / rel[k] if rel[k] > 0.005 else float('inf')) for k in ('score', 'level_eb', 'uplift_eb', 'cons_eb')}
    OUT[f'{role}_need'] = dict(half_balls=half_balls, **{k: (None if not np.isfinite(x) else float(x)) for k, x in need.items()})
    print(f"E [{role}] median pressure balls per player {t.n_P.median():.0f}; pressure balls needed for reliability 0.5 -> rating {need['score']:.0f}, level {need['level_eb']:.0f}, uplift {need['uplift_eb']:.0f}, consistency {need['cons_eb']:.0f}")
json.dump(OUT, open(OUTD + '/checks_summary.json', 'w'), indent=1)

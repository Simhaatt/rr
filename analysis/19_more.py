"""Further evidence (no effect on the rating):
P. power - how large would clutch differences have to be for the permutation test to find them?
Q. where each top-10 player's pressure value comes from (phase, innings, opponent type)
B. what changes under pressure league-wide (shot selection, delivery selection)
C. win-probability calibration by segment"""
import sys, json, pickle, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
OUTD = sys.argv[1]
b = pd.read_pickle(OUTD + "/balls_scored.pkl"); R = pickle.load(open(OUTD + "/results.pkl", 'rb'))
OUT = {}
rng = np.random.RandomState(77)

# ---------------- P. power of the permutation test
for role, pid, mask in (('bat', 'striker_id', b.bat_row.values), ('bowl', 'bowler_id', np.ones(len(b), bool))):
    t = R['res'][role]['table']
    dd = b[mask]; dd = dd[dd[pid].isin(t.index)]
    v = (dd[f'{role}_R'] + dd[f'{role}_K']).values; L = dd.L.values; P = dd[f'press_{role}'].values; ids = dd[pid].values
    o = np.argsort(ids, kind='stable'); v, L, P, ids = v[o], L[o], P[o], ids[o]
    st = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    segs = [(v[s:e], L[s:e], P[s:e]) for s, e in zip(st[:-1], st[1:])]
    def null_draw():
        out = []
        for vv, ll, pp in segs:
            q = rng.permutation(pp)
            out.append(vv[q].sum() / ll[q].sum() - vv[~q].sum() / ll[~q].sum())
        return np.array(out) * 1e4
    nulls = np.array([null_draw() for _ in range(300)])
    sd0 = nulls.std(1); thr = np.percentile(sd0, 95)
    level_sd = float(t.rA.std() * 1e4) if 'rA' in t else float('nan')
    res = {}
    for tau in (5, 10, 15, 20, 25, 30, 40):
        hits = [(nulls[rng.randint(len(nulls))] + rng.normal(0, tau, len(segs))).std() > thr for _ in range(1000)]
        res[tau] = float(np.mean(hits))
    need80 = next((k for k, pw in res.items() if pw >= 0.8), None)
    OUT[f'{role}_power'] = dict(null_sd=float(sd0.mean()), thr=float(thr), power=res, need80=need80, level_sd=level_sd, obs_sd=float(((np.array([vv[pp].sum() / ll[pp].sum() - vv[~pp].sum() / ll[~pp].sum() for vv, ll, pp in segs])) * 1e4).std()))
    print(f"P [{role}] chance spread {sd0.mean():.1f}; detection threshold {thr:.1f}; power by true clutch SD (pts/100 balls): {res}; 80% power needs SD >= {need80}; spread of overall level between players {level_sd:.1f}")

# ---------------- Q. profiles of the top 10
prof = {}
for role, pid, mask in (('bat', 'striker_id', b.bat_row.values), ('bowl', 'bowler_id', np.ones(len(b), bool))):
    t = R['res'][role]['table'].head(10)
    dd = b[mask & b[f'press_{role}'].values]
    dd = dd.assign(val=dd[f'{role}_R'] + dd[f'{role}_K'], opp=np.where(dd.pace_or_spin == 'Spin', 'v spin', 'v pace') if role == 'bat' else np.where(dd.batting_type == 'LHB', 'v left-handers', 'v right-handers'),
                   inn=np.where(dd.innings == 1, 'Setting' if role == 'bat' else 'Bowling first', 'Chasing' if role == 'bat' else 'Defending'),
                   ph=dd.phase.map({'PP': 'Powerplay', 'MID': 'Middle', 'DEATH': 'Death'}))
    rows = []
    for i, x in t.iterrows():
        d = dd[dd[pid] == i]; rec = {'name': x['name'], 'rank': int(x['rank'])}
        for col in ('ph', 'inn', 'opp'):
            g = d.groupby(col).agg(v=('val', 'sum'), L=('L', 'sum'), n=('val', 'size'))
            for k, r in g.iterrows():
                rec[k] = [float(r.v / r.L * 1e4) if r.L > 0 else None, int(r.n)]
        best = max(((k, v) for k, v in rec.items() if isinstance(v, list) and v[1] >= 40 and v[0] is not None), key=lambda kv: kv[1][0], default=(None, None))
        rec['best'] = best[0]
        rows.append(rec)
    prof[role] = rows
    cols = [c for c in ('Powerplay', 'Middle', 'Death', 'Setting', 'Chasing', 'Bowling first', 'Defending', 'v pace', 'v spin', 'v left-handers', 'v right-handers') if any(c in r for r in rows)]
    print(f"\nQ [{role}] pressure value per 100 balls (balls) by split")
    print(pd.DataFrame([{**{'name': r['name']}, **{c: (f"{r[c][0]:+.0f} ({r[c][1]})" if c in r else '-') for c in cols}, 'best': r['best']} for r in rows]).to_string(index=False))
OUT['profiles'] = prof

# ---------------- B. behaviour under pressure (context-adjusted: compared within the same innings and over)
DEF = ['Frontfoot defensive', 'Backfoot defensive', 'Push', 'Worked', 'Leg Glance', 'Steer', 'Defense', 'Intentional Padding', 'Glance']
st_ = b.stroke; pace = (b.pace_or_spin == 'Pace')
feats = {'Attacking shot (any non-defensive stroke)': ((~st_.isin(DEF)).astype(float), st_.notna() & b.bat_row),
         'Aerial/slog (slog, slog sweep, pick-up)': (st_.isin(['Slog', 'Slog Sweep', 'PickUp']).astype(float), st_.notna() & b.bat_row),
         'Innovative shot (scoop, reverse, switch, uppercut)': (st_.isin(['Paddle Scoop', 'Reverse Sweep', 'Switch', 'Reverse Paddle Scoop', 'UpperCut']).astype(float), st_.notna() & b.bat_row),
         'Dot ball': ((b.tot == 0).astype(float), b.legal == 1), 'Boundary': ((b.is_four | b.is_six).astype(float), b.legal == 1), 'Wicket': (b.is_wicket.astype(float), b.legal == 1),
         'Pace: yorker': (b.delivery_style.isin(['Yorker', 'Wide Yorker']).astype(float), pace), 'Pace: slower ball or cutter': (b.delivery_style.isin(['Off Cutter', 'Leg Cut', 'Knuckle', 'Back of Hand', 'Slow Bouncer']).astype(float), pace),
         'Pace: full toss': ((b.delivery_style == 'Full Toss').astype(float), pace), 'Wide or no-ball': ((b.legal == 0).astype(float), pd.Series(True, index=b.index))}
beh = []
cell = [b.innings, (b.legal_before // 6).clip(upper=19)]
for nm, (x, m) in feats.items():
    d = pd.DataFrame({'x': x[m], 'P': b.press0[m], 'i': b.innings[m], 'o': (b.legal_before[m] // 6).clip(upper=19), 'mt': b.match_id[m]})
    g = d.groupby(['i', 'o', 'P']).x.agg(['mean', 'size']).unstack('P')
    g = g[(g[('size', True)] >= 100) & (g[('size', False)] >= 100)]
    w = 1 / (1 / g[('size', True)] + 1 / g[('size', False)])
    diff = float(((g[('mean', True)] - g[('mean', False)]) * w).sum() / w.sum())
    # match-clustered bootstrap for the SE
    mids = d.mt.unique(); bs = []
    dm = d.groupby(['mt', 'i', 'o', 'P']).x.agg(['sum', 'size']).reset_index()
    for _ in range(100):
        ww = pd.Series(rng.poisson(1, len(mids)), index=mids)
        z = dm.assign(w=dm.mt.map(ww).values); z['s'] = z['sum'] * z.w; z['n'] = z['size'] * z.w
        gg = z.groupby(['i', 'o', 'P'])[['s', 'n']].sum().unstack('P')
        gg = gg.reindex(g.index).dropna()
        rate_p = gg[('s', True)] / gg[('n', True)]; rate_n = gg[('s', False)] / gg[('n', False)]
        w2 = 1 / (1 / gg[('n', True)] + 1 / gg[('n', False)])
        bs.append(float(((rate_p - rate_n) * w2).sum() / w2.sum()))
    beh.append(dict(name=nm, ordinary=float(d.x[~d.P].mean()), pressure=float(d.x[d.P].mean()), diff=diff, se=float(np.nanstd(bs))))
    print(f"B {nm:52s} ordinary {100 * d.x[~d.P].mean():5.1f}%  pressure {100 * d.x[d.P].mean():5.1f}%  same-over difference {100 * diff:+.2f} pts (SE {100 * np.nanstd(bs):.2f})")
OUT['behaviour'] = beh

# ---------------- C. calibration by segment
d = b[b.y != 0.5].assign(yy=lambda z: z.y.astype(int), p=lambda z: z.wp_pre.clip(1e-6, 1 - 1e-6))
m = d.drop_duplicates('match_id').set_index('match_id')
def seg(mask, lab):
    z = d[mask]
    ll = float(-(z.yy * np.log(z.p) + (1 - z.yy) * np.log(1 - z.p)).mean())
    bins = pd.cut(z.p, np.linspace(0, 1, 11)); gg = z.groupby(bins, observed=True).agg(p=('p', 'mean'), a=('yy', 'mean'), n=('p', 'size'))
    ece = float(((gg.p - gg.a).abs() * gg.n).sum() / gg.n.sum())
    return dict(segment=lab, matches=int(z.match_id.nunique()), balls=int(len(z)), pred=float(z.p.mean()), actual=float(z.yy.mean()), logloss=ll, ece=ece)
cal = [seg(np.ones(len(d), bool), 'All matches'), seg((d.group == 'IPL').values, 'IPL'), seg((d.group == 'INTL').values, 'Internationals')]
for y in sorted(d.year.unique()):
    cal.append(seg((d.year == y).values, str(y)))
cal += [seg((d.innings == 1).values, 'First innings'), seg((d.innings == 2).values, 'Chase'), seg(d.press0.values, 'Pressure balls only'), seg((d.knock_out.astype(int) == 1).values, 'Knockout matches')]
OUT['calibration'] = cal
print('\nC calibration by segment'); print(pd.DataFrame(cal).round(3).to_string(index=False))
json.dump(OUT, open(OUTD + '/more_summary.json', 'w'), indent=1)

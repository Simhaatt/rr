"""Consensus rating: the whole method run under three win-probability models (main blend, logistic only, boosted trees only),
each player's rating averaged across them. Resamples and splits are matched replicate by replicate (same seeds), so the consensus
gets its own bootstrap, split-half, chronological and season-to-season figures. Also: robustness re-runs, agreement, tiers."""
import os as _os
ROOT = _os.environ.get('RR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # repository root
import sys, json, pickle, os, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
MAIN = sys.argv[1]
DIRS = {'Main (blend)': _os.path.join(ROOT, 'output'), 'Logistic only': _os.path.join(ROOT, 'output_alt'), 'Boosted trees only': _os.path.join(ROOT, 'output_gbm')}
RES = {k: pickle.load(open(f"{d}/results.pkl", 'rb')) for k, d in DIRS.items()}
WTS = {'50/25/25 (base)': (0.5, 0.25, 0.25), '100/0/0': (1, 0, 0), '40/40/20': (0.4, 0.4, 0.2), '60/20/20': (0.6, 0.2, 0.2), '34/33/33': (0.34, 0.33, 0.33),
       '50/0/50': (0.5, 0, 0.5), '50/50/0': (0.5, 0.5, 0), '70/15/15': (.7, .15, .15)}
SFX = [('IPL matches only', '_ipl'), ('Internationals only', '_intl'), ('Without 2022', '_drop2022'), ('Without 2023', '_drop2023'), ('Without 2024', '_drop2024'),
       ('Without 2025', '_drop2025'), ('Without 2026', '_drop2026'), ('Pressure = top 33% of balls', '_pct0.67'), ('Pressure = top 20% of balls', '_pct0.80'),
       ('Eligibility: 100+ pressure balls', '_min100'), ('Eligibility: 200+ pressure balls', '_min200')]
OUT = {'models': list(DIRS), 'summary': {}}
cc = lambda a, b: float(pd.concat([a, b], axis=1).dropna().corr().iloc[0, 1])


def avg(frames):
    """mean across models of aligned Series/DataFrames, using whichever models have the player"""
    return pd.concat(frames, axis=1).mean(axis=1)


for role in ('bat', 'bowl'):
    tabs = {k: r['res'][role]['table'] for k, r in RES.items()}
    main = tabs['Main (blend)']; pool = main.index
    C = pd.DataFrame(index=pool); C['name'] = main.name
    for k, t in tabs.items():
        C[f'score|{k}'] = t.score.reindex(pool); C[f'rank|{k}'] = t['rank'].reindex(pool)
    C['score_c'] = C[[f'score|{k}' for k in tabs]].mean(axis=1)
    for z in ('z_level', 'z_uplift', 'z_cons'):
        C[z + '_c'] = avg([t[z].reindex(pool) for t in tabs.values()])
    C['rank_c'] = (-C.score_c).rank(method='first').astype(int)
    C['n_models_top10'] = sum((C[f'rank|{k}'] <= 10).astype(int) for k in tabs)
    # --- bootstrap, matched replicate by replicate
    boots = [r['res'][role]['boot'].reindex(columns=pool) for r in RES.values()]
    nb = min(len(x) for x in boots)
    B = sum(x.iloc[:nb].fillna(0).values for x in boots) / sum(x.iloc[:nb].notna().values for x in boots)
    rk = pd.DataFrame(B, columns=pool).rank(axis=1, ascending=False)
    C['p_top10_c'] = (rk <= 10).mean().values; C['rank_lo_c'] = rk.quantile(0.1).values; C['rank_hi_c'] = rk.quantile(0.9).values
    # --- weight sensitivity on the averaged pillars
    sens = pd.DataFrame({nm: (-(w[0] * C.z_level_c + w[1] * C.z_uplift_c + w[2] * C.z_cons_c)).rank() for nm, w in WTS.items()})
    C['sens_c'] = (sens <= 10).mean(axis=1)
    # --- split-half, matched split by split
    splits = [r['res'][role]['splits'] for r in RES.values()]
    ns = min(len(s) for s in splits); rel, pred, relm, predm = [], [], {k: [] for k in RES}, {k: [] for k in RES}
    for i in range(ns):
        # yardstick = actual pressure value measured with the MAIN model, the same yardstick as every other prediction figure in the report
        h = {j: (avg([s[i][j].score for s in splits]), splits[0][i][j].T0) for j in (0, 1)}
        rel.append(cc(h[0][0], h[1][0])); pred.append(0.5 * (cc(h[0][0], h[1][1]) + cc(h[1][0], h[0][1])))
        for k, s in zip(RES, splits):
            relm[k].append(cc(s[i][0].score, s[i][1].score)); predm[k].append(0.5 * (cc(s[i][0].score, h[1][1]) + cc(s[i][1].score, h[0][1])))
    # --- chronological and season to season (players filtered on the main model's pressure-ball counts)
    ch = [r['res'][role]['chrono'] for r in RES.values()]
    keep = [p for p in ch[0][0].index.intersection(ch[0][1].index) if ch[0][0].loc[p, 'nS'] >= 50 and ch[0][1].loc[p, 'nS'] >= 50]
    chrono = cc(avg([c[0].score for c in ch]).reindex(keep), avg([c[1].score for c in ch]).reindex(keep))
    yrs = [r['res'][role]['years'] for r in RES.values()]; ys = sorted(yrs[0]); acc, wts = [], []
    for y0, y1 in zip(ys[:-1], ys[1:]):
        if all(y0 in y and y1 in y for y in yrs):
            kp = [p for p in yrs[0][y0].index.intersection(yrs[0][y1].index) if yrs[0][y0].loc[p, 'nS'] >= 40 and yrs[0][y1].loc[p, 'nS'] >= 40]
            if len(kp) >= 15:
                acc.append(cc(avg([y[y0].score for y in yrs]).reindex(kp), avg([y[y1].score for y in yrs]).reindex(kp))); wts.append(len(kp))
    OUT['summary'][role] = dict(rel=float(np.mean(rel)), pred=float(np.mean(pred)), rel_models={k: float(np.mean(v)) for k, v in relm.items()}, pred_models={k: float(np.mean(v)) for k, v in predm.items()},
                                chrono=chrono, chrono_n=len(keep), yy=float(np.average(acc, weights=wts)) if wts else None,
                                n_sens_all=int((C.sort_values('rank_c').sens_c.head(10) == 1).sum()))
    # --- agreement with each single model
    top_c = list(C.sort_values('rank_c').index[:10])
    OUT['summary'][role]['agreement'] = {k: dict(overlap=int(sum(t['rank'].reindex(top_c) <= 10)), corr=cc(C.score_c, t.score.reindex(pool)),
                                                 out=[C.name[p] for p in top_c if not (t['rank'].get(p, 99) <= 10)]) for k, t in tabs.items()}
    # --- tiers
    C['tier'] = np.where((C.rank_c <= 10) & (C.n_models_top10 == 3) & (C.p_top10_c >= 0.6), 'Firm',
                         np.where((C.rank_c <= 10) | (C.p_top10_c >= 0.25), 'Contested', ''))
    OUT[role] = C.sort_values('rank_c')
    print(f"\n==== {role}: consensus split-half {np.mean(rel):.3f} (models {', '.join(f'{k} {np.mean(v):.3f}' for k, v in relm.items())}); predicts unseen {np.mean(pred):.3f}; chrono {chrono:.3f}; season-to-season {OUT['summary'][role]['yy']}")
    print(C.sort_values('rank_c').head(16)[['name', 'rank_c', 'score_c'] + [f'rank|{k}' for k in tabs] + ['n_models_top10', 'p_top10_c', 'rank_lo_c', 'rank_hi_c', 'sens_c', 'tier']].round(3).to_string())
    print('agreement:', OUT['summary'][role]['agreement'])

# --- robustness of the consensus, and of the main model alone, on the same eleven re-runs
rob_c, rob_m = [], []
for lab, sfx in SFX:
    V = {}
    for k, d in DIRS.items():
        f = f"{d}/results{sfx}.pkl"
        if os.path.exists(f):
            V[k] = pickle.load(open(f, 'rb'))
    if len(V) < len(DIRS):
        print('missing re-run', lab, [k for k in DIRS if k not in V]); continue
    rc, rm = [lab], [lab]
    for role in ('bat', 'bowl'):
        C = OUT[role]; top = list(C.index[:10]); vm = V['Main (blend)']['res'][role]['table']
        vc = avg([v['res'][role]['table'].score.reindex(vm.index) for v in V.values()])
        vtop = set(vc.sort_values(ascending=False).index[:10]); elig = [p for p in top if p in vm.index]
        rc += [int(len(vm)), len(elig), int(sum(p in vtop for p in elig)), cc(C.score_c, vc), [C.name[p] for p in elig if p not in vtop]]
        mt = RES['Main (blend)']['res'][role]['table']; mtop = list(mt.index[:10]); mel = [p for p in mtop if p in vm.index]; vmt = set(vm.index[:10])
        rm += [int(len(vm)), len(mel), int(sum(p in vmt for p in mel)), cc(mt.score, vm.score), [mt.name[p] for p in mel if p not in vmt]]
    rob_c.append(rc); rob_m.append(rm)
OUT['robust'] = rob_c; OUT['robust_main'] = rob_m
for nm, rb in (('consensus', rob_c), ('main model alone', rob_m)):
    if rb:
        print(f"\nrobustness, {nm}: batters keep {np.mean([r[3] / max(r[2], 1) for r in rb]):.0%}, bowlers keep {np.mean([r[8] / max(r[7], 1) for r in rb]):.0%}")
        for r in rb:
            print('  ', r[0], '| bat', r[3], 'of', r[2], r[5], '| bowl', r[8], 'of', r[7], r[10])
pickle.dump(OUT, open(f"{MAIN}/consensus.pkl", 'wb'))
for role in ('bat', 'bowl'):
    OUT[role].to_csv(f"{MAIN}/{role}_consensus_ratings.csv")

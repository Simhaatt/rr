"""Match importance: do players perform differently in knockout matches (finals, qualifiers, eliminators, semi-finals)?
Same within-player design as the squeeze test. Also: how much knockout evidence is there per player?"""
import sys, json, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
b = pd.read_pickle(sys.argv[1] + "/balls_scored.pkl")
b['KO'] = b.knock_out.astype(int) == 1
b['press'] = b.press0        # in-match stakes only, before any match-importance weight
m = b.drop_duplicates('match_id')
OUT = {'ko_matches': int(m.KO.sum()), 'matches': int(len(m)), 'ko_balls': int(b.KO.sum()), 'ko_press_balls': int((b.KO & b.press).sum()),
       'press_share_ko': float(b[b.KO].press.mean()), 'press_share_other': float(b[~b.KO].press.mean()),
       'rpo_ko': float(6 * b[b.KO].tot.sum() / b[b.KO].legal.sum()), 'rpo_other': float(6 * b[~b.KO].tot.sum() / b[~b.KO].legal.sum()),
       'wk_ko': float(b[b.KO].is_wicket.sum() / b[b.KO].legal.sum()), 'wk_other': float(b[~b.KO].is_wicket.sum() / b[~b.KO].legal.sum())}
print(OUT)
for role, pid, mask in (('bat', 'striker_id', b.bat_row), ('bowl', 'bowler_id', pd.Series(True, index=b.index))):
    d = b[mask]
    cnt = d[d.KO & d.press].groupby(pid).size()
    allp = d[d.press].groupby(pid).size(); el = allp.index[allp >= 150]
    c = cnt.reindex(el).fillna(0)
    OUT[f'{role}_elig_ko_press_median'] = float(c.median()); OUT[f'{role}_elig_ko_press_50plus'] = int((c >= 50).sum()); OUT[f'{role}_elig_ko_press_100plus'] = int((c >= 100).sum())
    print(role, 'eligible players: knockout pressure balls median', c.median(), '| with 50+:', (c >= 50).sum(), '| with 100+:', (c >= 100).sum(), '| max', c.max())
    for scope, dm in (('all balls', np.ones(len(d), bool)), ('pressure balls', d.press.values)):
        dd = d[dm]
        a = pd.DataFrame({'pid': dd[pid].values, 'm': dd.match_id.values, 'S': dd.KO.values, 'v': dd[f'{role}_T0'].values, 'L': dd.L0.values, 'one': 1.0})
        gm = a.groupby(['pid', 'm', 'S'])[['v', 'L', 'one']].sum().unstack('S', fill_value=0)
        gm.columns = [f"{x}_{'S' if s else 'N'}" for x, s in gm.columns]; gm = gm.reset_index()
        tot = gm.groupby('pid')[['one_S', 'one_N']].sum()
        el2 = tot.index[(tot.one_S >= 30) & (tot.one_N >= 100)]
        ge = gm[gm.pid.isin(el2)]; s = ge.groupby('pid').sum(numeric_only=True)
        diff = s.v_S / s.L_S - s.v_N / s.L_N; wts = 1 / (1 / s.L_S + 1 / s.L_N); mean = np.average(diff, weights=wts)
        gj = ge.join(s[['v_S', 'L_S', 'v_N', 'L_N']].rename(columns=lambda c: 'T' + c), on='pid').join(wts.rename('wt'), on='pid')
        contrib = gj.wt * ((gj.v_S - gj.Tv_S / gj.TL_S * gj.L_S) / gj.TL_S - (gj.v_N - gj.Tv_N / gj.TL_N * gj.L_N) / gj.TL_N) / wts.sum()
        se = np.sqrt((contrib.groupby(gj.m).sum() ** 2).sum())
        OUT[f'{role}_{scope}'] = dict(players=int(len(el2)), effect=float(mean * 1e4), se=float(se * 1e4), z=float(mean / se))
        print(f"  [{role}] {scope}: {len(el2)} players; same player in knockouts v other matches = {mean * 1e4:+.1f} pts/100 balls (SE {se * 1e4:.1f}, z={mean / se:+.1f})")
json.dump(OUT, open(sys.argv[1] + '/knockout_summary.json', 'w'), indent=1)



"""Tests of further ideas: home v away, and form slump (risk of being dropped). Same within-player design as the squeeze test."""
import sys, json, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
OUTD = sys.argv[1]
b = pd.read_pickle(OUTD + "/balls_scored.pkl"); x = pd.read_pickle(OUTD + "/matches.pkl")
if 'match_id' in x.columns:
    x = x.set_index('match_id')
print('fielding-related columns:', [c for c in b.columns if any(k in c.lower() for k in ('field', 'catch', 'drop', 'keeper'))])
m = x.reindex(b.match_id.values)
home_id = m.home_team_id.values; vc = m.venue_country.astype(str).values; ht = m.home_team.astype(str).values; at = m.away_team.astype(str).values
ipl = (b.group == 'IPL').values
# venue status of each side: IPL -> designated home side; internationals -> the side whose country hosts (else neutral)
home_side_is_home = np.where(ipl, True, ht == vc); away_side_is_home = np.where(ipl, False, at == vc)
bat_is_homeside = b.batting_team_id.values == home_id
b['bat_home'] = np.where(bat_is_homeside, home_side_is_home, away_side_is_home)
b['bat_away'] = np.where(bat_is_homeside, away_side_is_home, home_side_is_home)       # opponent is at home
b['bowl_home'] = b.bat_away; b['bowl_away'] = b.bat_home
print('share of balls: batter at home %.3f, away %.3f, neutral %.3f' % (b.bat_home.mean(), b.bat_away.mean(), 1 - b.bat_home.mean() - b.bat_away.mean()))

# form slump: batter's previous 3 innings total under 30 runs; bowler's previous 3 spells went for 10+ an over combined
inn = b[b.bat_row].groupby(['striker_id', 'match_id']).agg(r=('runs', 'sum'), d=('match_date', 'first')).reset_index().sort_values(['striker_id', 'd', 'match_id'])
g = inn.groupby('striker_id').r
inn['prev3'] = g.shift(1) + g.shift(2) + g.shift(3)
inn['slump'] = inn.prev3 < 30
b = b.merge(inn[['striker_id', 'match_id', 'slump', 'prev3']].rename(columns={'slump': 'bat_slump', 'prev3': 'bat_prev3'}), on=['striker_id', 'match_id'], how='left')
sp = b.groupby(['bowler_id', 'match_id']).agg(r=('bowl_runs', 'sum'), n=('legal', 'sum'), d=('match_date', 'first')).reset_index().sort_values(['bowler_id', 'd', 'match_id'])
gr, gn = sp.groupby('bowler_id').r, sp.groupby('bowler_id').n
sp['pr'] = gr.shift(1) + gr.shift(2) + gr.shift(3); sp['pn'] = gn.shift(1) + gn.shift(2) + gn.shift(3)
sp['slump'] = ((6 * sp.pr / sp.pn) >= 10).astype(object)
sp.loc[sp.pn.isna(), 'slump'] = np.nan
inn['slump'] = inn.slump.astype(object); inn.loc[inn.prev3.isna(), 'slump'] = np.nan
b = b.merge(sp[['bowler_id', 'match_id', 'slump']].rename(columns={'slump': 'bowl_slump'}), on=['bowler_id', 'match_id'], how='left')
print('share of batter innings in a slump: %.3f; bowler spells after expensive run: %.3f' % (inn.slump[inn.prev3.notna()].mean(), sp.slump.dropna().astype(bool).mean()))

rng = np.random.RandomState(9); mids = b.match_id.unique()
splits = [pd.Series(rng.randint(0, 2, len(mids)), index=mids) for _ in range(60)]
OUT = {}


def within(role, pid, d, flagS, flagN, label, minS=100, minN=100):
    a = pd.DataFrame({'pid': d[pid].values, 'm': d.match_id.values, 'v': d[f'{role}_T0'].values, 'L': d.L0.values, 'S': flagS, 'N': flagN})
    a = a[a.S | a.N]
    gm = a.assign(vS=a.v * a.S, LS=a.L * a.S, nS=a.S.astype(float), vN=a.v * a.N, LN=a.L * a.N, nN=a.N.astype(float)).groupby(['pid', 'm'])[['vS', 'LS', 'nS', 'vN', 'LN', 'nN']].sum().reset_index()
    tot = gm.groupby('pid')[['nS', 'nN']].sum(); el = tot.index[(tot.nS >= minS) & (tot.nN >= minN)]
    ge = gm[gm.pid.isin(el)]; s = ge.groupby('pid').sum(numeric_only=True)
    diff = s.vS / s.LS - s.vN / s.LN; wts = 1 / (1 / s.LS + 1 / s.LN); mean = np.average(diff, weights=wts)
    gj = ge.join(s.rename(columns=lambda c: 'T' + c), on='pid').join(wts.rename('wt'), on='pid')
    contrib = gj.wt * ((gj.vS - gj.TvS / gj.TLS * gj.LS) / gj.TLS - (gj.vN - gj.TvN / gj.TLN * gj.LN) / gj.TLN) / wts.sum()
    se = np.sqrt((contrib.groupby(gj.m).sum() ** 2).sum())
    cs = []
    for hh in splits:
        h = ge.m.map(hh).values
        q = [ge[h == k].groupby('pid').sum(numeric_only=True).reindex(el) for k in (0, 1)]
        dq = [z.vS / z.LS - z.vN / z.LN for z in q]
        ok = np.isfinite(dq[0]) & np.isfinite(dq[1])
        cs.append(np.corrcoef(dq[0][ok], dq[1][ok])[0, 1])
    OUT[label] = dict(players=int(len(el)), effect=float(mean * 1e4), se=float(se * 1e4), z=float(mean / se), rel=float(np.nanmean(cs)), sd_gap=float(diff.std() * 1e4))
    print(f"{label}: {len(el)} players; same player = {mean * 1e4:+.1f} pts/100 balls (SE {se * 1e4:.1f}, z={mean / se:+.1f}); spread of individual gaps {diff.std() * 1e4:.0f}; split-half r of individual gap = {np.nanmean(cs):.3f}")
    return diff


for role, pid, mask in (('bat', 'striker_id', b.bat_row.values), ('bowl', 'bowler_id', np.ones(len(b), bool))):
    d = b[mask]
    within(role, pid, d, d[f'{role}_away'].values.astype(bool), d[f'{role}_home'].values.astype(bool), f'{role}: away v home, all balls')
    dp = d[d.press0.values]
    within(role, pid, dp, dp[f'{role}_away'].values.astype(bool), dp[f'{role}_home'].values.astype(bool), f'{role}: away v home, pressure balls', 50, 50)
    sl = d[f'{role}_slump']
    within(role, pid, d, (sl == True).values, (sl == False).values, f'{role}: after a form slump v otherwise, all balls')
json.dump(OUT, open(OUTD + '/insights_summary.json', 'w'), indent=1)

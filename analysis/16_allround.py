"""Same-game all-round impact: how often does a player add win probability with BOTH bat and ball in one match, and does it repeat?"""
import sys, json, pickle, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
OUTD = sys.argv[1]; MINB = 6
b = pd.read_pickle(OUTD + "/balls_scored.pkl"); R = pickle.load(open(OUTD + "/results.pkl", 'rb'))
bt = b[b.bat_row].groupby(['striker_id', 'match_id']).agg(bv=('bat_T0', 'sum'), bn=('legal', 'sum'), runs=('runs', 'sum')).rename_axis(['pid', 'match_id'])
bw = b.groupby(['bowler_id', 'match_id']).agg(wv=('bowl_T0', 'sum'), wn=('legal', 'sum'), wk=('bowl_wkt', 'sum')).rename_axis(['pid', 'match_id'])
j = bt.join(bw, how='outer').fillna(0).reset_index()
j['both'] = (j.bn >= MINB) & (j.wn >= MINB)
j['dbl'] = j.both & (j.bv > 0) & (j.wv > 0)
j['classic'] = (j.runs >= 30) & (j.wk >= 2)
print('player-matches', len(j), '| batted and bowled 6+ balls each:', int(j.both.sum()), '| positive with both:', int(j.dbl.sum()), '| 30+ runs and 2+ wickets:', int(j.classic.sum()), '| 50+ and 2+:', int(((j.runs >= 50) & (j.wk >= 2)).sum()))
d = j[j.both]
print('when a player does both: share positive with bat %.3f, with ball %.3f, with both %.3f (independent would be %.3f); corr of bat and ball value in the same match %.3f'
      % ((d.bv > 0).mean(), (d.wv > 0).mean(), d.dbl.mean(), (d.bv > 0).mean() * (d.wv > 0).mean(), np.corrcoef(d.bv, d.wv)[0, 1]))
s = j.groupby('pid').agg(matches=('both', 'size'), both=('both', 'sum'), dbl=('dbl', 'sum'), classic=('classic', 'sum'))
names = R['res']['bat']['table'].name.to_dict(); names.update(R['res']['bowl']['table'].name.to_dict())
rng = np.random.RandomState(4); mids = b.match_id.unique(); cs = []
pool = s.index[s.both >= 10]
for _ in range(60):
    hh = pd.Series(rng.randint(0, 2, len(mids)), index=mids); h = j.match_id.map(hh).values
    q = [j[h == k].groupby('pid').agg(both=('both', 'sum'), dbl=('dbl', 'sum')).reindex(pool).fillna(0) for k in (0, 1)]
    r = [x.dbl / x.both.replace(0, np.nan) for x in q]; ok = r[0].notna() & r[1].notna()
    cs.append(np.corrcoef(r[0][ok], r[1][ok])[0, 1])
print('players with 10+ both-discipline matches:', len(pool), '| split-half r of their double-impact share: %.3f' % np.mean(cs))
for role in ('bat', 'bowl'):
    t = R['res'][role]['table']
    x = t[['name', 'rank', 'score', 'app']].join(s)
    x['dbl_share_of_matches'] = (x.dbl / x.matches).round(3)
    print(f"\n{role}: eligible players with any both-discipline match: {(x.both > 0).sum()} of {len(x)}; with 5+ double-impact matches: {(x.dbl >= 5).sum()}")
    print(x[x.dbl >= 3].sort_values('dbl', ascending=False)[['name', 'rank', 'score', 'matches', 'both', 'dbl', 'classic', 'dbl_share_of_matches']].head(20).to_string())
json.dump(dict(player_matches=int(len(j)), both=int(j.both.sum()), dbl=int(j.dbl.sum()), classic30=int(j.classic.sum()), classic50=int(((j.runs >= 50) & (j.wk >= 2)).sum()),
               pos_bat=float((d.bv > 0).mean()), pos_ball=float((d.wv > 0).mean()), pos_both=float(d.dbl.mean()), indep=float((d.bv > 0).mean() * (d.wv > 0).mean()),
               corr=float(np.corrcoef(d.bv, d.wv)[0, 1]), rel=float(np.mean(cs)), pool=int(len(pool))), open(OUTD + '/allround_summary.json', 'w'), indent=1)

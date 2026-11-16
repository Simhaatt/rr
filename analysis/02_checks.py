import sys
import pandas as pd, numpy as np
D = sys.argv[1]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
comp = pd.read_csv(f"{D}/competitions.csv")
m = pd.read_csv(f"{D}/matches.csv").merge(comp, on='comp_id', how='left')
b = pd.read_csv(f"{D}/ball_by_ball.csv", low_memory=False)
b = b[b.innings <= 2].sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)
# team list with match counts
t = pd.concat([m[['home_team', 'match_format']].rename(columns={'home_team': 'team'}), m[['away_team', 'match_format']].rename(columns={'away_team': 'team'})])
print(t.groupby(['match_format', 'team']).size().sort_values(ascending=False).to_string())
print(m[m.match_format == 'T20'].groupby('comp_name').size())
# is "other" a no-ball? count legal balls per over under both hypotheses
b['tot'] = b.runs + b.extras1_runs + b.extras_2_runs
for name, illegal in [('wide+Overstep', ['wide', 'Overstep']), ('wide+Overstep+other', ['wide', 'Overstep', 'other'])]:
    b['legal'] = ~b.extras_type.isin(illegal)
    ov = b.groupby(['match_id', 'innings', 'overid']).legal.sum()
    print(name, ov.value_counts().sort_index().to_dict())
print(b[b.extras_type == 'other'][['match_id', 'overid', 'over_ball_num', 'runs', 'extras1_runs', 'extras_2_runs', 'is_wicket', 'dismissal_type']].head(10))
print(b[b.extras_type == 'Penalty'])
# duplicates / ordering
print("dup ball nums", b.duplicated(['match_id', 'innings', 'innings_ball_num']).sum())
g = b.groupby(['match_id', 'innings'])
print("ball num gaps", (g.innings_ball_num.max() != g.size()).sum())
# wickets on wides / no-balls
print(b[b.is_wicket].groupby(b.extras_type.fillna('none')).dismissal_type.value_counts())
print("dismissed != striker:", (b.is_wicket & (b.dismissed_player_id != b.striker_id)).sum())
print(b[b.is_wicket & (b.dismissed_player_id != b.striker_id)].dismissal_type.value_counts())
# innings completeness
b['legal'] = ~b.extras_type.isin(['wide', 'Overstep', 'other'])
inn = g.agg(runs=('tot', 'sum'), wk=('is_wicket', 'sum'), legal=('legal', 'sum')).reset_index()
i1 = inn[inn.innings == 1].set_index('match_id'); i2 = inn[inn.innings == 2].set_index('match_id')
x = m.set_index('match_id').join(i1.add_suffix('_1')).join(i2.add_suffix('_2'))
x = x[x.runs_1.notna()]
x['dls'] = x.first_inn_revised_overs.notna() | x.second_inn_revised_overs.notna() | x.revised_target.notna()
x['i1_complete'] = (x.legal_1 == 120) | (x.wk_1 == 10)
x['chase_won'] = x.runs_2 > x.runs_1
x['i2_complete'] = (x.legal_2 == 120) | (x.wk_2 == 10) | x.chase_won
print(pd.crosstab([x.dls, x.i1_complete], [x.i2_complete.fillna(False), x.win_team_id.notna()]))
odd = x[(~x.dls) & (~(x.i1_complete & x.i2_complete.fillna(False)))]
print(odd[['comp_name', 'result', 'runs_1', 'wk_1', 'legal_1', 'runs_2', 'wk_2', 'legal_2', 'win_team_id']].to_string())
print(x[x.runs_1 == x.runs_2][['result', 'win_team_id', 'runs_1', 'runs_2', 'legal_2']])
# which team batted first vs winner consistency
bf = b[b.innings == 1].groupby('match_id').batting_team_id.first()
x['bat_first'] = bf
ok = x[(~x.dls) & x.i1_complete & x.i2_complete.fillna(False) & x.win_team_id.notna() & (x.runs_1 != x.runs_2)]
print("consistent winner:", ((ok.win_team_id == ok.bat_first) == (ok.runs_1 > ok.runs_2)).mean(), len(ok))
bad = ok[((ok.win_team_id == ok.bat_first) != (ok.runs_1 > ok.runs_2))]
print(bad[['comp_name', 'result', 'runs_1', 'runs_2', 'wk_2', 'legal_2', 'win_team_id', 'bat_first', 'home_team_id', 'away_team_id']].to_string())
print(m.groupby(['match_format']).match_date.agg(['min', 'max']))

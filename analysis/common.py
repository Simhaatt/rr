"""Shared loading + state construction for the pressure pipeline."""
import pandas as pd, numpy as np

ILLEGAL = ['wide', 'Overstep', 'other']  # 'other' behaves as a no-ball (over gets an extra delivery)


def load(D):
    comp = pd.read_csv(f"{D}/competitions.csv")
    m = pd.read_csv(f"{D}/matches.csv").merge(comp, on='comp_id', how='left')
    b = pd.read_csv(f"{D}/ball_by_ball.csv", low_memory=False)
    b = b[b.innings <= 2].sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)
    b['tot'] = b.runs + b.extras1_runs + b.extras_2_runs
    b['legal'] = (~b.extras_type.isin(ILLEGAL)).astype(int)
    b['wk'] = b.is_wicket.astype(int)
    g = b.groupby(['match_id', 'innings'])
    b['legal_before'] = g.legal.cumsum() - b.legal
    b['runs_before'] = g.tot.cumsum() - b.tot
    b['wk_before'] = g.wk.cumsum() - b.wk
    b['balls_rem'] = 120 - b.legal_before

    inn = g.agg(runs=('tot', 'sum'), wk=('wk', 'sum'), legal=('legal', 'sum')).reset_index()
    i1 = inn[inn.innings == 1].set_index('match_id').add_suffix('_1')
    i2 = inn[inn.innings == 2].set_index('match_id').add_suffix('_2')
    x = m.set_index('match_id').join(i1).join(i2)
    x['bat_first'] = b[b.innings == 1].groupby('match_id').batting_team_id.first()
    x['dls'] = x.first_inn_revised_overs.notna() | x.second_inn_revised_overs.notna() | x.revised_target.notna()
    x['i1_complete'] = (x.legal_1 == 120) | (x.wk_1 == 10)
    x['tie'] = x.runs_1 == x.runs_2
    x['i2_complete'] = (x.legal_2 == 120) | (x.wk_2 == 10) | (x.runs_2 > x.runs_1)
    x['good'] = (~x.dls) & x.i1_complete & x.i2_complete.fillna(False) & (x.win_team_id.notna() | x.tie)
    x['group'] = np.where(x.comp_name.str.startswith('IPL'), 'IPL', 'INTL')
    x['year'] = pd.to_datetime(x.match_date).dt.year
    # scoring environment ("par"): mean completed first-innings total for the competition group and year
    par = x[x.good].groupby(['group', 'year']).runs_1.mean().rename('par')
    x = x.join(par, on=['group', 'year'])
    # first-innings win label (1 = side batting first won, 0.5 = tie in regulation)
    x['bf_win'] = np.where(x.tie, 0.5, (x.runs_1 > x.runs_2).astype(float))

    b = b[b.match_id.isin(x.index[x.good])].copy()
    b = b.join(x[['par', 'runs_1', 'bf_win', 'group', 'year', 'comp_name', 'match_date', 'stage', 'knock_out']], on='match_id')
    b['need'] = np.where(b.innings == 2, b.runs_1 + 1 - b.runs_before, np.nan)
    b['y'] = np.where(b.innings == 1, b.bf_win, 1 - b.bf_win)  # batting side won
    b['phase'] = pd.cut(b.overid, [0, 6, 15, 20], labels=['PP', 'MID', 'DEATH']).astype(str)
    return b.reset_index(drop=True), x

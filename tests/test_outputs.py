"""Automated checks on the data reconstruction and the final outputs.

Run after run_all.py:   python tests/test_outputs.py      (or: pytest tests/)
Writes output/verification.json: dataset counts, exclusions, pressure balls, eligibility, model metrics, final lists,
random seeds and package versions, with the result of every check.
"""
import os, sys, json, pickle, re, importlib.metadata as md
import numpy as np, pandas as pd

ROOT = os.environ.get('RR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'output'); DATA = os.path.join(ROOT, 'data_pack', 'Data Pack'); SUB = os.path.join(ROOT, 'submission')
_cache = {}


def balls():
    if 'b' not in _cache:
        _cache['b'] = pd.read_pickle(os.path.join(OUT, 'balls_scored.pkl'))
    return _cache['b']


def summary():
    return json.load(open(os.path.join(OUT, 'summary.json')))


def consensus():
    return pickle.load(open(os.path.join(OUT, 'consensus.pkl'), 'rb'))


def test_match_counts():
    m = pd.read_csv(os.path.join(DATA, 'matches.csv')); bb = pd.read_csv(os.path.join(DATA, 'ball_by_ball.csv'), usecols=['match_id'])
    S = summary(); b = balls()
    assert len(m) == S['matches_total'] == 883
    assert bb.match_id.nunique() == S['matches_with_bbb'] == 867
    assert b.match_id.nunique() == S['matches_used'] == 803
    assert len(b) == S['balls_used'] == 191903


def test_innings_reconstruction_gives_recorded_winner():
    b = balls()
    tot = b.groupby(['match_id', 'innings']).tot.sum().unstack()
    res = b.groupby('match_id').bf_win.first()
    d = tot.join(res).dropna(subset=[1, 2])
    decided = d[(d.bf_win.isin([0, 1])) & (d[1] != d[2])]
    assert len(decided) == 794
    assert ((decided[1] > decided[2]).astype(int) == decided.bf_win).all()


def test_legal_balls_per_over():
    """Counting wides, oversteps and 'other' extras as illegal makes 31,757 of 32,338 overs exactly six balls long (report, Section 2);
    the rest are unfinished overs, apart from one seven-ball over (an umpire's miscount recorded as bowled)."""
    r = pd.read_csv(os.path.join(DATA, 'ball_by_ball.csv'), usecols=['match_id', 'innings', 'overid', 'extras_type'])
    r = r[r.innings <= 2].assign(legal=~r.extras_type.isin(['wide', 'Overstep', 'other']))
    per_over = r.groupby(['match_id', 'innings', 'overid']).legal.sum()
    assert len(per_over) == 32338 and (per_over == 6).sum() == 31757 and (per_over > 6).sum() == 1
    assert balls().groupby(['match_id', 'innings', 'overid']).legal.sum().gt(6).sum() <= 1


def test_wickets_per_innings():
    b = balls()
    assert b.groupby(['match_id', 'innings']).is_wicket.sum().max() <= 10


def test_pressure_threshold():
    b = balls(); S = summary()
    assert abs(b.li.quantile(0.75) - S['thr']) < 1e-9
    assert abs(b.press0.mean() - 0.25) < 0.001                  # stakes-only flag: exactly the top quarter
    assert (b.press == (b.li * b.mw >= S['thr'])).all()          # knockout-weighted flag uses the same bar
    assert int(b.press.sum()) == S['press_balls']


def test_eligibility():
    R = pickle.load(open(os.path.join(OUT, 'results.pkl'), 'rb'))
    for role in ('bat', 'bowl'):
        t = R['res'][role]['table']
        assert (t.nS >= R['MIN_PBALLS']).all() and (t.pmS >= R['MIN_PMATCH']).all()
        assert len(t) == 97


def final_lists():
    C = consensus()
    return {r: list(C[r].sort_values('rank_c').name.head(10)) for r in ('bat', 'bowl')}


def test_lists_agree_everywhere():
    L = final_lists()
    md_ = open(os.path.join(SUB, 'SUBMISSION_top10_lists.md'), encoding='utf-8').read()
    rows = re.findall(r'^\| (\d+) \| (.+?) \| (.+?) \|$', md_, re.M)
    assert [r[1] for r in rows] == L['bat'] and [r[2] for r in rows] == L['bowl']
    for role in ('bat', 'bowl'):
        c = pd.read_csv(os.path.join(SUB, f'{role}_consensus_ratings.csv'))
        assert list(c.sort_values('rank_c').name.head(10)) == L[role]
    docx_path = os.path.join(SUB, 'SupeRR_Selector_R1_Methodology_Report.docx')
    if os.path.exists(docx_path):
        import docx
        t = docx.Document(docx_path).tables[0]
        assert [r.cells[1].text for r in t.rows[1:]] == L['bat'] and [r.cells[2].text for r in t.rows[1:]] == L['bowl']


def test_no_player_repeated():
    L = final_lists()
    assert len(set(L['bat'])) == 10 and len(set(L['bowl'])) == 10


def test_names_exist_in_data_pack():
    m = pd.read_csv(os.path.join(DATA, 'competition_team_player_mapping.csv'))
    names = set(m.player_name)
    for role, L in final_lists().items():
        assert all(n in names for n in L)


def write_verification():
    S = summary(); C = consensus()
    checks = {}
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn(); checks[name] = 'pass'
            except AssertionError as e:
                checks[name] = f'FAIL {e}'
    pk = ['numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib', 'python-docx']
    ver = {'python': sys.version.split()[0], **{p: md.version(p) for p in pk}}
    v = {'checks': checks,
         'data': {k: S[k] for k in ('matches_total', 'matches_with_bbb', 'matches_used', 'ipl_matches', 'intl_matches', 'balls_used', 'dls_excluded')},
         'pressure': {'threshold': S['thr'], 'pressure_balls': S['press_balls']},
         'eligible': {'batters': S['bat_n'], 'bowlers': S['bowl_n']},
         'win_probability': {'first_innings': S['wp1'], 'chase': S['wp2']},
         'reliability': {r: {k: C['summary'][r][k] for k in ('rel', 'pred', 'yy')} for r in ('bat', 'bowl')},
         'final_lists': {r: [{'rank': int(C[r].rank_c[p]), 'name': C[r].name[p], 'rating': round(float(C[r].score_c[p]), 4), 'tier': C[r].tier[p]} for p in C[r].sort_values('rank_c').index[:10]] for r in ('bat', 'bowl')},
         'random_seeds': {'05_ratings resamples and splits': 11, '12_harness design tests': 21, '24_external intervals': 7, '27_leakfree splits': '1000 to 1099', 'win-probability folds': 'fixed in 04_pipeline.py'},
         'packages': ver}
    json.dump(v, open(os.path.join(OUT, 'verification.json'), 'w'), indent=1, default=str)
    return checks


if __name__ == '__main__':
    res = write_verification()
    for k, v in res.items():
        print(f'{v[:4]:4s}  {k}' + (f'  {v}' if v != 'pass' else ''))
    sys.exit(0 if all(v == 'pass' for v in res.values()) else 1)

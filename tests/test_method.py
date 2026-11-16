"""Tests of the method's core steps, each checked against its defining property on the pipeline's own outputs.

Run after run_all.py:   python tests/test_method.py      (or: pytest tests/)
"""
import os, sys, json, pickle
import numpy as np, pandas as pd

ROOT = os.environ.get('RR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'output')
_c = {}


def balls():
    if 'b' not in _c:
        _c['b'] = pd.read_pickle(os.path.join(OUT, 'balls_scored.pkl'))
    return _c['b']


def results(d='output'):
    if d not in _c:
        _c[d] = pickle.load(open(os.path.join(ROOT, d, 'results.pkl'), 'rb'))
    return _c[d]


# ---- win probability
def test_win_probability_is_out_of_fold():
    """Every match sits in exactly one of five folds, so it is predicted by a model fitted on the other four."""
    f = balls().groupby('match_id').fold.nunique()
    assert (f == 1).all() and balls().fold.nunique() == 5


def test_win_probability_in_range():
    b = balls()
    assert b.wp_pre.between(0, 1).all() and b.wp_post.between(0, 1).all()


def test_wicket_almost_never_helps_batting_side():
    """The wicket part of a ball's value (actual change minus the change without the wicket) should not be positive. The chase model
    is constrained so it never is; the first-innings model is not, and on 116 of 10,297 wickets (1.1%), all late in a first innings,
    it rates the wicket as slightly helping the batting side (median 0.9 points). The report states this (7.6); this test pins it."""
    b = balls(); w = b[b.is_wicket]; d = w.dwp - w.dwp_nowk; bad = w[d > 1e-9]
    assert len(w) == 10297 and len(bad) == 116 and (bad.innings == 1).all()
    assert abs(d[d > 1e-9].median() * 100 - 0.9) < 0.05


# ---- leverage and pressure
def test_leverage_scaled_to_average_one():
    assert abs(balls().li.mean() - 1.0) < 0.01


def test_pressure_flag_uses_documented_bar():
    b = balls(); R = results()
    thr = b.li.quantile(R['PCT'])
    assert ((b.li * b.mw >= thr) == b.press).all()


def test_leverage_cap():
    """No ball's weight exceeds the cap of three times the average ball."""
    b = balls(); R = results()
    assert R['CAP'] == 3.0 and (b.L0 <= R['CAP'] + 1e-9).all()


# ---- attribution
def test_wides_not_charged_to_batter():
    b = balls()
    assert not b.loc[b.extras_type == 'wide', 'bat_row'].any()


def test_non_striker_run_out_not_charged_to_striker():
    b = balls(); x = b[b.is_wicket & (b.dismissed_player_id != b.striker_id)]
    assert len(x) > 0 and (x.Kw_bat == 0).all()


def test_run_outs_not_credited_to_bowler():
    b = balls(); x = b[b.dismissal_type == 'RunOut']
    assert len(x) > 0 and (x.bowl_wkt == 0).all()


# ---- shrinkage and rating
def test_shrinkage_pulls_towards_average():
    """Shrunk overall levels spread less than the raw rates they come from, and every reliability lies between 0 and 1."""
    for role in ('bat', 'bowl'):
        t = results()['res'][role]['table']
        assert t.base_eb.std() < t.rA.std()
        for c in ('R_rA_rel', 'K_rA_rel', 'up_tot_rel'):
            assert t[c].between(0, 1).all()


def test_rating_is_weighted_sum_of_parts():
    R = results()
    for role in ('bat', 'bowl'):
        t = R['res'][role]['table']; w = R['W_ROLE'][role]
        calc = w['level'] * t.z_level + w['uplift'] * t.z_uplift + w['cons'] * t.z_cons
        assert np.allclose(calc, t.score, atol=1e-9)


def test_final_rating_is_average_of_three_models():
    C = pickle.load(open(os.path.join(OUT, 'consensus.pkl'), 'rb'))
    for role in ('bat', 'bowl'):
        c = C[role]
        assert np.allclose(c[[f'score|{m}' for m in C['models']]].mean(axis=1), c.score_c, atol=1e-9)


def test_resampling_is_by_match_and_complete():
    """500 resamples per list; the resampling weights are drawn per match (see 05_ratings.py), so balls from one match move together."""
    src = open(os.path.join(ROOT, 'analysis', '05_ratings.py'), encoding='utf-8').read()
    assert "rng.poisson(1.0, len(mids))" in src and "index=mids" in src
    for role in ('bat', 'bowl'):
        assert len(results()['res'][role]['boot']) == 500


def test_eligibility_uses_counts_not_outcomes():
    src = open(os.path.join(ROOT, 'analysis', '05_ratings.py'), encoding='utf-8').read()
    assert "el = list(s0.index[(s0.nS >= MIN_PBALLS) & (s0.pmS >= MIN_PMATCH)])" in src


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn(); print(f'pass  {name}')
            except AssertionError as e:
                fails += 1; print(f'FAIL  {name}  {e}')
    sys.exit(1 if fails else 0)

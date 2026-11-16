"""Pressure ratings for batters and bowlers from balls.pkl. Writes CSVs + results.pkl"""
import sys, pickle, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 300)
OUT = sys.argv[1]
b = pd.read_pickle(f"{OUT}/balls.pkl")
import os as _os
SFX = _os.environ.get('SFX', '')                    # robustness runs write to suffixed files and never touch the main outputs
if _os.environ.get('ONLY'):
    b = b[b.group == _os.environ['ONLY']].reset_index(drop=True)
if _os.environ.get('DROPYEAR'):
    b = b[b.year != int(_os.environ['DROPYEAR'])].reset_index(drop=True)
if _os.environ.get('HALF'):        # leakage-free tests (27_leakfree.py): keep one random half of the matches BEFORE anything is fitted
    _seed, _h = map(int, _os.environ['HALF'].split(','))
    _m = np.sort(b.match_id.unique()); _hh = np.random.RandomState(_seed).randint(0, 2, len(_m))
    b = b[b.match_id.isin(_m[_hh == _h])].reset_index(drop=True)
NSPLIT = int(_os.environ.get('NSPLIT', '100'))

PCT = float(_os.environ.get('PCT', '0.75'))            # pressure = top quartile of deliveries by leverage
CAP = 3.0             # leverage cap (x average ball) so no single delivery dominates a rating
K_OPP = 120.0         # shrinkage (leverage-balls) for opponent-quality estimates
MIN_PBALLS = int(_os.environ.get('MIN_PBALLS', '150'))
MIN_PMATCH = 15
MIN_APP = 4           # min pressure balls in a match for it to count as a "pressure appearance"
W = dict(level=0.50, uplift=0.25, cons=0.25)
# WEIGHTS='brief' (default): 50/25/25 for both lists. WEIGHTS='learned': weights fitted on unseen matches (23_weights.py), per list.
WEIGHTS = _os.environ.get('WEIGHTS', 'brief')
W_ROLE = {'bat': dict(level=0.68, uplift=0.0, cons=0.32), 'bowl': dict(level=0.38, uplift=0.0, cons=0.62)} if WEIGHTS == 'learned' else {'bat': dict(W), 'bowl': dict(W)}
ZMODE = __import__('os').environ.get('ZMODE', 'raw')
NBOOT = int(__import__('os').environ.get('NBOOT', '500'))
COMP = ['R', 'K']     # R = run component, K = wicket component
import os
CENTRE = os.environ.get('CENTRE', 'felt')        # 'press': compare with the average player on the same kind of ball; 'felt': also within the same squeeze state
PRESSMODE = os.environ.get('PRESSMODE', 'stakes')  # 'stakes': top-quartile leverage; 'felt': stakes OR (squeezed with above-average stakes)
VALUE = os.environ.get('VALUE', 'blend')              # 'wp': actual win probability moved; 'exam': standardised paper; 'blend': mean of the two
UPMODE = os.environ.get('UPMODE', 'binary')        # 'binary': pressure rate - ordinary rate; 'slope': pressure response curve
LI_SQ = 1.0
UPTOT = os.environ.get('UPTOT', 'total')            # 'total': shrink the whole uplift by its own reliability; 'comp': shrink run and wicket parts separately

# ---------------- match importance: a ball in a knockout carries more than its in-match leverage
KOW = float(os.environ.get('KOW', '0.5'))            # Final counts (1 + KOW) times; semi-final, qualifier, eliminator (1 + KOW / 2) times
# graded by how big the match is. With KOW = 0.5: IPL or Asia Cup final x1.5; IPL qualifier / eliminator x1.25; tri-series and A-team knockouts x1.125
_major = (b.group == 'IPL') | b.comp_name.isin(['Asia Cup 2025', 'Mens Twenty20 Asia Cup 2022'])
_tier = np.where(b.knock_out.astype(int) != 1, 0.0, np.where(~_major, 0.25, np.where(b.stage == 'Final', 1.0, 0.5)))
b['mw'] = 1 + KOW * _tier
f0 = np.minimum(1.0, CAP / b.li.clip(lower=1e-9))
b['L0'] = b.li * f0; thr0 = b.li.quantile(PCT); b['press0'] = b.li >= thr0      # unweighted versions: the fixed yardstick and eligibility
b['lie'] = b.li * b.mw                                                         # effective stakes = leverage x match importance
thr = thr0                                                                     # same bar for every match, so weighting knockouts changes nothing for other balls
b['press'] = b.lie >= thr
f = f0 * b.mw
b['L'] = b.li * f
b['x'] = np.log(b.L.clip(lower=0.2))               # log-leverage, the horizontal axis of the response curve
GAPX = float((b.x * b.L)[b.press].sum() / b.L[b.press].sum() - (b.x * b.L)[~b.press].sum() / b.L[~b.press].sum())


# ---------------- squeeze: pressure that builds from what has just happened
def last_n_legal(x, n):
    """sum of x over the previous n legal balls of the innings (not counting this ball)."""
    cx = x.groupby([b.match_id, b.innings]).cumsum() - x
    key = pd.DataFrame({'m': b.match_id, 'i': b.innings, 'lb': b.legal_before, 'cx': cx})
    first = key.drop_duplicates(['m', 'i', 'lb']).set_index(['m', 'i', 'lb']).cx
    prev = pd.MultiIndex.from_arrays([b.match_id, b.innings, (b.legal_before - n).clip(lower=0)])
    return cx.values - first.reindex(prev).values


b['d6'] = last_n_legal(((b.tot == 0) & (b.legal == 1)).astype(float), 6)
b['w6'] = last_n_legal(b.is_wicket.astype(float), 6)
b['b6'] = last_n_legal((b.is_four | b.is_six).astype(float), 6)
b['SQ_bat'] = ((b.d6 >= 4) | (b.w6 >= 1)) & (b.legal_before >= 6)      # batting side squeezed: 4+ dots in the last 6 balls, or a wicket in them
b['SQ_bowl'] = (b.b6 >= 2) & (b.legal_before >= 6)                     # bowler under the pump: 2+ boundaries in the last 6 balls
if PRESSMODE == 'felt':
    b['press_bat'] = b.press | (b.SQ_bat & (b.li >= LI_SQ)); b['press_bowl'] = b.press | (b.SQ_bowl & (b.li >= LI_SQ))
else:
    b['press_bat'] = b.press; b['press_bowl'] = b.press

# ---------------- attribution (batting-side perspective until the end)
nonbowler = b.dismissal_type.isin(['RunOut', 'Retire Out', 'Obstracting the field', 'Obstructing the field'])
striker_out = b.is_wicket & (b.dismissed_player_id == b.striker_id)
b['bat_row'] = b.extras_type != 'wide'
b['bowl_wkt'] = (b.is_wicket & ~nonbowler).astype(int)
b['bat_out'] = striker_out.astype(int)
b['bowl_runs'] = b.tot - np.where(b.extras_type.isin(['byes', 'legbyes', 'Penalty']), b.extras1_runs, 0)
# actual win probability moved on each ball (the observed outcome), split into runs part and wicket part
b['Rw'] = b.dwp_nowk * f                                              # value of the runs / ball used
b['Kw_bat'] = np.where(striker_out, b.dwp - b.dwp_nowk, 0.0) * f      # value lost through the striker's dismissal
b['Kw_bowl'] = np.where(b.is_wicket & ~nonbowler, b.dwp - b.dwp_nowk, 0.0) * f

ALL = pd.Series(True, index=b.index)
WKB = np.minimum(b.wk_before // 3, 2)
MATCHUP = os.environ.get('MATCHUP', '1') == '1'     # opponent quality measured against this TYPE of opponent (batter v pace/spin; bowler v left/right-hander)
VENUE = os.environ.get('VENUE', '0') == '1'         # remove the venue's own tilt towards bat or ball
CONSMODE = os.environ.get('CONSMODE', 'ball')      # 'match': share of positive pressure appearances; 'ball': share of pressure balls not lost
K_MU = 300.0; K_V = 600.0
_mx = pd.read_pickle(f"{OUT}/matches.pkl")
if 'match_id' in _mx.columns:
    _mx = _mx.set_index('match_id')
b['venue_id'] = _mx.venue_id.reindex(b.match_id.values).values
_spin = (b.pace_or_spin == 'Spin').astype(int); _lhb = (b.batting_type == 'LHB').astype(int)


def cells(role, exam=False):
    c = [b[f'press_{role}']]
    if CENTRE == 'felt':
        c += [b.SQ_bat, b.SQ_bowl]
    if exam:
        c += [b.innings, b.phase, WKB]
    return c


def centre(col, rowmask, cell, Lc='L'):
    """Subtract what an average player produces on the same kind of ball."""
    num = b[col].where(rowmask, 0.0).groupby(cell).transform('sum')
    den = b[Lc].where(rowmask, 0.0).groupby(cell).transform('sum')
    out = b[col] - (num / den).fillna(0.0) * b[Lc]
    if VENUE and Lc == 'L':     # shrunk venue x innings effect on batting-side value (a flat pitch flatters every batter who plays there)
        vk = [b.venue_id, b.innings]
        ve = out.where(rowmask, 0.0).groupby(vk).transform('sum') / (b[Lc].where(rowmask, 0.0).groupby(vk).transform('sum') + K_V)
        out = out - ve * b[Lc]
    return out


def opp_adjust(colbat, colbowl, cell_bat, cell_bowl, Lc='L'):
    """Iteratively remove the (shrunk) quality of the opponent faced. Returns batter-adjusted and bowler-adjusted columns."""
    vb = centre(colbat, b.bat_row, cell_bat, Lc); cb = centre(colbowl, ALL, cell_bowl, Lc); LL = b[Lc]
    bat_q = pd.Series(0.0, index=b.striker_id.unique()); bowl_q = pd.Series(0.0, index=b.bowler_id.unique())
    br = b.bat_row
    for it in range(8):
        adj = vb[br] - LL[br] * bowl_q.reindex(b.bowler_id[br]).values
        bat_q = adj.groupby(b.striker_id[br]).sum() / (LL[br].groupby(b.striker_id[br]).sum() + K_OPP)
        adj = cb - LL * bat_q.reindex(b.striker_id).fillna(0).values
        bowl_q = adj.groupby(b.bowler_id).sum() / (LL.groupby(b.bowler_id).sum() + K_OPP)
    aw = bowl_q.reindex(b.bowler_id).values; ab = bat_q.reindex(b.striker_id).fillna(0).values
    if MATCHUP and Lc == 'L':
        # second level: how much better or worse than their overall quality each opponent is against this type (shrunk harder)
        wk = b.bowler_id * 2 + _lhb; bk = b.striker_id * 2 + _spin
        for it in range(4):
            r_ = (vb - LL * aw)[br]; kb = bk[br]
            bat_d = (r_ - LL[br] * bat_q.reindex(b.striker_id[br]).fillna(0).values).groupby(kb).sum() / (LL[br].groupby(kb).sum() + K_MU)
            ab = bat_q.reindex(b.striker_id).fillna(0).values + bat_d.reindex(bk).fillna(0).values
            r_ = cb - LL * ab
            bowl_d = (r_ - LL * bowl_q.reindex(b.bowler_id).values).groupby(wk).sum() / (LL.groupby(wk).sum() + K_MU)
            aw = bowl_q.reindex(b.bowler_id).values + bowl_d.reindex(wk).fillna(0).values
    return vb - LL * aw, -(cb - LL * ab), vb, -cb


# fixed yardstick used by every out-of-sample test: actual win probability added, stakes definition, plain centring
b['Rw0'] = b.dwp_nowk * f0; b['Kw0_bat'] = np.where(striker_out, b.dwp - b.dwp_nowk, 0.0) * f0; b['Kw0_bowl'] = np.where(b.is_wicket & ~nonbowler, b.dwp - b.dwp_nowk, 0.0) * f0
_r0 = opp_adjust('Rw0', 'Rw0', [b.press0], [b.press0], 'L0'); _k0 = opp_adjust('Kw0_bat', 'Kw0_bowl', [b.press0], [b.press0], 'L0')
b['bat_T0'] = _r0[0] + _k0[0]; b['bowl_T0'] = _r0[1] + _k0[1]
# ---------------- same-game all-round impact: the player added win probability with BOTH bat and ball in the match (6+ balls of each)
AR_W = float(os.environ.get('AR_W', '0'))          # all-round bonus: tested and NOT used in the final rating (0 = off); see report 7.3
AR_K = 10.0                                        # prior matches with no double impact (protects short careers)
AR_MINB = 6
_bt = b[b.bat_row].groupby(['striker_id', 'match_id']).agg(bv=('bat_T0', 'sum'), bn=('legal', 'sum')).rename_axis(['pid', 'match_id'])
_bw = b.groupby(['bowler_id', 'match_id']).agg(wv=('bowl_T0', 'sum'), wn=('legal', 'sum')).rename_axis(['pid', 'match_id'])
_ar = _bt.join(_bw, how='outer').fillna(0)
DBL = ((_ar.bn >= AR_MINB) & (_ar.wn >= AR_MINB) & (_ar.bv > 0) & (_ar.wv > 0)).astype(float)
# actual win probability added under the active centring (used for the consistency pillar)
_tR = opp_adjust('Rw', 'Rw', cells('bat'), cells('bowl')); _tK = opp_adjust('Kw_bat', 'Kw_bowl', cells('bat'), cells('bowl'))
b['bat_T'] = _tR[0] + _tK[0]; b['bowl_T'] = _tR[1] + _tK[1]; b['bat_T_raw'] = _tR[2] + _tK[2]; b['bowl_T_raw'] = _tR[3] + _tK[3]
PRICES = {}; EXAM_TABLE = None
if VALUE in ('exam', 'blend'):
    # THE PRESSURE EXAM. Every ball is re-priced at the average worth of its outcome (dot, 1, 2, 3, 4, 6, wicket, extra) across ALL the
    # situations of the same kind (pressure flag x innings x phase x wickets-down band), using what the win-probability model says each
    # outcome would have been worth on every ball in the data. A six is worth the same to everyone who hits one in that kind of situation:
    # the luck of exactly which ball it came on is removed, and everyone is marked against the same paper.
    strata = [b.press, b.innings, b.phase, WKB]
    den = b.L.groupby(strata).transform('sum')
    V = np.stack([((b[f'h{j}'] * f).groupby(strata).transform('sum') / den).values for j in range(8)])
    rc = lambda r: np.select([r <= 3, r <= 5], [r, 4], default=5).astype(int)
    ar = np.arange(len(b))
    b['Re_bat'] = V[rc(b.runs.values), ar] * b.L.values
    b['Re_bowl'] = V[np.where(b.legal.values == 0, 7, rc(b.bowl_runs.values)), ar] * b.L.values
    b['Ke_bat'] = b.bat_out.values * (V[6] - V[0]) * b.L.values
    b['Ke_bowl'] = b.bowl_wkt.values * (V[6] - V[0]) * b.L.values
    _eR = opp_adjust('Re_bat', 'Re_bowl', cells('bat', True), cells('bowl', True)); _eK = opp_adjust('Ke_bat', 'Ke_bowl', cells('bat', True), cells('bowl', True))
    EXAM_TABLE = pd.DataFrame(V.T, columns=['dot', '1', '2', '3', '4', '6', 'wicket', 'extra']).assign(press=b.press.values, innings=b.innings.values, phase=b.phase.values, wkb=WKB.values, n=1).groupby(['press', 'innings', 'phase', 'wkb']).agg({**{k: 'first' for k in ['dot', '1', '2', '3', '4', '6', 'wicket', 'extra']}, 'n': 'sum'})
    mix = 1.0 if VALUE == 'exam' else 0.5
    for i, nm in enumerate(('bat_{}', 'bowl_{}', 'bat_{}_raw', 'bowl_{}_raw')):
        b[nm.format('R')] = mix * _eR[i] + (1 - mix) * _tR[i]; b[nm.format('K')] = mix * _eK[i] + (1 - mix) * _tK[i]
else:
    b['bat_R'], b['bowl_R'], b['bat_R_raw'], b['bowl_R_raw'] = _tR
    b['bat_K'], b['bowl_K'], b['bat_K_raw'], b['bowl_K_raw'] = _tK

# ball-level consistency: a pressure ball is 'not lost' if its value to the player is zero or better; compared with the same kind of ball
GBVAR = {}
for _role, _mask in (('bat', b.bat_row), ('bowl', ALL)):
    _g = ((b[f'{_role}_R'] + b[f'{_role}_K']) >= 0).astype(float)
    _cell = cells(_role, True)
    _gm = _g.where(_mask, np.nan).groupby(_cell).transform('mean')
    b[f'{_role}_gb'] = (_g - _gm).fillna(0.0)
    GBVAR[_role] = float(b.loc[_mask & b[f'press_{_role}'], f'{_role}_gb'].var())
names = pd.concat([b[['striker_id', 'striker_name']].set_axis(['id', 'name'], axis=1), b[['bowler_id', 'bowler_name']].set_axis(['id', 'name'], axis=1)]).drop_duplicates('id').set_index('id').name
mgroup = b.drop_duplicates('match_id').set_index('match_id').group
mdate = b.drop_duplicates('match_id').set_index('match_id').match_date
AGG = ['gb', 'R', 'K', 'T', 'L', 'n', 'cr', 'cb', 'Rx', 'Kx', 'Lx', 'Lxx', 'T0p', 'T0n', 'Lp', 'Ln', 'nS', 'Tq', 'Lq', 'nq']
SUMCOLS = [f"{c}_{s}" for c in AGG for s in 'PN']


def match_agg(role, raw=False):
    d = b[b.bat_row] if role == 'bat' else b
    pid = 'striker_id' if role == 'bat' else 'bowler_id'
    sfx = '_raw' if raw else ''
    R_, K_ = d[f'{role}_R{sfx}'], d[f'{role}_K{sfx}']
    sp = d.press0.astype(float); sq = d[f'SQ_{role}'].astype(float)
    q = pd.DataFrame({'pid': d[pid].values, 'match_id': d.match_id.values, 'press': d[f'press_{role}'].values, 'R': R_.values, 'K': K_.values, 'T': d[f'{role}_T{sfx}'].values,
                      'L': d.L.values, 'n': 1.0, 'cr': (d.runs if role == 'bat' else -d.bowl_runs).astype(float).values, 'cb': d.legal.astype(float).values,
                      'Rx': (R_ * d.x).values, 'Kx': (K_ * d.x).values, 'Lx': (d.L * d.x).values, 'Lxx': (d.L * d.x ** 2).values,
                      'T0p': (d[f'{role}_T0'] * sp).values, 'T0n': (d[f'{role}_T0'] * (1 - sp)).values, 'Lp': (d.L0 * sp).values, 'Ln': (d.L0 * (1 - sp)).values, 'nS': sp.values, 'Tq': (d[f'{role}_T0'] * sq).values, 'Lq': (d.L0 * sq).values, 'nq': sq.values, 'gb': d[f'{role}_gb'].values})
    g = q.groupby(['pid', 'match_id', 'press'])[AGG].sum().unstack('press', fill_value=0)
    g.columns = [f"{a}_{'P' if p else 'N'}" for a, p in g.columns]
    for c in SUMCOLS:
        if c not in g.columns:
            g[c] = 0.0
    g = g.reset_index()
    g['role'] = role
    g['dbl'] = DBL.reindex(pd.MultiIndex.from_arrays([g.pid, g.match_id])).fillna(0).values
    return g


def rates(a, w=None):
    """Per-player sums and rates from player-match aggregates (optionally with match weights)."""
    ww = np.ones(len(a)) if w is None else a.match_id.map(w).values
    x = a[SUMCOLS].multiply(ww, axis=0)
    app = (a.n_P >= MIN_APP).values
    x['app'] = app * ww
    x['pos'] = (app & (a.T_P > 0).values) * ww          # appearance counts as positive if actual win probability was added
    x['pm'] = (a.n_P > 0).values * ww
    x['pmS'] = ((a.nS_P + a.nS_N) > 0).values * ww
    x['dbl'] = a.dbl.values * ww; x['mt'] = ww
    s = x.groupby(a.pid.values).sum()
    SL = s.L_P + s.L_N; xb = (s.Lx_P + s.Lx_N) / SL; D = (s.Lxx_P + s.Lxx_N) - xb ** 2 * SL
    s['xb'] = xb; s['D'] = D
    for c in COMP:
        s[f'{c}_rP'] = s[f'{c}_P'] / s.L_P; s[f'{c}_rN'] = s[f'{c}_N'] / s.L_N; s[f'{c}_upB'] = s[f'{c}_rP'] - s[f'{c}_rN']
        s[f'{c}_rA'] = (s[f'{c}_P'] + s[f'{c}_N']) / SL
        s[f'{c}_sl'] = ((s[f'{c}x_P'] + s[f'{c}x_N']) - xb * (s[f'{c}_P'] + s[f'{c}_N'])) / D          # change in value per unit leverage, per unit of log-leverage
        s[f'{c}_upS'] = s[f'{c}_sl'] * GAPX                                                              # ...expressed as the gap between a typical pressure and a typical ordinary ball
        s[f'{c}_up'] = s[f'{c}_upS'] if UPMODE == 'slope' else s[f'{c}_upB']
    s['rP'] = s.R_rP + s.K_rP; s['rN'] = s.R_rN + s.K_rN; s['up'] = s.R_up + s.K_up; s['rA'] = s.R_rA + s.K_rA
    s['upB'] = s.R_upB + s.K_upB; s['upS'] = s.R_upS + s.K_upS
    s['wP'] = s.L_P / SL; s['nA'] = SL
    s['nU'] = D / GAPX ** 2 if UPMODE == 'slope' else 1 / (1 / s.L_P + 1 / s.L_N)
    s['T_rP'] = s.T_P / s.L_P; s['T_rN'] = s.T_N / s.L_N          # actual win probability added per unit leverage
    s['T0_rP'] = (s.T0p_P + s.T0p_N) / (s.Lp_P + s.Lp_N); s['T0_rN'] = (s.T0n_P + s.T0n_N) / (s.Ln_P + s.Ln_N)   # fixed yardstick (stakes pressure)
    s['nS'] = s.nS_P + s.nS_N
    s['T_rQ'] = (s.Tq_P + s.Tq_N) / (s.Lq_P + s.Lq_N); s['nQ'] = s.nq_P + s.nq_N     # actual value when squeezed (batters) / when being hit (bowlers)
    for c in COMP:
        s[f'{c}_xP'] = 2 * s[f'{c}_rP'] - 0 if VALUE != 'blend' else s[f'{c}_rP']
    s['exam_rP'] = (2 * s.rP - s.T_rP) if VALUE == 'blend' else s.rP; s['exam_rN'] = (2 * s.rN - s.T_rN) if VALUE == 'blend' else s.rN
    s['cons'] = s.pos / s.app
    s['cball'] = s.gb_P / s.n_P                     # share of pressure balls not lost, relative to the same kind of ball
    s['gbvar'] = GBVAR[a.role.iloc[0]] if len(a) else np.nan
    s['ar'] = s.dbl / (s.mt + AR_K)                 # share of the player's matches with double impact, shrunk towards zero
    s['conv_P'] = s.cr_P / s.cb_P; s['conv_A'] = (s.cr_P + s.cr_N) / (s.cb_P + s.cb_N)     # runs per ball (negated for bowlers): the conventional yardstick
    return s


def cluster_se(a, s):
    a2 = a.join(s[[f'{c}_{r}' for c in COMP for r in ('rP', 'rN', 'rA', 'sl')] + ['L_P', 'L_N', 'xb', 'D']].rename(columns={'L_P': 'TLP', 'L_N': 'TLN'}), on='pid')
    out = {}; dU = 0
    for c in COMP:
        eP = (a2[f'{c}_P'] - a2[f'{c}_rP'] * a2.L_P) / a2.TLP
        eN = (a2[f'{c}_N'] - a2[f'{c}_rN'] * a2.L_N) / a2.TLN
        out[f'{c}_seN'] = np.sqrt((eN ** 2).groupby(a2.pid).sum())
        eA = (a2[f'{c}_P'] + a2[f'{c}_N'] - a2[f'{c}_rA'] * (a2.L_P + a2.L_N)) / (a2.TLP + a2.TLN)
        out[f'{c}_seA'] = np.sqrt((eA ** 2).groupby(a2.pid).sum())
        if UPMODE == 'slope':     # match-clustered error of the response-curve slope, in uplift units
            Lm = a2.L_P + a2.L_N; Lxm = a2.Lx_P + a2.Lx_N; Lxxm = a2.Lxx_P + a2.Lxx_N; cm = a2[f'{c}_P'] + a2[f'{c}_N']; cxm = a2[f'{c}x_P'] + a2[f'{c}x_N']
            eS = ((cxm - a2.xb * cm) - a2[f'{c}_rA'] * (Lxm - a2.xb * Lm) - a2[f'{c}_sl'] * (Lxxm - 2 * a2.xb * Lxm + a2.xb ** 2 * Lm)) / a2.D * GAPX
            out[f'{c}_seU'] = np.sqrt((eS ** 2).groupby(a2.pid).sum())
        else:
            out[f'{c}_seU'] = np.sqrt(((eP - eN) ** 2).groupby(a2.pid).sum())
        out[f'{c}_seP'] = np.sqrt((eP ** 2).groupby(a2.pid).sum())
        dU = dU + (eP - eN)
    out['up_seU'] = np.sqrt((dU ** 2).groupby(a2.pid).sum())      # error of the TOTAL uplift (run and wicket errors are not independent)
    return pd.DataFrame(out)


def eb_params(s, el):
    """Empirical-Bayes prior mean, between-player variance and pooled noise constant per component, from the eligible pool.
    Sampling variance of a player's rate is modelled as k / n (k pooled across players) so reliability depends on sample size, not on a noisy per-player SE."""
    e = s.loc[el]; prm = {}
    for c in COMP:
        for q, se, n in (('rA', 'seA', 'nA'), ('up', 'seU', 'nU')):
            col, sec = f'{c}_{q}', f'{c}_{se}'
            ok = np.isfinite(e[col]) & np.isfinite(e[sec]) & (e[sec] > 0)
            k = float(np.median(e.loc[ok, sec] ** 2 * e.loc[ok, n]))
            se2 = k / e.loc[ok, n]
            mu = np.average(e.loc[ok, col], weights=1 / se2)
            tau2 = max(e.loc[ok, col].var() - se2.mean(), 0.02 * e.loc[ok, col].var())
            prm[col] = (mu, tau2, k)
    ok = np.isfinite(e.upB) & np.isfinite(e.up_seU) & (e.up_seU > 0); nUb = 1 / (1 / e.L_P + 1 / e.L_N)
    k = float(np.median(e.loc[ok, 'up_seU'] ** 2 * nUb[ok])); se2 = k / nUb[ok]
    prm['up_tot'] = (float(np.average(e.loc[ok, 'upB'], weights=1 / se2)), float(max(e.loc[ok, 'upB'].var() - se2.mean(), 0.02 * e.loc[ok, 'upB'].var())), k)
    pbar = e.pos.sum() / e.app.sum(); pi = e.pos / e.app
    t2 = max(pi.var() - (pbar * (1 - pbar) / e.app).mean(), 1e-6)
    prm['cons'] = pbar
    prm['KC'] = float(np.clip(pbar * (1 - pbar) / t2 - 1, 5, 200))
    t2b = max(e.cball.var() - (e.gbvar / e.n_P).mean(), 1e-8)
    prm['KCB'] = float(np.clip(e.gbvar.iloc[0] / t2b, 20, 5000))
    prm['sd'] = dict(level_eb=float(e.rP.std()), uplift_eb=float(e.up.std()), cons_eb=float(e.cball.std() if CONSMODE == 'ball' else pi.std()))
    return prm


def add_rel(s, prm):
    for c in COMP:
        for q, n in (('rA', 'nA'), ('up', 'nU')):
            mu, tau2, k = prm[f'{c}_{q}']
            s[f'{c}_{q}_rel'] = tau2 / (tau2 + k / s[n].clip(lower=1e-9))
    mu, tau2, k = prm['up_tot']
    s['up_tot_rel'] = tau2 / (tau2 + k * (1 / s.L_P + 1 / s.L_N))
    return s


def pillars(s, relsrc, prm):
    """Three pillars in natural units. relsrc supplies the reliability weights.
    base  = shrunk rate over ALL the player's balls (their usual level)
    uplift = shrunk (pressure rate - normal rate)
    level = estimated pressure rate = base + (1 - pressure share) * uplift   [the all-ball rate already contains pressure share x uplift]"""
    o = pd.DataFrame(index=s.index)
    base = 0; upl = 0
    for c in COMP:
        mu_a, mu_u = prm[f'{c}_rA'][0], prm[f'{c}_up'][0]
        bn = mu_a + relsrc[f'{c}_rA_rel'].reindex(s.index) * (s[f'{c}_rA'] - mu_a)
        bu = mu_u + relsrc[f'{c}_up_rel'].reindex(s.index) * (s[f'{c}_up'] - mu_u)
        o[f'{c}_base_eb'] = bn; o[f'{c}_up_eb'] = bu
        base = base + bn; upl = upl + bu
    if UPTOT == 'total' and UPMODE == 'binary':
        upl = prm['up_tot'][0] + relsrc['up_tot_rel'].reindex(s.index) * (s.upB - prm['up_tot'][0])
    o['base_eb'] = base; o['uplift_eb'] = upl; o['level_eb'] = base + (1 - s.wP) * upl
    o['cons_eb'] = (s.gb_P / (s.n_P + prm['KCB'])) if CONSMODE == 'ball' else (s.pos + prm['KC'] * prm['cons']) / (s.app + prm['KC'])
    o['ar_p'] = s.ar
    return o


def composite(p, ref, prm, w=None):
    """Each pillar is centred on the pool and divided by the spread of its RAW (unshrunk) version, so a pillar that is mostly noise
    has been shrunk towards zero and cannot decide the ranking; nominal weights apply to true-talent units."""
    sd = prm['sd'] if ZMODE == 'raw' else {c: ref[c].std() for c in ('level_eb', 'uplift_eb', 'cons_eb')}
    z = lambda c: (p[c] - ref[c].mean()) / sd[c]
    w = w or W
    out = pd.DataFrame({'z_level': z('level_eb'), 'z_uplift': z('uplift_eb'), 'z_cons': z('cons_eb')})
    out['core'] = w['level'] * out.z_level + w['uplift'] * out.z_uplift + w['cons'] * out.z_cons
    out['bonus'] = AR_W * p['ar_p'].fillna(0)                      # all-round bonus, in rating points
    out['score'] = out.core + out.bonus
    return out


def full_pipeline(a, el, prm=None):
    """rates -> SE -> EB -> pillars -> composite on a set of player-match rows (used for full sample and split halves)."""
    s = rates(a)
    s = s.join(cluster_se(a, s))
    el = [p for p in el if p in s.index and np.isfinite(s.loc[p, ['rP', 'rN']]).all() and s.loc[p, 'app'] > 0]
    prm = prm or eb_params(s, el)
    s = add_rel(s, prm)
    p = pillars(s.loc[el], s, prm)
    return s, p, composite(p, p, prm), prm


def conventional(role):
    if role == 'bat':
        d = b[b.bat_row]
        g = d.groupby(['striker_id', 'press']).agg(runs=('runs', 'sum'), balls=('legal', 'sum'), outs=('bat_out', 'sum'), fours=('is_four', 'sum'), sixes=('is_six', 'sum'),
                                                   dots=('runs', lambda x: (x == 0).sum()))
        g['SR'] = 100 * g.runs / g.balls; g['Avg'] = g.runs / g.outs.replace(0, np.nan); g['Bdry%'] = 100 * (g.fours + g.sixes) / g.balls; g['Dot%'] = 100 * g.dots / g.balls
        g['BPD'] = g.balls / g.outs.replace(0, np.nan)
        cols = ['runs', 'balls', 'outs', 'SR', 'Avg', 'Bdry%', 'Dot%', 'BPD']
    else:
        d = b.assign(bd=(b.is_four | b.is_six), dot=(b.tot == 0))
        g = d.groupby(['bowler_id', 'press']).agg(runs=('bowl_runs', 'sum'), balls=('legal', 'sum'), wkts=('bowl_wkt', 'sum'), dots=('dot', 'sum'), bdry=('bd', 'sum'))
        g['Econ'] = 6 * g.runs / g.balls; g['SR'] = g.balls / g.wkts.replace(0, np.nan); g['Avg'] = g.runs / g.wkts.replace(0, np.nan)
        g['Dot%'] = 100 * g.dots / g.balls; g['Bdry%'] = 100 * g.bdry / g.balls
        cols = ['runs', 'balls', 'wkts', 'Econ', 'SR', 'Avg', 'Dot%', 'Bdry%']
    g = g[cols].unstack('press')
    g.columns = [f"{a}_{'P' if p else 'N'}" for a, p in g.columns]
    return g


if __name__ == '__main__':
    print(f"pressure threshold LI >= {thr:.3f}; pressure balls {b.press.sum()} ({b.press.mean():.1%}); share of total leverage {b.li[b.press].sum() / b.li.sum():.1%}; capped balls {(b.li > CAP).mean():.2%}")
    print("composition of pressure balls:\n", (b[b.press].groupby(['innings', 'phase']).size() / b.press.sum()).round(3).unstack())
    print("pressure share by group:", b.groupby('group').press.mean().round(3).to_dict())
    res = {}
    rng = np.random.RandomState(11)
    mids = b.match_id.unique()
    order = mdate.sort_values().index.values
    for role in ('bat', 'bowl'):
        W = W_ROLE[role]
        a = match_agg(role)
        s0 = rates(a)
        el = list(s0.index[(s0.nS >= MIN_PBALLS) & (s0.pmS >= MIN_PMATCH)])     # eligibility always on the stakes definition
        s, p, comp, prm = full_pipeline(a, el)
        el = list(p.index)
        t = s.loc[el].join(p).join(comp).join(conventional(role))
        t.insert(0, 'name', names.reindex(t.index))
        pidc = 'striker_id' if role == 'bat' else 'bowler_id'
        t = t.join(b.groupby(pidc).agg(last_year=('year', 'max'), first_year=('year', 'min')))
        gp = a.assign(g=a.match_id.map(mgroup)).groupby(['pid', 'g']).n_P.sum().unstack(fill_value=0)
        t['ipl_share_P'] = ((gp['IPL'] if 'IPL' in gp else 0 * gp.sum(axis=1)) / gp.sum(axis=1)).reindex(t.index)
        t = t.sort_values('score', ascending=False)
        t['rank'] = np.arange(1, len(t) + 1)

        # --- bootstrap over matches (Poisson weights); EB reliabilities and z-reference held fixed
        ae = a[a.pid.isin(el)]
        ranks = np.zeros((NBOOT, len(el))); scores = np.zeros((NBOOT, len(el)))
        fill = p[['level_eb', 'uplift_eb', 'cons_eb']].mean()
        for i in range(NBOOT):
            w = pd.Series(rng.poisson(1.0, len(mids)).astype(float), index=mids)
            sb = rates(ae, w).reindex(el)
            pb = pillars(sb, s, prm).fillna(fill)
            sc = composite(pb, p, prm).score
            scores[i] = sc.values; ranks[i] = (-sc).rank().values
        t = t.join(pd.DataFrame({'p_top10': (ranks <= 10).mean(0), 'p_top20': (ranks <= 20).mean(0), 'rank_lo': np.percentile(ranks, 10, axis=0),
                                 'rank_hi': np.percentile(ranks, 90, axis=0), 'score_sd': scores.std(0)}, index=el))

        # --- split-half reliability of every ingredient and the final score (whole pipeline re-run in each half)
        rel = []
        SPL = []
        for i in range(NSPLIT):
            hh = pd.Series(rng.randint(0, 2, len(order)), index=order)
            q = [full_pipeline(ae[ae.match_id.map(hh).values == h], el) for h in (0, 1)]
            SPL.append({h: pd.DataFrame({'score': q[h][2].score, 'T0': q[h][0].T0_rP.reindex(q[h][2].index)}) for h in (0, 1)})
            common = q[0][1].index.intersection(q[1][1].index)
            row = {}
            for c in ('level_eb', 'uplift_eb', 'cons_eb', 'base_eb'):
                row[c] = np.corrcoef(q[0][1].loc[common, c], q[1][1].loc[common, c])[0, 1]
            for c in ('rP', 'rN', 'up', 'R_rP', 'K_rP', 'R_rN', 'K_rN', 'R_up', 'K_up'):
                row['raw_' + c] = np.corrcoef(q[0][0].loc[common, c], q[1][0].loc[common, c])[0, 1]
            row['score'] = np.corrcoef(q[0][2].loc[common, 'score'], q[1][2].loc[common, 'score'])[0, 1]
            row['score_spearman'] = q[0][2].loc[common, 'score'].corr(q[1][2].loc[common, 'score'], method='spearman')
            # does first-half baseline+uplift predict second-half *raw* pressure performance?
            row['pred_level_to_rawP'] = 0.5 * (np.corrcoef(q[0][1].loc[common, 'level_eb'], q[1][0].loc[common, 'T0_rP'])[0, 1] + np.corrcoef(q[1][1].loc[common, 'level_eb'], q[0][0].loc[common, 'T0_rP'])[0, 1])
            row['pred_rawP_to_rawP'] = np.corrcoef(q[0][0].loc[common, 'T0_rP'], q[1][0].loc[common, 'T0_rP'])[0, 1]
            for nm_, src_, col_ in (('score', 2, 'score'), ('core', 2, 'core'), ('convP', 0, 'conv_P'), ('convA', 0, 'conv_A'), ('base', 1, 'base_eb')):
                row[f'pred_{nm_}_to_rawP'] = 0.5 * (np.corrcoef(q[0][src_].loc[common, col_], q[1][0].loc[common, 'T0_rP'])[0, 1] + np.corrcoef(q[1][src_].loc[common, col_], q[0][0].loc[common, 'T0_rP'])[0, 1])
            rel.append(row)
        rel = pd.DataFrame(rel).mean().round(3)
        # chronological: first half of matches -> second half
        cut = order[len(order) // 2]
        early = mdate.reindex(ae.match_id).values < mdate[cut]
        q = [full_pipeline(ae[early], el), full_pipeline(ae[~early], el)]
        CHR = {h: pd.DataFrame({'score': q[h][2].score, 'nS': q[h][0].nS.reindex(q[h][2].index), 'T0': q[h][0].T0_rP.reindex(q[h][2].index)}) for h in (0, 1)}
        common = q[0][1].index.intersection(q[1][1].index)
        common = [c for c in common if q[0][0].loc[c, 'n_P'] >= 50 and q[1][0].loc[c, 'n_P'] >= 50]
        rel['chrono_n'] = len(common)
        rel['chrono_score'] = q[0][2].loc[common, 'score'].corr(q[1][2].loc[common, 'score'])
        rel['chrono_level'] = q[0][1].loc[common, 'level_eb'].corr(q[1][1].loc[common, 'level_eb'])
        rel['chrono_level_to_rawP'] = q[0][1].loc[common, 'level_eb'].corr(q[1][0].loc[common, 'T0_rP'])
        # --- season to season: the whole pipeline run on each season alone, consecutive seasons compared (players with 40+ pressure balls in both)
        myear = b.drop_duplicates('match_id').set_index('match_id').year
        ay = ae.assign(y=ae.match_id.map(myear).values); ys = sorted(ay.y.unique()); qy = {}
        for y_ in ys:
            try:
                qy[y_] = full_pipeline(ay[ay.y == y_].drop(columns='y'), el)
            except Exception:
                pass
        acc = {k: [] for k in ('score', 'level_eb', 'upB', 'T0_rP')}; wts = []
        for y0, y1 in zip(ys[:-1], ys[1:]):
            if y0 in qy and y1 in qy:
                cm = qy[y0][1].index.intersection(qy[y1][1].index)
                cm = [c_ for c_ in cm if qy[y0][0].loc[c_, 'nS'] >= 40 and qy[y1][0].loc[c_, 'nS'] >= 40]
                if len(cm) >= 15:
                    acc['score'].append(qy[y0][2].loc[cm, 'score'].corr(qy[y1][2].loc[cm, 'score'])); acc['level_eb'].append(qy[y0][1].loc[cm, 'level_eb'].corr(qy[y1][1].loc[cm, 'level_eb']))
                    acc['upB'].append(qy[y0][0].loc[cm, 'upB'].corr(qy[y1][0].loc[cm, 'upB'])); acc['T0_rP'].append(qy[y0][0].loc[cm, 'T0_rP'].corr(qy[y1][0].loc[cm, 'T0_rP'])); wts.append(len(cm))
        if wts:
            for k_, v_ in acc.items():
                rel['yy_' + k_] = float(np.average(v_, weights=wts))
            rel['yy_pairs'] = len(wts); rel['yy_n'] = float(np.mean(wts))

        # --- sensitivity to pillar weights
        sens = {}
        for nm, ww in {'50/25/25 (base)': (0.5, 0.25, 0.25), '100/0/0': (1, 0, 0), '40/40/20': (0.4, 0.4, 0.2), '60/20/20': (0.6, 0.2, 0.2), '34/33/33': (0.34, 0.33, 0.33), '50/0/50': (0.5, 0, 0.5), '50/50/0': (0.5, 0.5, 0), '70/15/15': (.7, .15, .15)}.items():
            sens[nm] = (-(ww[0] * t.z_level + ww[1] * t.z_uplift + ww[2] * t.z_cons + t.bonus)).rank()
        sens = pd.DataFrame(sens)
        t['sens_top10_share'] = (sens <= 10).mean(axis=1)
        t['rank_nobonus'] = (-t.core).rank()
        # raw (no opponent adjustment) rank for comparison
        ar = match_agg(role, raw=True)
        _, pr, cr, _ = full_pipeline(ar, el)
        t['rank_noadj'] = (-cr.score).rank().reindex(t.index)

        YRS = {y_: pd.DataFrame({'score': v_[2].score, 'nS': v_[0].nS.reindex(v_[2].index)}) for y_, v_ in qy.items()}
        res[role] = dict(table=t, all=s, prm=prm, rel=rel, sens=sens, agg=a, boot=pd.DataFrame(scores, columns=el), splits=SPL, chrono=CHR, years=YRS)
        print(f"\n================ {role.upper()} : eligible {len(el)}")
        print("EB params:", {k: (round(float(v[0]) * 1e4, 2), round(float(v[1]) * 1e8, 2)) if isinstance(v, tuple) else v for k, v in prm.items()})
        print("mean EB reliabilities:", t[[c for c in t.columns if c.endswith('_rel')]].mean().round(2).to_dict())
        print("split-half:\n", rel.to_string())
        show = ['name', 'rank', 'score', 'z_level', 'z_uplift', 'z_cons', 'rP', 'rN', 'level_eb', 'base_eb', 'uplift_eb', 'cons', 'n_P', 'pm', 'app', 'p_top10', 'rank_lo', 'rank_hi', 'sens_top10_share', 'rank_noadj', 'ipl_share_P', 'last_year']
        conv = ['SR_P', 'SR_N', 'Avg_P', 'Avg_N', 'Dot%_P'] if role == 'bat' else ['Econ_P', 'Econ_N', 'SR_P', 'SR_N', 'Dot%_P', 'wkts_P']
        tt = t[show + conv].copy()
        for c in ['rP', 'rN', 'level_eb', 'base_eb', 'uplift_eb']:
            tt[c] = tt[c] * 1e4
        print(tt.head(40).round(2).to_string())
        print("... bottom 6")
        print(tt.tail(6).round(2).to_string())
        t.to_csv(f"{OUT}/{role}_ratings_eligible{SFX}.csv")

    pickle.dump(dict(res=res, thr=thr, PCT=PCT, CAP=CAP, W=W, W_ROLE=W_ROLE, WEIGHTS=WEIGHTS, MIN_PBALLS=MIN_PBALLS, MIN_PMATCH=MIN_PMATCH, K_OPP=K_OPP, MIN_APP=MIN_APP, VALUE=VALUE, CENTRE=CENTRE, ZMODE=ZMODE, PRESSMODE=PRESSMODE, UPMODE=UPMODE, KOW=KOW, UPTOT=UPTOT, MATCHUP=MATCHUP, VENUE=VENUE, CONSMODE=CONSMODE, AR_W=AR_W, AR_K=AR_K, AR_MINB=AR_MINB, EXAM_TABLE=EXAM_TABLE, GAPX=GAPX, LI_SQ=LI_SQ, PRICES=PRICES, NBOOT=NBOOT), open(f"{OUT}/results{SFX}.pkl", 'wb'))  # W saved is the last role's; W_ROLE holds both
    if not SFX:
        b.to_pickle(f"{OUT}/balls_scored.pkl")







"""Do 'how they play' traits (shot selection, delivery mix, speed, role) sharpen the estimate of pressure value beyond the Level pillar?
Tested on unseen matches AND unseen players: traits + Level from half A predict actual pressure value in half B, with the regression
fitted on other players only (5-fold over players)."""
import os as _os
ROOT = _os.environ.get('RR_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # repository root
import sys, importlib.util, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
spec = importlib.util.spec_from_file_location('r5', _os.path.join(ROOT, 'analysis', '05_ratings.py')); r5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r5)
b = r5.b
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
rng = np.random.RandomState(31)
mids = b.match_id.unique()
DEF = ['Frontfoot defensive', 'Backfoot defensive', 'Push', 'Worked', 'Leg Glance', 'Steer', 'Defense', 'Intentional Padding', 'Glance']
st = b.stroke
F = {'bat': pd.DataFrame({'n': 1.0, 'att': (~st.isin(DEF) & st.notna()).astype(float),
                          'innov': st.isin(['Paddle Scoop', 'Reverse Sweep', 'Switch', 'Reverse Paddle Scoop', 'UpperCut', 'Inside out', 'Late Cut']).astype(float),
                          'sweep': st.isin(['Sweep Shot', 'Slog Sweep', 'Paddle Sweep']).astype(float), 'slog': st.isin(['Slog', 'Slog Sweep', 'PickUp']).astype(float),
                          'vspin': (b.pace_or_spin == 'Spin').astype(float), 'order': b.batting_order.astype(float), 'dot': (b.runs == 0).astype(float),
                          'bdry': (b.is_four | b.is_six).astype(float), 'six': b.is_six.astype(float), 'lhb': (b.batting_type == 'LHB').astype(float), 'death': (b.phase == 'DEATH').astype(float), 'pp': (b.phase == 'PP').astype(float)}),
     'bowl': pd.DataFrame({'n': 1.0, 'spin': (b.pace_or_spin == 'Spin').astype(float), 'speed': np.where(b.bowl_speed.between(60, 165), b.bowl_speed, np.nan), 'spd_n': b.bowl_speed.between(60, 165).astype(float),
                           'yorker': b.delivery_style.isin(['Yorker', 'Wide Yorker']).astype(float), 'slower': b.delivery_style.isin(['Off Cutter', 'Leg Cut', 'Knuckle', 'Back of Hand', 'Slow Bouncer']).astype(float),
                           'ftoss': (b.delivery_style == 'Full Toss').astype(float), 'bouncer': b.delivery_style.isin(['Bouncer']).astype(float), 'wide': (b.extras_type == 'wide').astype(float),
                           'dot': (b.tot == 0).astype(float), 'bdry': (b.is_four | b.is_six).astype(float), 'death': (b.phase == 'DEATH').astype(float), 'pp': (b.phase == 'PP').astype(float),
                           'left': b.bowling_type.str.startswith('Left').astype(float), 'drawn': (~st.isin(DEF) & st.notna()).astype(float)})}
F['bowl']['speed'] = F['bowl']['speed'].fillna(0.0)


def ridge_cv(X, y, alpha, folds):
    pred = np.zeros(len(y))
    for k in np.unique(folds):
        tr = folds != k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = (X[tr] - mu) / sd; ym = y[tr].mean()
        co = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y[tr] - ym))
        pred[~tr] = ((X[~tr] - mu) / sd) @ co + ym
    return pred


for role in ('bat', 'bowl'):
    pidc = 'striker_id' if role == 'bat' else 'bowler_id'
    mask = b.bat_row if role == 'bat' else pd.Series(True, index=b.index)
    fm = F[role][mask].assign(pid=b[pidc][mask].values, match_id=b.match_id[mask].values).groupby(['pid', 'match_id']).sum().reset_index()
    a = r5.match_agg(role); s0 = r5.rates(a)
    el = list(s0.index[(s0.nS >= r5.MIN_PBALLS) & (s0.pmS >= r5.MIN_PMATCH)])
    ae = a[a.pid.isin(el)]; fm = fm[fm.pid.isin(el)]
    feats = [c for c in F[role].columns if c not in ('n', 'spd_n')]
    out = []
    for i in range(NS):
        hh = pd.Series(rng.randint(0, 2, len(mids)), index=mids)
        q = [r5.full_pipeline(ae[ae.match_id.map(hh).values == h], el) for h in (0, 1)]
        fs = []
        for h in (0, 1):
            x = fm[fm.match_id.map(hh).values == h].groupby('pid').sum(numeric_only=True)
            r = x[feats].div(x.n, axis=0)
            if role == 'bowl':
                r['speed'] = (x.speed / x.spd_n.replace(0, np.nan)).fillna(0) * (1 - r.spin.round())     # pace bowlers only
            fs.append(r)
        cm = q[0][1].index.intersection(q[1][1].index)
        folds = rng.randint(0, 5, len(cm))
        row = {}
        for h in (0, 1):
            y = q[1 - h][0].T0_rP.loc[cm].values * 1e4
            lev = q[h][1].level_eb.loc[cm].values * 1e4
            X = fs[h].loc[cm].values
            cc = lambda p: np.corrcoef(p, y)[0, 1]
            row.setdefault('Level alone', []).append(cc(lev))
            for al in (10, 40, 150):
                row.setdefault(f'Level + traits (ridge {al})', []).append(cc(ridge_cv(np.c_[lev, X], y, al, folds)))
            row.setdefault('traits alone (ridge 40)', []).append(cc(ridge_cv(X, y, 40, folds)))
            row.setdefault('Level alone, refit (ridge 40)', []).append(cc(ridge_cv(lev[:, None], y, 40, folds)))
            for j, fnm in enumerate(feats):
                row.setdefault('  trait: ' + fnm, []).append(cc(X[:, j]))
        out.append({k: np.mean(v) for k, v in row.items()})
    print(f"\n==== {role}\n", pd.DataFrame(out).mean().round(3).to_string())

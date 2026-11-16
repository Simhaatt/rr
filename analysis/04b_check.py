import sys
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
b = pd.read_pickle(sys.argv[1] + "/balls.pkl")
c = b[(b.innings == 2) & (b.y != 0.5)].copy()
c['rrr'] = c.need / c.balls_rem * 6
for lo, hi in [(1, 6), (7, 12), (13, 24), (25, 36), (37, 60)]:
    d = c[c.balls_rem.between(lo, hi)]
    print(f"balls_rem {lo}-{hi}")
    print(d.groupby(pd.cut(d.rrr, [0, 4, 6, 8, 10, 12, 14, 17, 20, 30, 200]), observed=True).agg(pred=('wp_pre', 'mean'), actual=('y', 'mean'), n=('y', 'size'), matches=('match_id', 'nunique')).round(3))
d = c[c.balls_rem <= 24]
print(d.groupby([pd.cut(d.rrr, [0, 6, 9, 12, 15, 200]), pd.cut(d.wk_before, [-1, 3, 5, 7, 9])], observed=True).agg(pred=('wp_pre', 'mean'), actual=('y', 'mean'), n=('y', 'size')).round(3))

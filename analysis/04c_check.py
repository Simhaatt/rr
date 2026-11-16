import sys
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
b = pd.read_pickle(sys.argv[1] + "/balls.pkl")
c = b[(b.innings == 2) & (b.y != 0.5) & (b.legal_before % 6 == 0)]
c = c.drop_duplicates(['match_id', 'legal_before'])  # first delivery at the start of each over
for br in (6, 12, 18, 24):
    d = c[c.balls_rem == br]
    print("start of over, balls_rem", br)
    print(d.groupby(pd.cut(d.need, [0, 6, 9, 12, 15, 18, 22, 26, 30, 36, 45, 60, 300]), observed=True).agg(pred=('wp_pre', 'mean'), actual=('y', 'mean'), n=('y', 'size')).round(3).T.to_string())

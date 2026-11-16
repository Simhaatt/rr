"""Builds explainer/index.html from the final pipeline outputs, so every number on the page matches the report."""
import json, pickle, os, html
import numpy as np, pandas as pd
O = 'F:/rr/output'; E = 'F:/rr/explainer'
S = json.load(open(f'{O}/summary.json')); R = pickle.load(open(f'{O}/results.pkl', 'rb')); CON = pickle.load(open(f'{O}/consensus.pkl', 'rb'))
CT = json.load(open(f'{O}/contest_summary.json')); WS = json.load(open(f'{O}/weights_summary.json'))
_bp = 'F:/rr/kaggle/brief_version/output/consensus.pkl'; BRF = pickle.load(open(_bp, 'rb'))['summary'] if os.path.exists(_bp) else None
U = 1e4; cs = CON['summary']; WR = R['W_ROLE']
TAB = {}
for role in ('bat', 'bowl'):
    t = R['res'][role]['table']; C = CON[role]
    t = t.assign(rank_main=t['rank'])
    for a, b in (('score', 'score_c'), ('rank', 'rank_c'), ('p_top10', 'p_top10_c'), ('sens', 'sens_c'), ('tier', 'tier'), ('nm', 'n_models_top10')):
        t[a] = C[b].reindex(t.index)
    TAB[role] = t.sort_values('rank')
bat, bowl = TAB['bat'], TAB['bowl']
pc = lambda v: f"{100 * v:.0f}%"
sg = lambda v: f"{v:+.1f}".replace('-', '\u2212')
f2 = lambda v: f"{v:.2f}".replace('-', '\u2212')
ordn = lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
esc = html.escape
out = []
w = out.append


def table(head, rows, right=()):
    h = ''.join(f'<th class="r">{x}</th>' if i in right else f'<th>{x}</th>' for i, x in enumerate(head))
    b = '\n'.join('<tr>' + ''.join(f'<td class="r">{x}</td>' if i in right else (f'<td class="wrap">{x}</td>' if isinstance(x, str) and len(x) > 40 else f'<td>{x}</td>') for i, x in enumerate(r)) + '</tr>' for r in rows)
    w(f'<div class="scroll"><table>\n<tr>{h}</tr>\n{b}\n</table></div>')


def fig(name, alt, cap):
    w(f'<figure><div class="img"><img src="figs/{name}.png" alt="{esc(alt)}"></div>\n<figcaption>{cap}</figcaption></figure>')


P = lambda s: w(f'<p>{s}</p>')
H2 = lambda i, s: w(f'\n<h2 id="{i}">{s}</h2>')
H3 = lambda s: w(f'<h3>{s}</h3>')
NOTE = lambda s: w(f'<div class="note">{s}</div>')
UL = lambda items: w('<ul>\n' + '\n'.join(f'<li>{x}</li>' for x in items) + '\n</ul>')
br, wr = S['bat_rel'], S['bowl_rel']
ck = S['checks']; mo = S['more']; sq = S['squeeze']; ko = S['knockout']; dt = S['design_tests']
nfirm = lambda tab: int((tab.head(10).tier == 'Firm').sum())

w(open(f'{E}/index.html', encoding='utf-8').read().split(chr(10) + '<main>')[0].rstrip())  # reuse the page's own head and styles
w('\n<main>')
w('<span class="eyebrow">SupeRR Selector Season 2 · Round 1</span>')
w('<h1>How we worked out who delivers under pressure</h1>')
w('<p class="lede">The whole project from scratch: the raw files, the method, every test it went through, and the two top-10 lists it produced. No statistics background is assumed.</p>')

# ---- where things stand
rows = ''.join(f"<tr><td class='r'>{i + 1}</td><td>{esc(a)}{' <span class=\"num\">·</span>' if bat.iloc[i].tier != 'Firm' else ''}</td><td>{esc(b)}{' <span class=\"num\">·</span>' if bowl.iloc[i].tier != 'Firm' else ''}</td></tr>" for i, (a, b) in enumerate(zip(bat.name.head(10), bowl.name.head(10))))
w(f'''<div class="plain">
<span class="step">The lists to enter on Unstop</span>
<div class="scroll" style="margin-top:0.4rem"><table>
<tr><th class="r">#</th><th>Batters</th><th>Bowlers</th></tr>
{rows}
</table></div>
<p style="margin-top:0.7rem">A dot (·) marks a contested place: a near-equal player could hold it instead (<a href="#firm">section 11</a>). The rest are firm. The data pack has two players called Rashid Khan; ours is the Afghanistan and Gujarat Titans leg-spinner (player_id 4722), not the Nepal player. The most important finding is that nobody's ability to rise under pressure can be told apart from luck in this data, so the lists rank the players worth most in pressure situations (<a href="#trust">section 13</a>).</p>
</div>''')
w('''<nav aria-label="Contents"><ol>
<li><a href="#task">The task</a></li><li><a href="#idea">The idea in one minute</a></li><li><a href="#data">The data</a></li>
<li><a href="#wp">Win probability</a></li><li><a href="#pressure">Pressure balls</a></li><li><a href="#kinds">Two more kinds of pressure</a></li>
<li><a href="#value">Valuing each ball</a></li><li><a href="#luck">Luck, and how we discount it</a></li><li><a href="#rating">The rating</a></li>
<li><a href="#fair">Fair comparisons</a></li><li><a href="#firm">Three models; firm and contested places</a></li><li><a href="#results">The results</a></li>
<li><a href="#trust">How far to trust them</a></li><li><a href="#built">How the method was built</a></li><li><a href="#limits">What the method cannot see</a></li>
<li><a href="#files">Files and glossary</a></li></ol></nav>''')

# ---- 1
H2('task', '1. The task')
P('Rajasthan Royals supplied a data pack of T20 matches, IPL and internationals, and asked one question: <strong>which batters and bowlers are the most reliable under pressure?</strong> Two things go in on Unstop: an ordered top 10 for batters and one for bowlers, picked from dropdowns, and a methodology report. The report must say how we defined pressure, how we measured performance, how we kept comparisons fair, what evidence backs the lists, and where the method is weak.')
P('The brief adds that the answer must come from the supplied data, not from reputation. That sentence shaped most decisions below.')

# ---- 2
H2('idea', '2. The idea in one minute')
P('Most people define pressure with rules of thumb: the last five overs, a required rate above 10, three wickets down early. Those rules clash, and each needs an arbitrary cut-off. We used one quantity that combines them.')
NOTE('<b>Pressure is what the next ball can do to the result.</b> If one ball can swing the chance of winning a long way, it is a pressure ball. If the match is already decided, or there is plenty of time to recover, it is not.')
P('Once pressure is defined that way, performance uses the same currency: how much did a player\'s balls change their team\'s chance of winning? The rest of the work makes that comparison fair and stops luck from deciding it.')
w('''<pre class="mermaid">
flowchart LR
  A[Ball-by-ball data] --> B[Match state before every ball]
  B --> C[Win-probability model]
  C --> D[Leverage: how much the next ball can swing the result]
  D --> E[Pressure balls: top 25% by leverage]
  K[Knockout weight] --> E
  C --> F[Value of each ball: actual and exam]
  Q[Squeeze: what has just happened] --> G
  E --> G[Rates on pressure and ordinary balls]
  F --> G
  G --> H[Adjust for opponents, discount luck]
  H --> I[Level and consistency]
  I --> J[Average over three models]
  J --> L[Top 10 with firm and contested places]
</pre>''')

# ---- 3
H2('data', '3. The data')
P(f"The pack has six files: competitions, teams, matches, the ball-by-ball log, squad lists and playing elevens. The ball-by-ball log does most of the work. Each row is one delivery: who bowled, who faced, the runs, the extras and any wicket.")
table(['Step', 'Count', 'Why'], [['Matches in the pack', S['matches_total'], 'IPL 2022 to 2026 and T20 internationals'], ['With ball-by-ball records', S['matches_with_bbb'], 'The rest have only a scorecard line'],
                               ['Rain-revised matches removed', S['dls_excluded'], 'A revised target makes "runs needed" ambiguous'], ['Abandoned or incomplete removed', 30, 'No clean result to learn from'],
                               ['<strong>Matches analysed</strong>', f"<strong>{S['matches_used']}</strong>", f"{S['ipl_matches']} IPL and {S['intl_matches']} international; {S['balls_used']:,} deliveries"]], right=(1,))
UL(['<strong>Which balls are legal.</strong> The extras column marks wides, "Overstep" and "other". We guessed that "other" is a no-ball and tested it: counting all three as illegal makes 31,757 of 32,338 overs exactly six balls long; all but one of the rest are overs that never finished (the exception has seven legal balls, a miscount by the umpire). The guess holds.',
    '<strong>The state before every ball.</strong> We rebuilt balls left, runs, wickets lost and, in a chase, runs needed. The rebuilt totals give the recorded winner in all 794 decided matches.',
    '<strong>Par.</strong> A total of 180 is ordinary in IPL 2026 and big in a 2022 international. For each competition and year we computed the average first-innings total (171 in IPL 2022, 196 in IPL 2026, 166 to 180 in internationals) and gave it to the model.'])

# ---- 4
H2('wp', '4. Win probability')
P('A win-probability model answers one question: from this exact situation, how often does the batting side go on to win? Needing 30 off 12 with five wickets left might be 22%. The model learns this from what happened in hundreds of matches from similar situations.')
P('In a chase it uses balls left, runs needed, wickets in hand and par, and averages two standard statistical models: a logistic regression, which turns a weighted sum of cricket-shaped inputs such as the required rate into a probability, and a tree model, built from many small decision rules and forced to obey common sense, so more balls or wickets can never lower the chance of winning. In the first innings a logistic regression uses runs, wickets, balls left, par and a projected total.')
P(f"A model can look brilliant on matches it has already seen, so we split the matches into five groups and always predicted each group with a model fitted on the other four. On those unseen matches, the model rates the eventual winner's position above the loser's {pc(S['wp2']['auc'])} of the time in the chase and {pc(S['wp1']['auc'])} in the first innings, where far less is known before a target exists (50% would be guessing; statisticians call this the AUC).")
fig('f2_calibration', 'Calibration chart: predicted against actual win rate, points close to the diagonal', 'When the model says 70%, the batting side wins about 70% of the time. Points on the diagonal mean the probabilities can be trusted.')
cal = mo['calibration']
table(['Segment', 'Matches', 'Predicted', 'Actual', 'Gap'], [[r['segment'] + (' season' if r['segment'].isdigit() else ''), r['matches'], f"{r['pred']:.3f}", f"{r['actual']:.3f}", f"{r['ece']:.3f}"] for r in cal], right=(1, 2, 3, 4))
P('The model stays calibrated within each segment. The gap is largest for the 28 knockout matches, where there is least to judge by.')

# ---- 5
H2('pressure', '5. Pressure balls')
P('For each ball we ask eight what-ifs: if this ball were a dot, 1, 2, 3, 4, 6, a wicket or a wide, where would the chance of winning go? The ball\'s <strong>leverage</strong> is the average size of those moves, weighted by how often each outcome happens. We scale it so the average ball has a leverage of 1.0; a ball with leverage 5 has five times the usual amount at stake.')
fig('f1_leverage_maps', 'Heat maps of leverage by match situation', 'Leverage peaks when the target is hard but reachable and time is short, and falls to nothing once either side has the match won. In the last over of a tight chase a ball carries 7 to 14 times the stakes of an average ball.')
P(f"<strong>A pressure ball is any delivery with leverage of at least {S['thr']:.2f}, the top quarter of all deliveries.</strong> With the knockout weight of section 6 that gives {S['press_balls']:,} pressure balls. We tried the top 33%, 25%, 20%, 15% and 10%; a stricter bar sounds purer but leaves each player too few balls to measure, and the top quarter was the best balance.")
pv = pd.DataFrame(S['press_vs_ord']); pcs = S['press_by_close']; c = S['composition']
sw = 'Mean |win-prob swing| per ball (pts)'
beh = {r['name']: r for r in mo['behaviour']}
UL([f"<strong>The stakes are three times higher.</strong> An average pressure ball moves win probability by {pv.loc[sw, 'Pressure']:.1f} points, an ordinary one by {pv.loc[sw, 'Ordinary']:.1f}. Run and wicket rates barely differ, so the set picks out situations, not a phase where the numbers run naturally higher.",
    f"<strong>It follows close matches.</strong> {pcs['Chase won in final over']:.0f}% of balls are pressure balls in chases won in the last over, against {pcs['Chase won with 19+ balls left']:.0f}% in chases won with 19 or more balls to spare.",
    f"<strong>It covers every phase but one.</strong> Most pressure balls come in the chase ({pc(c['inn2_PP'])} in the powerplay, {pc(c['inn2_MID'])} in the middle overs, {pc(c['inn2_DEATH'])} at the death). No middle-overs ball of a first innings qualifies, because a single ball changes least then.",
    f"<strong>Players behave differently on these balls.</strong> In the same innings and over, the dot-ball rate falls by {abs(100 * beh['Dot ball']['diff']):.1f} percentage points, batters play a few more scoops and switch hits, and pace bowlers bowl more full tosses, which are missed yorkers. Boundary and wicket rates do not move."])
fig('f3_example_match', 'Win probability and leverage across one close match', 'One real match. Top: the chance of winning for the side batting first. Bottom: leverage, with pressure balls in pink. Almost all come late in the chase.')

# ---- 6
H2('kinds', '6. Two more kinds of pressure')
H3('Squeeze: what has just happened')
P(f"Leverage cannot see how a side reached its score. Sixty for two after eight overs feels different if the last over was six dots. We call the batting side <strong>squeezed</strong> when at least four of the last six legal balls were dots or a wicket fell in them ({pc(sq['share_bat'])} of balls), and a bowler <strong>under attack</strong> after two or more boundaries in six ({pc(sq['share_bowl'])}).")
P(f"To test whether squeeze matters, we compared each player with themselves. The same batter is worth {abs(sq['bat']['effect']):.1f} win-probability points per 100 balls less when squeezed, far outside the margin of error ({sq['bat']['se']:.1f}); the same bowler is worth {abs(sq['bowl']['effect']):.1f} less when under attack.")
fig('f10_squeeze', 'Runs fall after dot balls; the same player does worse when squeezed', 'Left: the more dots just bowled, the fewer runs come off the next ball. Right: the same player, squeezed against not squeezed, with the margin of error.')
P(f"So every ball is now compared with the average player in the same squeeze state, and a batter who keeps walking in after a collapse is no longer judged against too easy a standard. We did not rank players on how they respond to squeeze, because that response does not repeat from one half of the matches to the other ({sq['bat']['rel']:.2f} for batters). A ranking built on it would rank luck.")
H3('The importance of the match')
mwc = S['mw_counts']
P(f"Leverage measures how much a ball matters to the match, not how much the match matters. {ko['ko_matches']} of the {ko['matches']} matches are knockouts: IPL playoffs, two Asia Cup finals, tri-series finals and A-team knockouts; there is no World Cup in the pack. Batters do no better in knockouts ({sg(ko['bat_all balls']['effect'])} points per 100 balls); bowlers do {ko['bowl_all balls']['effect']:.1f} better, which 30 matches cannot confirm.")
table(['Kind of match', 'Matches', 'Weight'], [['IPL or Asia Cup final', mwc.get('1.5', 0), '×1.5'], ['IPL qualifier or eliminator', mwc.get('1.25', 0), '×1.25'], ['Tri-series final, A-team semi-final or final', mwc.get('1.125', 0), '×1.125'], ['Everything else', mwc.get('1.0', 0), '×1']], right=(1, 2))
kt = dt['knockout']
P(f"The weight is light on purpose, and the held-out test chose its size: at this scale nothing moves by more than 0.01, while eight times the weight drops bowlers' repeatability from {kt[0][4]:.2f} to {kt[4][4]:.2f}. Only players in strong sides reach knockouts at all, so a heavy weight would start to rank teams.")

# ---- 7
H2('value', '7. Valuing each ball')
P('Each ball is worth the change in the batting side\'s chance of winning, from just before it to just after it. The batter earns that change and the bowler its negative. A dot ball when 14 an over is needed might cost the batter 3 points; the same dot at 4 an over costs almost nothing. Strike rate, economy, dots and boundaries need no separate rules, because each ball is priced by what it did to the match.')
UL(['Wides count against the bowler only. A run-out of the non-striker does not count against the striker, and run-outs do not count for the bowler.',
    'A player\'s rate is their total value divided by the total leverage of their balls, which puts pressure and ordinary balls on one scale. Rates are win-probability points per 100 balls of average leverage, with 0 as the average player. A rate of +40 means 40 percentage points of win probability more than an average player over 100 typical balls.',
    'No single ball counts for more than three times the average, so one last-ball six cannot define a career.'])
H3('The Pressure Exam')
P('Raw win probability added has a flaw. Two batters who each hit a six, a four, three singles and a dot under pressure can end with very different totals, because one six came when it was worth 4 points and the other when it was worth 12. A batter rarely chooses which ball that is. This timing luck is much of the noise in any "clutch" number.')
P(f"Because the model can price every possible outcome of every ball, we set every player the same paper. Balls are grouped into {S['exam_strata']} kinds of situation, and within each kind every outcome gets one fixed mark: its average worth across all balls of that kind. A player's exam value for a ball is the mark for what they did.")
ex = S['exam_examples']
table(['Pressure situation', 'Dot', 'Single', 'Four', 'Six', 'Wicket'], [[k_, sg(v['dot']), sg(v['1']), sg(v['4']), sg(v['6']), sg(v['wicket'])] for k_, v in ex.items()], right=(1, 2, 3, 4, 5))
P('These are the marks, in win-probability points per unit of leverage. A wicket in a chase powerplay costs about twice what it costs at the death, and a single is a small loss at the death but a small gain in the powerplay. Because the exam is only an approximation, each ball\'s final value is <strong>half exam mark, half actual win probability added</strong>.')

# ---- 8
H2('luck', '8. Luck, and how we discount it')
P('This part decides whether the lists mean anything. A batter\'s pressure record over 200 balls is mostly luck: a few edges that fell safe, one dropped catch, and the number looks heroic.')
P('The test for luck is simple. Toss a coin for every match to make two piles, rate every player on each pile, and see whether the two ratings agree. Real skill shows up in both; luck does not. The agreement is a correlation from 0 (pure luck) to 1 (perfectly repeatable). We repeat the coin tosses 100 times and average.')
P('The fix is <strong>shrinkage</strong>: every measured number is pulled towards the average player, harder when there is less evidence. A player with 150 balls is pulled a long way; one with 800 hardly at all. How hard depends on how much real spread there is between players compared with the noise at that sample size.')
mr, mw_ = S['bat_mean_rel'], S['bowl_mean_rel']
table(['Measurement', 'Batters: share kept', 'Bowlers: share kept'], [['Overall level, run part', pc(mr['R_rA_rel']), pc(mw_['R_rA_rel'])], ['Overall level, wicket part', pc(mr['K_rA_rel']), pc(mw_['K_rA_rel'])], ['Uplift (pressure minus usual)', pc(mr['up_tot_rel']), pc(mw_['up_tot_rel'])]], right=(1, 2))
k1 = bat.set_index('name').loc[bat.name.iloc[0]]
P(f"The last row matters most: a player's \"extra under pressure\" keeps only {pc(mw_['up_tot_rel'])} to {pc(mr['up_tot_rel'])} of its measured size, because the data cannot tell it from chance. For example, {esc(bat.name.iloc[0])}'s actual pressure rate is {sg(k1.T_rP * U)} on only {k1.n_P:.0f} pressure balls; shrinkage cuts the level to {sg(k1.level_eb * U)}, still the best in the pool, but by a believable margin.")

# ---- 9
H2('rating', '9. The rating')
P('The rating has two parts.')
w(f'''<dl>
<dt>Level</dt><dd>What a player is worth per pressure ball. It starts from the player's level on all their balls, the most reliable thing we know, and moves towards their pressure record only as far as that evidence allows. That is how each player is compared with their usual performance.</dd>
<dt>Consistency</dt><dd>How rarely a player loses a pressure ball: the share of pressure balls with a value of zero or better, compared with the average for the same kind of ball. Counting balls instead of good matches gives about ten times the evidence.</dd>
</dl>''')
w(f'<div class="formula">Batters: {WR["bat"]["level"]:.2f} × Level + {WR["bat"]["cons"]:.2f} × Consistency &nbsp;&nbsp; Bowlers: {WR["bowl"]["level"]:.2f} × Level + {WR["bowl"]["cons"]:.2f} × Consistency</div>')
P(f"We did not pick those weights by feel. We fitted them to predict what players actually did under pressure in matches the rating had not seen. A third part, <strong>uplift</strong> (pressure rate minus usual rate), was tested and got a weight of zero: it predicted nothing beyond what Level already carries. A player who rises under pressure still gains, because Level includes uplift. Bowlers lean on consistency because, for them, it predicts unseen matches better than level does.")
if BRF:
    P(f"The brief's suggested split of 50% level, 25% uplift and 25% consistency was our previous version. Against it, the fitted weights made the rating more repeatable ({cs['bat']['rel']:.2f} against {BRF['bat']['rel']:.2f} for batters, {cs['bowl']['rel']:.2f} against {BRF['bowl']['rel']:.2f} for bowlers) and steadier from season to season ({cs['bat']['yy']:.2f} against {BRF['bat']['yy']:.2f}; {cs['bowl']['yy']:.2f} against {BRF['bowl']['yy']:.2f}). With the old weights the batting ten would be the same names in another order, and the bowling ten would have Mitchell Santner instead of Ravichandran Ashwin.")

# ---- 10
H2('fair', '10. Fair comparisons')
w(f'''<dl>
<dt>Opponents</dt><dd>A boundary off a top bowler earns more than one off a weak bowler. We estimate every batter's and bowler's quality from all the data at once, then adjust each ball for who was at the other end, including how good they are against that type: a batter against pace or spin, a bowler against left- or right-handers.</dd>
<dt>Role and phase</dt><dd>Because each ball is valued from its exact match state, a finisher is not punished for facing hard situations, and a death bowler is not punished for going at 10 an over when 11 was expected.</dd>
<dt>What has just happened</dt><dd>Each ball is compared within its squeeze state (section 6).</dd>
<dt>IPL and internationals</dt><dd>Both feed one model. Par handles scoring levels; the opponent adjustment handles strength. Pressure balls are about equally common in each (26.0% and 24.2%).</dd>
<dt>Who is eligible</dt><dd>At least 150 pressure balls across 15 matches: {S['bat_n']} batters and {S['bowl_n']} bowlers.</dd>
</dl>''')

# ---- 11
H2('firm', '11. Three models; firm and contested places')
P('Everything depends on the win-probability model, and no model is exactly right. Reasonable models disagree slightly about how much a ball mattered, and for players a few hundredths of a rating point apart that is enough to swap them. So we ran the whole method three times, once with each of three models, and averaged the ratings. This ran on Kaggle, because this PC ran out of memory; Kaggle reproduced the main model\'s results here to twelve decimal places.')
m_ = 'Main (blend)'
table(['Model', 'How it works', 'Batters: same top 10', 'Bowlers: same top 10'], [[mm, d_, f"{cs['bat']['agreement'][mm]['overlap']} of 10", f"{cs['bowl']['agreement'][mm]['overlap']} of 10"] for mm, d_ in zip(CON['models'], ['Logistic regression plus boosted trees in the chase', 'Logistic regression, both innings', 'Boosted trees, both innings'])], right=(2, 3))
P(f"Averaging does not make the rating more accurate; the tests change by 0.03 or less either way. What it buys is that no single modelling choice decides the close places.")
P('The data cannot separate players a few hundredths of a point apart, so each list splits into two groups. A place is <strong>firm</strong> when all three models put the player in the top 10 and the player stays there in at least 60% of resamples. (A resample redraws the matches at random, so some appear twice and others not at all, and re-rates everyone; we did this 500 times to see how much each rank could move.) Every other place is <strong>contested</strong>. The dropdowns need ten names, so contested places follow rating order, but that order means little.')
for role, lab in (('bat', 'Batter'), ('bowl', 'Bowler')):
    tt = TAB[role][TAB[role].tier != '']
    table(['#', lab, 'Tier', 'Top 10 under models', 'Stays top 10', 'Top 10 under weightings'], [[int(r['rank']), esc(r['name']), '<strong>Firm</strong>' if r.tier == 'Firm' else 'Contested', f"{int(r.nm)} of 3", pc(r.p_top10), f"{int(round(8 * r.sens))} of 8"] for _, r in tt.iterrows()], right=(0, 3, 4, 5))
b = CT['bat']; sn = lambda n: n.split()[-1]
table(['Batter'] + [sn(n) for n in b['group']], [[esc(n)] + ['\u2013' if i == j else pc(b['H'][i][j]) for j in range(len(b['group']))] for i, n in enumerate(b['group'])], right=tuple(range(1, len(b['group']) + 1)))
wg = CT['bowl']; av = dict(zip(wg['group'], wg['ahead']))
P(f"The head-to-head table shows how often the row player rates above the column player across resamples. {esc(b['group'][0])}, {esc(b['group'][1])} and {esc(b['group'][2])} are a three-way tie, but each beats {esc(b['group'][3])} about three times in five, and no batter outside the ten comes close; the four contested batting names are settled, only their order is open. Among bowlers, {esc(wg['group'][0])} and {esc(wg['group'][1])} win {pc(av[wg['group'][0]])} and {pc(av[wg['group'][1]])} of their head-to-heads on average, so those places are reasonably secure. The last three bowling places are close, and several bowlers just outside could fill them.")

# ---- 12
H2('results', '12. The results')
ex_rows = lambda t: [[int(r['rank']), esc(r['name']), int(r.n_P), sg(r.T_rN * U), sg(r.T_rP * U), sg(r.exam_rP * U), sg(r.level_eb * U), sg(r.cball * 100), pc(r.p_top10)] for _, r in t.head(10).iterrows()]
hd = ['#', 'Player', 'Pressure balls', 'Usual', 'Pressure', 'Exam', 'Level', 'Consistency', 'Stays top 10']
H3('Batters')
table(hd, ex_rows(bat), right=(0, 2, 3, 4, 5, 6, 7, 8))
P('"Usual" and "Pressure" are actual value on ordinary and pressure balls; "Exam" is the exam mark under pressure; "Level" is the final shrunk level; "Consistency" is pressure balls not lost, in points above the average for the same kind of ball.')
fig('f4_top10_batters', 'Top ten batters: usual, pressure, exam and final level, with consistency and robustness', 'Navy: ordinary balls. Pink: pressure balls. Triangle: exam mark. Diamond: final level. Iyer and Kohli rise most under pressure; Tim David is the reverse, elite overall and slightly below that norm under pressure.')
H3('Bowlers')
table(hd, ex_rows(bowl), right=(0, 2, 3, 4, 5, 6, 7, 8))
fig('f5_top10_bowlers', 'Top ten bowlers: usual, pressure, exam and final level, with consistency and robustness', 'Same layout for bowlers.')
jb = bowl.set_index('name').loc['Jasprit Bumrah']; bs_ = (-bowl.set_index('name').base_eb).rank(method='min').astype(int); up_ = (-bowl.set_index('name').uplift_eb).rank(method='min').astype(int); cs_ = (-bowl.set_index('name').cball).rank(method='min').astype(int)
NOTE(f"<b>Jasprit Bumrah ranks {ordn(int(jb['rank']))}, and the reason is unusual.</b> Bumrah's overall level is {ordn(bs_['Jasprit Bumrah'])} of 97 and consistency {ordn(cs_['Jasprit Bumrah'])}, but in this data Bumrah's pressure balls did less than ordinary ones (a wicket every {jb.SR_P:.0f} balls against {jb.SR_N:.0f}), so the uplift ranks {ordn(up_['Jasprit Bumrah'])}. The exam marks the pressure bowling higher ({sg(jb.exam_rP * U)}) than its actual result ({sg(jb.T_rP * U)}), so timing explains part of the gap. Consistency, which carries most of the bowling weight, keeps Bumrah firm.")
H3('Where each player\'s value comes from')
for role, cols, lab in (('bat', ['Powerplay', 'Middle', 'Death', 'Chasing', 'v pace', 'v spin'], 'Batter'), ('bowl', ['Powerplay', 'Middle', 'Death', 'Defending', 'v left-handers', 'v right-handers'], 'Bowler')):
    rk_ = dict(zip(TAB[role].name, TAB[role]['rank'])); pr = sorted(mo['profiles'][role], key=lambda r: rk_.get(r['name'], 99))
    table([lab] + cols, [[f"{int(rk_.get(r['name'], r['rank']))}. {esc(r['name'])}"] + [(f"{sg(r[c_][0])} ({r[c_][1]})" if c_ in r and r[c_][0] is not None else '\u2013') for c_ in cols] for r in pr], right=tuple(range(1, len(cols) + 1)))
P('Value per 100 pressure balls, with the number of pressure balls in brackets; splits under 40 balls are mostly noise. The lists are not one type of player: some batters earn their value in the powerplay, others in the middle overs or at the death.')
fig('f7_pressure_vs_usual', 'Every eligible player: ordinary against pressure value', 'Every eligible player. Above the dashed line means better under pressure than usual. Pink marks the top 10.')

# ---- 13
H2('trust', '13. How far to trust them')
sb = lambda r: 2 * r / (1 + r)
P(f"The final rating agrees {cs['bat']['rel']:.2f} between random halves of the matches for batters and {cs['bowl']['rel']:.2f} for bowlers. Each half has only half the data, so the full sample's reliability is about {sb(cs['bat']['rel']):.2f} and {sb(cs['bowl']['rel']):.2f}. In plain words, a little over half of the differences between players in the rating are real, and the rest is noise that no method can remove from this much data. The lists show who delivered most reliably in 2022 to 2026 once luck is discounted; they are not a forecast.")
fig('f6_rank_intervals', 'Rank and resampled rank range for the top twenty', 'We redrew the matches at random 500 times and re-rated everyone. Dot: final rank. Bar: where the rank falls in 80% of redraws.')
H3('Other checks')
rb = S['robust']
UL([f"<strong>Season to season.</strong> Run on each season alone, the rating carries over at {cs['bat']['yy']:.2f} for batters and {cs['bowl']['yy']:.2f} for bowlers. Raw uplift carries over at {f2(br['yy_upB'])} and {f2(wr['yy_upB'])}: zero, as baseball finds for its clutch measures.",
    f"<strong>Different data.</strong> We reran the method on eleven altered datasets: IPL only, internationals only, each season left out, other pressure bars and eligibility bars. On average {np.mean([r[3] / max(r[2], 1) for r in rb]) * 100:.0f}% of our batters and {np.mean([r[8] / max(r[7], 1) for r in rb]) * 100:.0f}% of our bowlers keep their places, and those who drop out are mostly the contested ones.",
    f"<strong>Unseen matches.</strong> Built on half the matches, the rating predicts what players actually did under pressure in the other half at {br['pred_score_to_rawP']:.2f} for batters and {wr['pred_score_to_rawP']:.2f} for bowlers. A player's own raw pressure record manages {br['pred_rawP_to_rawP']:.2f} and {wr['pred_rawP_to_rawP']:.2f}; economy rate manages {wr['pred_convA_to_rawP']:.2f}. For batters, plain career strike rate does as well as the rating ({br['pred_convA_to_rawP']:.2f}): batting under pressure is mostly general scoring ability carrying over."])
fig('f9_out_of_sample', 'Bar chart of how well each measure predicts unseen matches', 'Each measure built on one half of the matches, scored against actual pressure value in the other half.')
H3('Is there such a thing as clutch?')
P(f"We shuffled which balls count as pressure within each player, 400 times, and compared the spread of uplift between players with the real one. If some players truly rose under pressure, the real spread would be bigger. It is not: {ck['bat_perm']['sd_obs']:.1f} real against {ck['bat_perm']['sd_null_mean']:.1f} shuffled for batters, {ck['bowl_perm']['sd_obs']:.1f} against {ck['bowl_perm']['sd_null_mean']:.1f} for bowlers.")
P(f"The test was strong enough to find clutch if it were large. When we planted clutch differences of 15 points per 100 balls, it found them {pc(mo['bat_power']['power']['15'])} of the time for batters and {pc(mo['bowl_power']['power']['15'])} for bowlers; players differ in overall ability by about {mo['bat_power']['level_sd']:.0f} and {mo['bowl_power']['level_sd']:.0f} points. So any clutch skill in this pool is clearly smaller than the gaps in overall quality, and the lists rank who is worth most under pressure, not who rises to it.")
P(f"Baseball's standard clutch statistic, the Clutch score published by the analytics site FanGraphs, agrees with our uplift measure (correlation {ck['bat_fg']['corr_raw_uplift']:.2f}) and is just as unstable. Ranked on it alone, the lists would be led by {esc(ck['bat_fg']['top5'][0][0])} and {esc(ck['bowl_fg']['top5'][0][0])}, whom our rating places {ordn(int(ck['bat_fg']['top5'][0][2]))} and {ordn(int(ck['bowl_fg']['top5'][0][2]))}.")

# ---- 14
H2('built', '14. How the method was built')
P('The method went through three rounds, each tested the same way: build on half the matches, check against the other half, repeat 40 to 100 times. Only changes that improved those checks were kept.')
table(['Version', 'Batters: agrees between halves', 'Bowlers: agrees between halves'], [['1: first submission', '0.22', '0.19'], ['2: anchored on all balls, better shrinkage', '0.27', '0.21'], ['3: exam, squeeze, match-ups, ball-level consistency', f"{dt['round3'][4][1]:.2f}", f"{dt['round3'][4][4]:.2f}"], ['Final: fitted weights, three models averaged', f"{cs['bat']['rel']:.2f}", f"{cs['bowl']['rel']:.2f}"]], right=(1, 2))
fig('f11_design_tests', 'Bar chart comparing design choices', 'The first round of tests. Coloured bars mark the adopted design; dashed lines mark the previous version.')
H3('Ideas that passed')
UL(['<strong>The Pressure Exam,</strong> blended half and half with actual value (section 7).', '<strong>The squeeze adjustment</strong> (section 6).',
    f"<strong>Counting balls instead of matches</strong> for consistency, together with judging opponents by type: the largest single gain of the project, raising bowlers' repeatability from {dt['round3'][0][4]:.2f} to {dt['round3'][4][4]:.2f}.",
    '<strong>A light knockout weight</strong> (section 6), <strong>fitted weights</strong> (section 9) and <strong>averaging three models</strong> (section 11).'])
H3('Ideas that failed, including yours')
ins = S['insights']; ar = S['allround']
UL([f"<strong>Response curve.</strong> Fitting each player a line for how value changes as stakes rise made batters' uplift more repeatable but the overall rating worse.",
    '<strong>Venue adjustment.</strong> It made batters\' ratings worse; par and the opponent adjustment already handle scoring conditions.',
    '<strong>Recency weighting.</strong> Weighting recent seasons more did not predict the latest matches any better.',
    '<strong>Playing-style traits.</strong> Adding shot and delivery types made predictions much worse; with 97 players they fit noise.',
    f"<strong>Home and away.</strong> The same player is no different away ({sg(ins['bat: away v home, all balls']['effect'])} points per 100 balls for batters), and individual home-away gaps do not repeat.",
    f"<strong>Form slumps.</strong> After three low scores the same batter does no worse ({sg(ins['bat: after a form slump v otherwise, all balls']['effect'])}).",
    f"<strong>An all-round bonus.</strong> Contributing with bat and ball in the same match happened {ar['dbl']} times, but the brief ranks batters and bowlers separately, and the bonus rewarded a role rather than a pressure skill.",
    '<strong>Fielding.</strong> The data has no record of catches or drops, so it could not be tested.'])
P('Your other points were already in the method: getting out or conceding runs when it matters is penalised through win probability, and a dot ball at the death is valued against a wicket by the same yardstick.')

# ---- 15
H2('limits', '15. What the method cannot see')
UL(['<strong>The scoreboard, not the conditions.</strong> The model knows the match state, not team strength, venue, dew, pitch or which batters are still to come.',
    '<strong>All credit goes to striker and bowler.</strong> Fielding, the non-striker\'s running and captaincy are not separated out.',
    '<strong>Finer match-ups.</strong> Opponents are judged by pace or spin and left or right hand, not by pairings such as left-arm spin to a left-hander.',
    '<strong>Judgement in the knockout weights.</strong> The data shows how much weight is affordable, not which final matters more.',
    '<strong>Excluded matches and players.</strong> Rain-revised matches are left out, and players under 150 pressure balls are not rated, however good their record.',
    '<strong>No ball-tracking.</strong> Measures such as false-shot rate would probably settle faster than outcomes, but the pack does not contain them.'])

# ---- 16
H2('files', '16. Files and glossary')
table(['File', 'What it is'], [['<code>submission/SupeRR_Selector_R1_Methodology_Report.pdf</code>', 'The report to upload (a .docx copy sits beside it)'], ['<code>submission/SUBMISSION_top10_lists.md</code>', 'The two lists to enter'],
                               ['<code>submission/*_consensus_ratings.csv</code>', 'Every eligible player with the final ratings and tiers'], ['<code>analysis/</code>', 'The scripts, run in order by run_all.py (section 9 of the report; code at github.com/Simhaatt/rr)'],
                               ['<code>kaggle/</code>', 'The Kaggle notebook and dataset that ran the three-model version'], ['<code>kaggle/brief_version/</code>', 'The previous version with the brief\'s 50/25/25 weights, kept for comparison']])
w('''<dl>
<dt>Win probability</dt><dd>The chance that the batting side wins from a given situation.</dd>
<dt>Leverage</dt><dd>How far the next ball can move win probability; the average ball is 1.</dd>
<dt>Pressure ball</dt><dd>A ball in the top 25% by leverage.</dd>
<dt>Squeeze</dt><dd>Pressure from what has just happened: dots or a wicket for the batting side, boundaries against the bowler.</dd>
<dt>Pressure Exam</dt><dd>Re-marking every ball at the average worth of its outcome in that kind of situation, so all players sit the same paper.</dd>
<dt>Uplift</dt><dd>A player's pressure rate minus their usual rate.</dd>
<dt>Shrinkage</dt><dd>Pulling a noisy number towards the average, harder when there is less evidence.</dd>
<dt>Split-half</dt><dd>Rating players on two random halves of the matches to see whether the ratings agree.</dd>
<dt>Resampling</dt><dd>Redrawing the matches at random many times to see how much ranks move.</dd>
<dt>Firm / contested</dt><dd>Firm: top 10 under all three models and in at least 60% of resamples. Contested: everything else near the cut.</dd>
<dt>Correlation</dt><dd>A number from &minus;1 to 1 for how closely two rankings agree; 0 means no relation.</dd>
</dl>''')
w('</main>')
w('<script type="module">import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs"; mermaid.initialize({startOnLoad: true, theme: "neutral"});</script>')
open(f'{E}/index.html', 'w', encoding='utf-8').write('\n'.join(out))
print('written', sum(len(x) for x in out))

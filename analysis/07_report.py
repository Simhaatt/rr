"""Builds the methodology report (.docx) and the submission sheet from the pipeline outputs."""
import sys, os, json, pickle
import pandas as pd, numpy as np
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
OUT = sys.argv[1]; DEST = sys.argv[2]; FIG = OUT + "/figs"
S = json.load(open(f"{OUT}/summary.json"))
R = pickle.load(open(f"{OUT}/results.pkl", 'rb'))

# ---- consensus view: the final rating is the average over three win-probability models (see 20_consensus.py)
CON = pickle.load(open(f"{OUT}/consensus.pkl", 'rb'))
_W = {'50/25/25 (base)': (0.5, 0.25, 0.25), '100/0/0': (1, 0, 0), '40/40/20': (0.4, 0.4, 0.2), '60/20/20': (0.6, 0.2, 0.2), '34/33/33': (0.34, 0.33, 0.33),
      '50/0/50': (0.5, 0, 0.5), '50/50/0': (0.5, 0.5, 0), '70/15/15': (.7, .15, .15)}
for _role in ('bat', 'bowl'):
    _t = R['res'][_role]['table']; _C = CON[_role]
    _t = _t.assign(score_main=_t.score, rank_main=_t['rank'], p_top10_main=_t.p_top10)
    for _a, _b in (('score', 'score_c'), ('rank', 'rank_c'), ('p_top10', 'p_top10_c'), ('rank_lo', 'rank_lo_c'), ('rank_hi', 'rank_hi_c'), ('sens_top10_share', 'sens_c'),
                   ('z_level', 'z_level_c'), ('z_uplift', 'z_uplift_c'), ('z_cons', 'z_cons_c'), ('tier', 'tier'), ('n_models_top10', 'n_models_top10')):
        _t[_a] = _C[_b].reindex(_t.index)
    for _k in CON['models']:
        _t['rank|' + _k] = _C['rank|' + _k].reindex(_t.index)
    _t = _t.sort_values('rank'); R['res'][_role]['table'] = _t
    R['res'][_role]['sens'] = pd.DataFrame({nm: (-(w[0] * _t.z_level + w[1] * _t.z_uplift + w[2] * _t.z_cons)).rank() for nm, w in _W.items()})
    _s = CON['summary'][_role]; _r = R['res'][_role]['rel']
    _r['score_main'] = _r['score']; _r['pred_score_main'] = _r['pred_score_to_rawP']
    _r['score'] = _s['rel']; _r['pred_score_to_rawP'] = _s['pred']; _r['chrono_score'] = _s['chrono']
    if _s['yy'] is not None:
        _r['yy_score'] = _s['yy']

WR = R.get('W_ROLE', {'bat': R['W'], 'bowl': R['W']}); LEARNED = R.get('WEIGHTS', 'brief') == 'learned'
wp = lambda role, k: f"{100 * WR[role][k]:.0f}%"
def wsame(k):
    return wp('bat', k) if WR['bat'][k] == WR['bowl'][k] else f"{wp('bat', k)} batters, {wp('bowl', k)} bowlers"
WFORM = ' + '.join(f"{WR['bat'][k]:.2f} x {n}" for k, n in (('level', 'Level'), ('uplift', 'Uplift'), ('cons', 'Consistency')) if WR['bat'][k] > 0)
if WR['bat'] != WR['bowl']:
    WFORM = 'batters ' + WFORM + '; bowlers ' + ' + '.join(f"{WR['bowl'][k]:.2f} x {n}" for k, n in (('level', 'Level'), ('uplift', 'Uplift'), ('cons', 'Consistency')) if WR['bowl'][k] > 0)
bat = R['res']['bat']['table']; bowl = R['res']['bowl']['table']
for _role in ('bat', 'bowl'):      # consensus reliability figures replace the single-model ones wherever the report quotes "the rating"
    _s = CON['summary'][_role]; S[f'{_role}_rel'].update(score=_s['rel'], pred_score_to_rawP=_s['pred'], chrono_score=_s['chrono'], **({'yy_score': _s['yy']} if _s['yy'] is not None else {}))
WPLL ={'Main (blend)': (S['wp1']['logloss'], S['wp2']['logloss']), 'Logistic only': (0.625, 0.387), 'Boosted trees only': (0.635, 0.392)}
if os.path.exists(f"{OUT}/wp_logloss.json"):          # written by the Kaggle notebook from each model's own out-of-fold run
    WPLL.update({k: tuple(v) for k, v in json.load(open(f"{OUT}/wp_logloss.json")).items()})
PINK = RGBColor(0xE6, 0x00, 0x7E); NAVY = RGBColor(0x0B, 0x1F, 0x4B)

doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.0); s.top_margin = s.bottom_margin = Cm(1.8)
st = doc.styles['Normal']; st.font.name = 'Calibri'; st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(6)
for lvl, sz in ((1, 15), (2, 12)):
    h = doc.styles[f'Heading {lvl}']; h.font.name = 'Calibri'; h.font.size = Pt(sz); h.font.bold = True; h.font.color.rgb = NAVY
    h.paragraph_format.space_before = Pt(12 if lvl == 1 else 8); h.paragraph_format.space_after = Pt(4)


def P(text, bold=False, italic=False, size=None, color=None, align=None):
    p = doc.add_paragraph()
    parts = text.split('**')
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part); r.bold = bold or (i % 2 == 1); r.italic = italic
        if size: r.font.size = Pt(size)
        if color: r.font.color.rgb = color
    if align: p.alignment = align
    return p


def B(text):
    p = doc.add_paragraph(style='List Bullet')
    parts = text.split('**')
    for i, part in enumerate(parts):
        if part:
            r = p.add_run(part); r.bold = (i % 2 == 1)
    p.paragraph_format.space_after = Pt(2)
    return p


def shade(cell, hexcol):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hexcol); tcPr.append(shd)


def T(df, widths=None, size=8.5, header_fill='0B1F4B', left=()):
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]; cell.text = ''
        r = cell.paragraphs[0].add_run(str(c)); r.bold = True; r.font.size = Pt(size); r.font.color.rgb = RGBColor(255, 255, 255)
        shade(cell, header_fill)
    for i, row in enumerate(df.itertuples(index=False)):
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ''
            r = cells[j].paragraphs[0].add_run(str(v)); r.font.size = Pt(size)
            if (j > 1 and j not in left) or (j == 0 and len(str(v)) < 4):
                cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if i % 2 == 1:
                shade(cells[j], 'F3F4F8')
    for row in t.rows:
        for c in row.cells:
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def F(name, caption, width=17.0):
    doc.add_picture(f"{FIG}/{name}.png", width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = P(caption, italic=True, size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER); c.paragraph_format.space_after = Pt(8)


f1 = lambda v: f"{v:.1f}"
f0 = lambda v: f"{v:.0f}"
pc = lambda v: f"{100 * v:.0f}%"
sgn = lambda v: f"{v:+.1f}"
U = 1e4   # rate -> win-probability points per 100 balls

assert LEARNED, 'the prose below describes the learned-weights rating'
ordn = lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
rcol = lambda tab, col: (-tab.set_index('name')[col]).rank(method='min').astype(int)
pcw = lambda w: f"{100 * w:.0f}%"
LB, CB = WR['bat']['level'], WR['bat']['cons']; LW, CW = WR['bowl']['level'], WR['bowl']['cons']
nfirm = lambda tab: int((tab.head(10).tier == 'Firm').sum())
WORDS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']
num = lambda n: WORDS[int(n)] if 0 <= int(n) <= 10 else str(int(n))
sup = lambda n, w: f"the {w}" if n == 1 else f"the {ordn(n)} {w}"
andj = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' and ' + xs[-1]

# ============ Title
P("SupeRR Selector Season 2  |  Round 1", bold=True, size=10, color=PINK)
P("Who delivers when the match is on the line?", bold=True, size=20, color=NAVY)
P("A leverage-based pressure rating for T20 batters and bowlers: methodology report", size=12, color=NAVY)

# ============ 1. Summary (one page)
doc.add_heading("1. Summary", 1)
top = pd.DataFrame({'Rank': range(1, 11), 'Top 10 batters': bat.name.head(10).values, 'Top 10 bowlers': bowl.name.head(10).values})
T(top, widths=[1.5, 6.5, 6.5], size=9, left=(2,))
P("Names are spelled as in the supplied data.", italic=True, size=8.5)
_pcs = S['press_by_close']; SZ = 10
P(f"**The question.** Rajasthan Royals asked which T20 batters and bowlers are the most reliable under pressure, judged only on the supplied data. We used {S['matches_used']} IPL and international matches from 2022 to 2026: {S['balls_used']:,} deliveries.", size=SZ)
P(f"**What we mean by pressure.** A ball is under pressure when its outcome can change who wins. From the score, wickets, balls left and target, we estimated each side's chance of winning before every ball, and from that how far the next ball could move it. The quarter of balls with the most at stake are our pressure balls, {S['press_balls']:,} in all. They gather in close finishes: {_pcs['Chase won in final over']:.0f}% of the balls in chases won in the last over, against {_pcs['Chase won with 19+ balls left']:.0f}% in chases won with more than three overs to spare. Balls in finals and playoffs count up to one and a half times.", size=SZ)
P("**How we measured performance.** Each ball is credited with how much it changed the batting side's chance of winning: a boundary in a tight chase earns the batter a lot, and a dot ball at the same moment earns the bowler credit. Half the credit is what actually happened; the other half is a standard mark for that outcome in that kind of situation, so nobody gains from the luck of which ball a six happened to land on. We allowed for the strength of the opponent and for what had just happened, such as a batter arriving after a run of dots, and we pulled short records towards the average so that a brief hot streak cannot top a list. Players needed 150 pressure balls in 15 matches; 97 batters and 97 bowlers qualified.", size=SZ)
P(f"**The rating.** It has two parts: what a player is worth per pressure ball, and how rarely they lose one. Batters are weighted {LB * 100:.0f}% and {CB * 100:.0f}%, bowlers {LW * 100:.0f}% and {CW * 100:.0f}%, because those weights best predicted performance in matches held back from the calculation. We repeated everything with three different models of the chance of winning and averaged the three results.", size=SZ)
P(f"**What we found.** No player's ability to raise their game under pressure can be told apart from luck in this data: the gaps between players' pressure and usual performance are no bigger than random reshuffling produces. The lists therefore rank the players worth most when the pressure is on. Mostly they are excellent players who keep performing in those moments.", size=SZ)
P(f"**How sure we are.** Ratings built from two random halves of the matches agree at {S['bat_rel']['score']:.2f} for batters and {S['bowl_rel']['score']:.2f} for bowlers, on a scale where 0 is pure luck and 1 is perfect agreement. Section 6 shows how secure each place is and which players come next.", size=SZ)

# ============ 2. Data
doc.add_heading("2. Data and preparation", 1).paragraph_format.page_break_before = True
P(f"We used all six supplied files. The pack holds {S['matches_total']} matches: IPL 2022 to 2026, T20 internationals from January 2022 to November 2026, and two A-team Asia Cup events. Of these, {S['matches_with_bbb']} have ball-by-ball records, and we analysed {S['matches_used']} of them ({S['ipl_matches']} IPL, {S['intl_matches']} international; {S['balls_used']:,} deliveries).")
P(f"We left out {S['dls_excluded']} matches revised by Duckworth-Lewis-Stern, because a revised target makes the state of an innings ambiguous, and 30 matches that were abandoned, ended without a result or have incomplete records. We dropped super overs (innings 3 and 4) and kept the regulation play of tied matches.")
P("The extras column marks illegal deliveries as wide, Overstep or 'other'. We checked that 'other' means a no-ball: counting all three as illegal makes 31,757 of 32,338 overs exactly six balls long, and the rest are overs that were never finished.")
P("From the delivery log we rebuilt the state before every ball: legal balls left, runs, wickets lost and, in a chase, runs still needed. The rebuilt totals give the recorded winner in all 794 decided, untied matches.")
P("Scoring rose sharply over the period, so the model also receives a par score: the mean first-innings total for the competition and year. Par rose from 171 in IPL 2022 to 196 in IPL 2026 and ranged from 166 to 180 in internationals. A total of 180 means something different in each.")

# ============ 3. Pressure
doc.add_heading("3. Pressure definition", 1)
doc.add_heading("3.1 Pressure is what the next ball can do to the result", 2)
P("Required rate, wickets in hand, balls left and the closeness of the match all shape pressure, and they interact. Needing 12 an over is routine with 60 balls and eight wickets left; with 9 balls left it decides the match. Instead of setting a threshold for each, we combine them in one quantity, **leverage**: how far the next ball is expected to move the chance of winning.")
P("The indicators we used, and why:", bold=True)
T(pd.DataFrame({'Indicator': ['Balls left', 'Runs needed and required rate (chase)', 'Wickets in hand', 'Runs, run rate and projected total (first innings)', 'Par score for the competition and year', 'Stage of the match (final, playoff)', 'Dots, wickets and boundaries in the last six balls'],
                'How we use it': ['Input to the win-probability model', 'Input to the win-probability model', 'Input to the win-probability model', 'Input to the win-probability model', 'Input to the win-probability model', 'Multiplies the stakes of every ball (3.6)', 'Squeeze state (3.5)'],
                'Why': ['Less time left means each ball matters more', 'A hard but reachable target is where pressure peaks', 'Fewer wickets leave less room to recover from a mistake', 'Shows how far ahead of or behind par the batting side is', 'A total of 180 means different things in 2022 and 2026', 'Bigger matches carry more at stake', 'Pressure is also felt from what has just happened']}), widths=[5.0, 5.0, 7.0], size=8)
doc.add_heading("3.2 Step 1: a win-probability model", 2)
P("A win-probability model estimates, from the match situation alone, how often the batting side goes on to win. It learns this from how matches in the data that passed through similar situations actually ended.")
P("In a chase the model uses balls left, runs needed, wickets in hand and par. It averages two standard statistical models. The first is a logistic regression, which turns a weighted sum of inputs into a probability; its inputs are cricket-shaped (required rate, required rate times log balls left, runs needed against resources left, and wickets times log balls left). The second is gradient-boosted trees, a flexible model built from many small decision rules, constrained so that more balls or wickets never lower the chance of winning and more runs needed never raise it.")
P("In the first innings a logistic regression uses balls left, runs, wickets lost, par, current rate and a projected total. When the innings ends, the chase model takes over with the real target.")
P(f"Every match's probabilities come from a model fitted without that match: we split the matches into five groups and predicted each group with a model fitted on the other four. We scored those predictions two ways. Log-loss penalises confident wrong predictions (lower is better; a coin toss scores 0.693). AUC is the chance that the model rates the eventual winner's position above the loser's (0.5 is guessing, 1 is perfect). Log-loss is {S['wp1']['logloss']:.3f} and AUC {S['wp1']['auc']:.2f} in the first innings, where little is known before a target exists, and {S['wp2']['logloss']:.3f} and {S['wp2']['auc']:.2f} in the chase. Predicted and actual win rates agree across the whole range (Figure 1).")
F('f2_calibration', "Figure 1. Calibration of the win-probability model on matches it was not fitted to ('out-of-fold'). Dot size shows the number of deliveries.", 13.5)
cal = S['more']['calibration']
P("The model stays calibrated within each segment below. 'Gap' is the average distance between predicted and actual win rates across ten probability bands; it is largest for the 28 knockout matches, where the sample is smallest.", size=9.5)
T(pd.DataFrame({'Segment': [('IPL matches' if r['segment'] == 'IPL' else r['segment'] + (' season' if r['segment'].isdigit() else '')) for r in cal], 'Matches': [r['matches'] for r in cal], 'Mean predicted': [f"{r['pred']:.3f}" for r in cal], 'Actual win rate': [f"{r['actual']:.3f}" for r in cal],
                'Log-loss': [f"{r['logloss']:.3f}" for r in cal], 'Gap': [f"{r['ece']:.3f}" for r in cal]}), widths=[4.4, 2.0, 2.8, 2.8, 2.2, 2.0], size=8)
doc.add_heading("3.3 Step 2: the leverage of every delivery", 2)
P("For each ball we ask where win probability would go if the ball were a dot, 1, 2, 3, 4, 6, a wicket or a wide. Leverage is the average size of those eight moves, weighted by how often each outcome happens in that innings and phase (powerplay, middle or death overs). We scale it so that the average delivery has a leverage of 1.0.")
F('f1_leverage_maps', "Figure 2. Mean leverage by match situation; cells with fewer than 25 deliveries are blank. Leverage peaks when the target is hard but reachable and time is short, and falls away once either side has the match won.")
doc.add_heading("3.4 Step 3: the pressure set", 2)
P(f"**A pressure ball is any delivery with leverage of at least {S['thr']:.2f}, the 75th percentile.** With the knockout weighting of 3.6, that gives {S['press_balls']:,} pressure balls, {pc(S['press_balls'] / S['balls_used'])} of all deliveries. Within the set, a more leveraged ball counts for more, up to a cap of three times the average ball, so that one last-ball finish cannot make a career.")
P("Three checks show that the definition captures pressure and not just a phase of play.")
pv = pd.DataFrame(S['press_vs_ord'])
_lab = {'Mean leverage (x avg ball)': 'Average stakes (1 = average ball)', 'Mean |win-prob swing| per ball (pts)': 'Average change in win probability per ball (points)', 'Batting side win prob (mean)': "Batting side's average chance of winning (%)"}
tbl = pd.DataFrame({'Measure': [_lab.get(i, i) for i in pv.index], 'Ordinary balls': [f"{v:,.0f}" if k == 'Balls' else f"{v:.2f}" for k, v in pv['Ordinary'].items()],
                    'Pressure balls': [f"{v:,.0f}" if k == 'Balls' else f"{v:.2f}" for k, v in pv['Pressure'].items()]})
T(tbl, widths=[8, 3.5, 3.5])
c = S['composition']; pcs = S['press_by_close']
swing = 'Mean |win-prob swing| per ball (pts)'
B(f"Pressure balls carry three times the stakes: {pv.loc[swing, 'Pressure']:.1f} points of win probability per ball against {pv.loc[swing, 'Ordinary']:.1f}. Their scoring and wicket rates barely differ from those of ordinary balls, so the set picks out situations, not a phase in which the numbers run naturally higher.")
B(f"The share of pressure balls follows the closeness of the result: {pcs['Chase won in final over']:.0f}% of balls in chases won in the final over and {pcs['Defended by 10 or fewer']:.0f}% in totals defended by ten runs or fewer, against {pcs['Chase won with 19+ balls left']:.0f}% in chases won with 19 or more balls to spare and {pcs['Defended by 31+']:.0f}% in wins by 31 runs or more.")
B(f"Pressure balls come from every phase but one. The first-innings powerplay supplies {pc(c['inn1_PP'])} and the first-innings death {pc(c['inn1_DEATH'])}; the chase supplies {pc(c['inn2_PP'])} in the powerplay, {pc(c['inn2_MID'])} in the middle overs and {pc(c['inn2_DEATH'])} at the death. No middle-overs ball of a first innings reaches the threshold, because a single ball changes least then.")
P("The same ball is a pressure ball for striker and bowler alike, so one definition serves both lists.")
F('f3_example_match', f"Figure 3. Worked example: the most leveraged close finish in the data. {pc(S['example']['press_share'])} of this match's deliveries are pressure balls, almost all of them in the chase.", 15)

doc.add_heading("3.5 A second kind of pressure: squeeze", 2)
sq = S['squeeze']
P("Leverage measures the stakes but cannot see how a side arrived at them: two innings can reach the same score by very different routes. We therefore also track pressure from what has just happened.")
P(f"We call the batting side **squeezed** when at least four of the previous six legal balls were dots or a wicket fell in them ({pc(sq['share_bat'])} of deliveries), and the bowler **under attack** when at least two of the previous six went for boundaries ({pc(sq['share_bowl'])}).")
P(f"Squeeze matters. Comparing each player with themselves, after the opposition adjustment, across {sq['bat']['players']} batters with at least 100 balls of each kind, the same batter is worth **{abs(sq['bat']['effect']):.1f} win-probability points per 100 balls less** when squeezed (standard error {sq['bat']['se']:.1f}), about {abs(sq['bat']['runs']):.2f} runs a ball. Across {sq['bowl']['players']} bowlers, being under attack costs {abs(sq['bowl']['effect']):.1f} points (standard error {sq['bowl']['se']:.1f}).")
F('f10_squeeze', "Figure 4. Squeeze is real. Left: runs on the next ball by the number of dots just bowled, relative to the same innings, over, wickets and pressure state. Right: the effect on the same player, with 95% intervals.", 16)
P(f"**How we use it.** The win-probability model does not know about squeeze, so it would mark a batter who keeps arriving after a collapse against too easy a standard. We therefore compare every ball with the average player in the same squeeze state (4.1). We do not reward or penalise how players respond to squeeze, because that response does not repeat between random halves of the matches (correlation {sq['bat']['rel']:.2f} for batters, {sq['bowl']['rel']:.2f} for bowlers), and the pressure set itself rests on stakes alone (Appendix A.1).")

doc.add_heading("3.6 A third kind of pressure: the importance of the match", 2)
ko = S['knockout']; kt = S['design_tests']['knockout']; mwc = S['mw_counts']
P(f"Leverage measures how much a ball matters to the match, not how much the match matters. The matches file records each game's stage, and {ko['ko_matches']} of the {ko['matches']} analysed matches are knockouts: 19 IPL playoff games, the 2022 and 2025 Asia Cup finals, three tri-series finals and six A-team semi-finals and finals. The pack contains no World Cup.")
P(f"Leverage alone does not single knockouts out: {pc(ko['press_share_ko'])} of their deliveries reach the pressure bar, the same as elsewhere. Yet a final carries more at stake than a league game, so we add a light weight for the stage of the match.")
P(f"We therefore multiply each ball's stakes by the importance of its match: **x1.5** for an IPL or Asia Cup final ({mwc.get('1.5', 0)} matches), **x1.25** for an IPL qualifier or eliminator ({mwc.get('1.25', 0)}), **x1.125** for a tri-series or A-team knockout ({mwc.get('1.125', 0)}), and x1 otherwise. The pressure bar stays at {S['thr']:.2f}, so no other match changes. In knockouts, {S['ko_new_press']:,} deliveries that fell just short of the bar become pressure balls, and every ball counts for more in a player's rates.")
P("Appendix A.2 shows how we chose the size of these weights.")

# ============ 4. Performance
doc.add_heading("4. Performance measurement", 1)
doc.add_heading("4.1 Ball value: win probability added", 2)
P("Each delivery is worth the change in the batting side's chance of winning, from the state before the ball to the state after it. The batter earns that change and the bowler its negative. A dot ball costs a batter dearly when 14 an over is needed and almost nothing at 4 an over. Because every ball is valued by what it did to the match, we need no separate rules for strike rate, economy, dots or boundaries.")
B("Wides count against the bowler only. A run-out of the non-striker does not count against the striker, and run-outs, retirements and obstruction do not count for the bowler.")
B("We split each ball's value into a **run component**, for the runs scored and the ball used, and a **wicket component**, for the extra cost of a dismissal. The two behave very differently from one sample to the next (4.4).")
B("A player's rate on a set of balls is their total value divided by the total leverage of those balls. Dividing by leverage puts pressure and ordinary balls on one scale. We quote rates as **win-probability points added per 100 balls of average leverage**, with 0 as the average player.")
B("We compare every ball with what the average player does on the same kind of ball (pressure or ordinary, in the same squeeze state; 3.5), so no player is judged against a standard that the situation put out of reach.")
doc.add_heading("4.2 The Pressure Exam: every player sits the same paper", 2)
P("Win probability added has one weakness as a measure of a player. Take two batters who each hit a six, a four, three singles and a dot under pressure. Their totals can differ widely, because one six landed on a ball where it was worth 4 points and the other on a ball where it was worth 12. A batter rarely chooses which ball that is, and this timing luck makes up much of the noise in any 'clutch' statistic.")
P(f"The win-probability model can price every possible outcome of every ball, not only the one that happened, so we use it to set a standard paper. We group deliveries into {S['exam_strata']} kinds of situation (pressure or ordinary; innings; powerplay, middle or death overs; and 0-2, 3-5 or 6+ wickets down). Within each kind, every outcome gets a single mark: its average worth per unit of leverage across all balls of that kind. A player's **exam value** for a ball is the mark for the outcome they produced.")
ex = S['exam_examples']
T(pd.DataFrame({'Kind of pressure situation': list(ex), 'Dot': [f"{v['dot']:+.1f}" for v in ex.values()], 'Single': [f"{v['1']:+.1f}" for v in ex.values()], 'Two': [f"{v['2']:+.1f}" for v in ex.values()],
                'Four': [f"{v['4']:+.1f}" for v in ex.values()], 'Six': [f"{v['6']:+.1f}" for v in ex.values()], 'Wicket': [f"{v['wicket']:+.1f}" for v in ex.values()]}), widths=[6.4, 1.6, 1.6, 1.6, 1.6, 1.6, 1.8])
P("Exam marks in three pressure situations, in win-probability points for the batting side per unit of leverage. A wicket costs about twice as much in a chase powerplay as at the death, and a single is a small loss at the death but a small gain in the powerplay.", italic=True, size=8.5)
P("The exam removes timing luck within each kind of situation, along with differences in how hard each player's particular situations were. It keeps the outcomes the player actually produced, weighted by the stakes and compared with other players in the same situations against the same opposition.")
P("The exam treats a six as worth the same anywhere within one kind of situation, which is only roughly true, and the brief asks who delivered in real matches. We therefore value each ball as **half exam mark and half actual win probability added**. Appendix A.1 shows why we chose this blend over the exam alone.")
doc.add_heading("4.3 Each player's usual level", 2)
P("We compute the same blended rate on each player's ordinary balls, with the same opponent adjustment. **Uplift** is the pressure rate minus this usual rate. Because both sets are centred on the league-average player, uplift measures how a player changes under pressure relative to how the average player changes, not relative to zero.")
doc.add_heading("4.4 From raw rates to a rating", 2)
P("A raw pressure rate from 200 balls is mostly noise, so we shrink every measurement towards the pool average, a standard technique called empirical-Bayes shrinkage: a player's estimate is a weighted average of their own record and the average player, and the less evidence the player has, the more weight the average gets. The pull depends on reliability: the genuine spread between players, divided by that spread plus the player's sampling noise. Sampling noise falls as balls accumulate, and we estimate the noise per ball once for the whole pool from match-to-match variation. A player's reliability therefore depends on sample size, not on a noisy individual error estimate.")
rel = pd.DataFrame({'Quantity': ['Overall level (all balls), run component', 'Overall level (all balls), wicket component', 'Uplift (pressure minus usual), whole'],
                    'Batters: weight kept': [pc(S['bat_mean_rel'][k_]) for k_ in ('R_rA_rel', 'K_rA_rel', 'up_tot_rel')],
                    'Bowlers: weight kept': [pc(S['bowl_mean_rel'][k_]) for k_ in ('R_rA_rel', 'K_rA_rel', 'up_tot_rel')]})
T(rel, widths=[7, 4, 4])
P(f"Average share of each player's own measurement that survives shrinkage. Shrinkage keeps {pc(S['bat_mean_rel']['R_rA_rel'])} of a batter's overall run value but only {pc(S['bat_mean_rel']['up_tot_rel'])} of a batter's uplift and {pc(S['bowl_mean_rel']['up_tot_rel'])} of a bowler's, and the clutch test of 7.4 suggests that even this is generous. We shrink uplift as a whole, because the errors in its run and wicket parts are not independent.", italic=True, size=8.5)
P("The rating has two parts.", bold=True)
B(f"**Level ({pcw(LB)} for batters, {pcw(LW)} for bowlers)** is our best estimate of what a player is worth per pressure ball. It starts from the player's shrunk level on all their balls, the most reliable thing we know about them, and adds the share of their shrunk uplift that this overall level does not already contain: Level = overall level + (1 - pressure share) x uplift, where the pressure share is the fraction of the player's balls that were pressure balls (those balls are already inside the overall level). Moving away from a player's own record only as far as the pressure evidence allows is how the rating compares each player with their usual performance.")
B(f"**Consistency ({pcw(CB)} for batters, {pcw(CW)} for bowlers)** measures how rarely a player loses a pressure ball. A ball counts as not lost when its value to the player, after the exam blend and the opposition adjustment, is zero or better. We compare each player's share with the average for the same kind of ball (pressure, squeeze state, innings, phase and wickets down) and shrink it by sample size, with a prior worth about {S['bat_KCB']:.0f} balls for batters and {S['bowl_KCB']:.0f} for bowlers. We count balls rather than matches because a player has about ten times as many pressure balls as matches, so the measure rests on far more evidence (Appendix A.1).")
P(f"To put the two parts on one scale, we express each as a standardised score: 0 is the average eligible player, and +1 is one standard deviation (the typical spread between players' raw, unshrunk values) above it. Then we combine them: **Rating = {WFORM}.** These weights best predicted actual pressure value in matches the rating had not seen (Appendix A.4). Uplift, tested as a third part, predicted nothing beyond what Level already carries, so it has no separate weight; a player who rises under pressure still gains through Level. Since each part is already shrunk by its reliability, a part that is mostly noise has little spread left to sway the ranking. In practice Level drives {pc(S['bat_influence'][0])} of the differences between batters' ratings and {pc(S['bowl_influence'][0])} of the differences between bowlers', and consistency drives the rest.")

doc.add_heading("4.5 One rating, three win-probability models", 2)
cs_ = CON['summary']; mods = CON['models']
P("Every step above depends on the win-probability model, and no model is exactly right. Reasonable models disagree slightly about how much a ball mattered, and for players a few hundredths of a rating point apart, that is enough to swap their places. So we ran the whole method three times, once with each of three win-probability models, and averaged the three ratings.")
_m = 'Main (blend)'; rbm = CON['robust_main']; rbc = S['robust']
keep_b = lambda rb: np.mean([r[3] / max(r[2], 1) for r in rb]) * 100; keep_w = lambda rb: np.mean([r[8] / max(r[7], 1) for r in rb]) * 100
P("A place counts as firm only when all three models put the player in the top 10 (Section 6). Appendix A.3 compares the three models and their reliability.")

# ============ 5. Fairness
doc.add_heading("5. Fair comparisons", 1)
B(f"**Eligibility.** A player needs at least {R['MIN_PBALLS']} pressure balls, faced or bowled, across at least {R['MIN_PMATCH']} matches with pressure balls. That admits {S['bat_n']} batters and {S['bowl_n']} bowlers; the median batter has 238 pressure balls in 28 matches and the median bowler 281 in 27.")
B("**Sample size.** We handle it twice. Eligibility removes the thinnest records, and shrinkage then scales each remaining player's evidence by its reliability, so a 150-ball player needs a much stronger record than a 600-ball player to rank level with them.")
B(f"**Opposition.** We adjust every ball for who was at the other end. We estimate batter and bowler quality jointly from all deliveries by an iterative, shrunk procedure (eight passes, each estimate damped by {R['K_OPP']:.0f} balls of prior), separately for runs and wickets. A second level measures each opponent against the type faced: a batter against pace or spin, a bowler against left- or right-handers, damped by 300 balls. A boundary off an elite bowler therefore earns more than one off a weak bowler, and a bowler who mostly bowls to the hand they prefer gains nothing from it.")
B("**IPL and internationals together.** Both feed one model. The par term absorbs differences in scoring level, and the opponent adjustment absorbs differences in strength, such as an associate attack against an IPL attack. Pressure balls are about as common in each (26.0% of IPL deliveries, 24.2% of international), so neither format dominates.")
B("**Role and situation.** Because each ball is valued from its exact match state, phase, batting position, innings and role are already priced in. A finisher is not penalised for facing harder situations, nor a death bowler for conceding more runs than a middle-overs spinner.")
B("**Single balls.** Leverage weights stop at three times the average ball; 2.8% of deliveries reach the cap.")
B("**Match importance and what has just happened.** Knockout balls carry a light extra weight (3.6), and every ball is compared within its squeeze state (3.5), so batters who arrive after a collapse are not judged against an easier standard than they faced.")
B("**Tested, not assumed.** We compared pressure thresholds (the top 33%, 25%, 20%, 15% and 10% of balls), leverage caps (none, 6, 4, 3 and 2), ways of anchoring Level, ways of estimating reliability and every new idea on held-out matches. Appendix A lists what we adopted and what we rejected.")

# ============ 6. Evidence
doc.add_heading("6. Evidence for the top 10", 1)
P("Rates are win-probability points added per 100 balls of average leverage, adjusted for the opponent, with 0 as the average player. 'Usual' and 'Pressure' are actual value on ordinary and pressure balls; 'Exam' is the exam mark on pressure balls; 'Level' is the blended, shrunk pressure level; 'Uplift' is the shrunk pressure-minus-usual gap, which enters the rating through Level; 'Consistency' is the share of pressure balls not lost, in percentage points above the average for the same kind of ball. 'Rating' is the final score in standardised units: 0 is the average eligible player and 1.0 is one standard deviation above. 'Stays top 10' is the share of 500 resamples in which the player keeps a top-10 place; a resample redraws the matches at random, with replacement, to show how much a rank could move with slightly different data.")
P("Firm and contested places", bold=True)
P("A place is **firm** when the player ranks in the top 10 under all three win-probability models and keeps a top-10 place in at least 60% of resamples. Every other top-10 place is **contested**, and so is any player outside the ten who reaches the top 10 in at least a quarter of resamples. The dropdowns need ten names, so we fill contested places in rating order, but that order means little.", size=9.5)
for _tab, _lab in ((bat, 'Batter'), (bowl, 'Bowler')):
    _tt = _tab[_tab.tier != ''].sort_values('rank')
    T(pd.DataFrame({'#': _tt['rank'].values, _lab: _tt.name.values, 'Tier': _tt.tier.values, 'Rating': _tt.score.map(lambda v: f"{v:.2f}").values,
                    'Top 10 under how many models': [f"{int(v)} of 3" for v in _tt.n_models_top10], 'Stays top 10 (resamples)': _tt.p_top10.map(pc).values,
                    'Top 10 under weightings': [f"{int(round(8 * v))} of 8" for v in _tt.sens_top10_share]}), widths=[0.8, 4.4, 2.0, 1.5, 2.8, 2.6, 2.6], size=8)
P("'Top 10 under weightings' counts how many of eight alternative ways of weighting the rating's parts keep the player in the top 10 (7.2).", size=8.5, italic=True)
CT = json.load(open(f"{OUT}/contest_summary.json")) if os.path.exists(f"{OUT}/contest_summary.json") else None
if CT:
    P("Head to head inside the contested group", bold=True)
    P("For each pair of contested batters, the share of the 500 resamples (averaged over the three models) in which the row player rates above the column player. 50% is a coin toss.", size=9.5)
    _b = CT['bat']; _sn = lambda n: n.split()[-1]
    T(pd.DataFrame({**{'Batter': _b['group']}, **{_sn(n): [('-' if i == j else pc(_b['H'][i][j])) for i in range(len(_b['group']))] for j, n in enumerate(_b['group'])}}), widths=[4.4] + [2.2] * len(_b['group']), size=8)
    _nb = len(_b['group']); _bt = [_b['H'][i][j] for i in range(_nb - 1) for j in range(_nb - 1) if i != j]; _bl = [_b['H'][i][-1] for i in range(_nb - 1)]
    _first = ', '.join(_b['group'][:-2]) + ' and ' + _b['group'][-2]
    P(f"Among batters, {_first} split their head-to-heads {min(_bt) * 100:.0f}-{max(_bt) * 100:.0f}%, a three-way tie, but each beats {_b['group'][-1]} in {min(_bl) * 100:.0f}-{max(_bl) * 100:.0f}% of resamples. No batter outside our ten reaches the top 10 in a quarter of resamples, so these four names are settled; only their order is open.")
    _w = CT['bowl']; _g = _w['group']; _f = _w['firm']; _top = _g[:2]; _r3 = _g[2:5]; _out = _g[5:]
    _h = lambda a, b: _w['H'][_g.index(a)][_g.index(b)]
    _av = dict(zip(_g, _w['ahead'])); _mn = {a: min(_h(a, x) for x in _g if x != a) for a in _top}
    _rt3 = [_h(a_, b_) for a_ in _r3 for b_ in _r3 if a_ != b_]
    _beat = min(_h(a_, b_) for a_ in _r3 for b_ in _out) if _out else None
    P(f"Among bowlers, {_top[0]} and {_top[1]} (places {_f + 1} and {_f + 2}) win {pc(_av[_top[0]])} and {pc(_av[_top[1]])} of their head-to-heads on average, so those places are reasonably secure. The last three places are close: {_r3[0]}, {_r3[1]} and {_r3[2]} beat one another in {min(_rt3) * 100:.0f}-{max(_rt3) * 100:.0f}% of resamples, so their order means little" + (f", though each beats every bowler outside the ten in at least {_beat * 100:.0f}% of resamples." if _beat and _beat >= 0.5 else "."))


def commentary(tab, role):
    k = tab.set_index('name'); top = list(tab.name.head(10)); n = len(tab)
    lv, bs, up, cs = rcol(tab, 'level_eb'), rcol(tab, 'base_eb'), rcol(tab, 'uplift_eb'), rcol(tab, 'cball')
    names = lambda ps: ', '.join(ps[:-1]) + ' and ' + ps[-1] if len(ps) > 1 else ps[0]
    p1 = top[0]; app = f"{int(k.loc[p1, 'pos'])} of {int(k.loc[p1, 'app'])}"
    did = (f"struck at {k.loc[p1, 'SR_P']:.0f} and averaged {k.loc[p1, 'Avg_P']:.0f}" if role == 'bat' else f"conceded {k.loc[p1, 'Econ_P']:.2f} an over and took a wicket every {k.loc[p1, 'SR_P']:.0f} balls")
    B(f"**{p1}** ranks first. On {k.loc[p1, 'n_P']:.0f} pressure balls {p1.split()[-1]} {did}, adding value on pressure balls in {app} of the matches in which {p1.split()[-1]} {'faced' if role == 'bat' else 'bowled'} any. The actual pressure value is {k.loc[p1, 'T_rP'] * U:+.0f} and the exam mark {k.loc[p1, 'exam_rP'] * U:+.0f}; shrinkage turns these into a level of {k.loc[p1, 'level_eb'] * U:+.0f}, {sup(lv[p1], 'highest')} of {n}. {p1.split()[-1]} ranks {ordn(cs[p1])} on consistency and keeps a top-10 place in {pc(k.loc[p1, 'p_top10'])} of resamples.")
    lead = [p for p in top[1:] if lv[p] <= 6]
    if lead:
        rng_ = (f"pressure strike rates from {min(k.loc[lead, 'SR_P']):.0f} to {max(k.loc[lead, 'SR_P']):.0f}" if role == 'bat' else f"pressure economy from {min(k.loc[lead, 'Econ_P']):.2f} to {max(k.loc[lead, 'Econ_P']):.2f}")
        B(f"**{names(lead)}** all rank in the top six of the pool on pressure level, with {rng_}. They earn their places by what they are worth in these situations, whether or not they beat their own norm.")
    upl = [p for p in top if up[p] <= 6 and p != p1]
    if upl:
        det = '; '.join(f"{p.split()[-1]} from {k.loc[p, 'T_rN'] * U:+.0f} on ordinary balls to {k.loc[p, 'T_rP'] * U:+.0f} under pressure" for p in upl)
        B(f"**{names(upl)}** do get better when it matters: their shrunk uplifts rank {andj([ordn(up[p]) for p in upl])} in the pool ({det}).")
    con = [p for p in top if cs[p] <= 5 and p != p1]
    if con:
        det = '; '.join(f"{p.split()[-1]} {k.loc[p, 'cball'] * 100:+.1f} points ({ordn(cs[p])})" for p in con)
        B(f"**{names(con)}** lose the fewest pressure balls, measured against the average for the same balls: {det}.")
    neg = [p for p in top if k.loc[p, 'uplift_eb'] < 0 and p != 'Jasprit Bumrah']
    for p in neg:
        B(f"**{p}** makes the list despite doing worse under pressure than usual. {p.split()[-1]} has {sup(bs[p], 'highest')} overall level in the pool but ranks {ordn(up[p])} of {n} on uplift, and is here for value in pressure situations, not for rising to them.")
    weak = sorted(top, key=lambda p: k.loc[p, 'p_top10'])[:2]; nxt = list(tab.name.iloc[10:13])
    nx = ', '.join(f"{p} ({pc(k.loc[p, 'p_top10'])})" for p in nxt)
    B(f"**{weak[0]}** and **{weak[1]}** hold the least secure places, keeping a top-10 place in {pc(k.loc[weak[0], 'p_top10'])} and {pc(k.loc[weak[1], 'p_top10'])} of resamples. Next in line are {nx}; 10th and 11th are {tab.score.iloc[9] - tab.score.iloc[10]:.2f} rating points apart.")
    return k, lv, bs, up, cs


TCOLS = lambda t: {'Pressure balls': t.n_P.map(f0).values, 'Usual': (t.T_rN * U).map(sgn).values, 'Pressure': (t.T_rP * U).map(sgn).values, 'Exam': (t.exam_rP * U).map(sgn).values,
                   'Level': (t.level_eb * U).map(sgn).values, 'Uplift': (t.uplift_eb * U).map(sgn).values, 'Consistency': (t.cball * 100).map(sgn).values,
                   'Rating': t.score.map(lambda v: f"{v:.2f}").values, 'Stays top 10': t.p_top10.map(pc).values}
doc.add_heading("6.1 Batters", 2)
t = bat.head(10)
T(pd.DataFrame({'#': t['rank'].values, 'Batter': t.name.values, **TCOLS(t)}), widths=[0.7, 3.5, 1.4, 1.2, 1.3, 1.2, 1.3, 1.3, 1.7, 1.2, 1.4], size=8)
F('f4_top10_batters', "Figure 5. Top-10 batters. Left: actual value on ordinary balls (navy) and pressure balls (pink), the exam mark (triangle) and the final shrunk level (diamond). Middle: consistency. Right: share of resamples in which the player stays in the top 10.")
P("The same players in conventional numbers, pressure against ordinary balls:", bold=True)
T(pd.DataFrame({'Batter': t.name.values, 'Pressure runs': t.runs_P.map(f0).values, 'Strike rate, pressure': t.SR_P.map(f1).values, 'Strike rate, usual': t.SR_N.map(f1).values, 'Average, pressure': t.Avg_P.map(f1).values, 'Average, usual': t.Avg_N.map(f1).values,
                'Boundary %, pressure': t['Bdry%_P'].map(f1).values, 'Boundary %, usual': t['Bdry%_N'].map(f1).values, 'Pressure balls in IPL': t.ipl_share_P.map(pc).values}), widths=[3.6, 1.4, 1.6, 1.5, 1.7, 1.5, 1.7, 1.9, 1.4], size=8)
k, lv, bs, up, cs = commentary(bat, 'bat')
if 'Pathum Nissanka' in k.index and k.loc['Pathum Nissanka', 'rank'] > 10:
    pn = k.loc['Pathum Nissanka']
    B(f"**Pathum Nissanka**, {ordn(int(pn['rank']))}, has {sup(up['Pathum Nissanka'], 'largest')} uplift in the pool, from {pn.T_rN * U:+.0f} on ordinary balls to {pn.T_rP * U:+.0f} under pressure over {pn.n_P:.0f} pressure balls. A modest overall level ({ordn(bs['Pathum Nissanka'])} of {len(bat)}) keeps Nissanka out of the ten.")
doc.add_heading("6.2 Bowlers", 2)
t = bowl.head(10)
T(pd.DataFrame({'#': t['rank'].values, 'Bowler': t.name.values, **TCOLS(t)}), widths=[0.7, 3.5, 1.4, 1.2, 1.3, 1.2, 1.3, 1.3, 1.7, 1.2, 1.4], size=8)
F('f5_top10_bowlers', "Figure 6. Top-10 bowlers, in the same layout as Figure 5.")
P("The same players in conventional numbers, pressure against ordinary balls:", bold=True)
T(pd.DataFrame({'Bowler': t.name.values, 'Pressure wickets': t.wkts_P.map(f0).values, 'Economy, pressure': t.Econ_P.map(lambda v: f"{v:.2f}").values, 'Economy, usual': t.Econ_N.map(lambda v: f"{v:.2f}").values,
                'Balls per wicket, pressure': t.SR_P.map(f1).values, 'Balls per wicket, usual': t.SR_N.map(f1).values, 'Dot %, pressure': t['Dot%_P'].map(f1).values, 'Dot %, usual': t['Dot%_N'].map(f1).values, 'Pressure balls in IPL': t.ipl_share_P.map(pc).values}),
  widths=[3.6, 1.4, 1.7, 1.5, 1.6, 1.8, 1.4, 1.5, 1.4], size=8)
k, lv, bs, up, cs = commentary(bowl, 'bowl')
if bowl.score.iloc[0] - bowl.score.iloc[3] < 0.15:
    B(f"The top four ({', '.join(bowl.name.iloc[:4])}) sit within {bowl.score.iloc[0] - bowl.score.iloc[3]:.2f} rating points of one another, so their order among themselves means little.")
jb = k.loc['Jasprit Bumrah']
_noadj = (f" The opposition adjustment matters here, because Bumrah bowls to the best batters at the hardest times: without it, the main model would rank Bumrah {ordn(int(jb.rank_noadj))}." if jb.rank_noadj - jb.rank_main >= 3 else "")
B(f"**Jasprit Bumrah** ranks {ordn(int(jb['rank']))}. Bumrah's overall level is {ordn(bs['Jasprit Bumrah'])} of {len(bowl)} and consistency {ordn(cs['Jasprit Bumrah'])} ({jb.cball * 100:+.1f} points of pressure balls not lost). In this sample, though, Bumrah's pressure balls did less than ordinary ones (a wicket every {jb.SR_P:.0f} balls against {jb.SR_N:.0f}; economy {jb.Econ_P:.2f} against {jb.Econ_N:.2f}), which puts the uplift at {ordn(up['Jasprit Bumrah'])} of {len(bowl)}. The exam marks the pressure bowling at {jb.exam_rP * U:+.0f} against an actual {jb.T_rP * U:+.0f}, so timing explains part of that gap." + _noadj + f" Bumrah keeps a top-10 place in {pc(jb.p_top10)} of resamples and under {int(round(8 * jb.sens_top10_share))} of the 8 weightings.")
doc.add_heading("6.3 Where each player's pressure value comes from", 2)
P("Pressure value per 100 balls by phase, innings and type of opponent, with the number of pressure balls in brackets. Splits of fewer than 40 balls are mostly noise.", size=9.5)
for role_, cols_ in (('bat', ['Powerplay', 'Middle', 'Death', 'Setting', 'Chasing', 'v pace', 'v spin']), ('bowl', ['Powerplay', 'Middle', 'Death', 'Bowling first', 'Defending', 'v left-handers', 'v right-handers'])):
    _rk = dict(zip((bat if role_ == 'bat' else bowl).name, (bat if role_ == 'bat' else bowl)['rank']))   # final (consensus) order, not the main model's
    pr = sorted(S['more']['profiles'][role_], key=lambda r: _rk.get(r['name'], 99))
    T(pd.DataFrame({**{'Batter' if role_ == 'bat' else 'Bowler': [f"{int(_rk[r['name']])}. {r['name']}" for r in pr]}, **{c_: [(f"{r[c_][0]:+.0f} ({r[c_][1]})" if c_ in r and r[c_][0] is not None else '-') for r in pr] for c_ in cols_},
                    'Strongest': [r['best'] or '-' for r in pr]}), widths=[3.9, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 2.0], size=7.5)
_grp = {}
for _r in S['more']['profiles']['bat']:
    _ph = max(('Powerplay', 'Middle', 'Death'), key=lambda k_: (_r[k_][1] if k_ in _r else 0))
    _grp.setdefault(_ph, []).append(_r['name'] if _r['name'] == 'Suryakumar Yadav' else _r['name'].split()[-1])
P("The lists are not one type of player. Judged by where most of their pressure balls come, the batters cover the powerplay (" + ', '.join(_grp.get('Powerplay', [])) + "), the middle overs (" + ', '.join(_grp.get('Middle', [])) + ") and the death (" + ', '.join(_grp.get('Death', [])) + "). Almost all pressure bowling in this data comes while defending a total, and the bowlers cover the powerplay, the middle overs and the death.", italic=True, size=8.5)
F('f7_pressure_vs_usual', "Figure 7. Every eligible player's actual value on ordinary balls against pressure balls. Pink marks the final top 10; points above the line did better under pressure.")

# ============ 7. Reliability
doc.add_heading("7. Reliability and limitations", 1)
doc.add_heading("7.1 How stable the lists are", 2)
br, wr = S['bat_rel'], S['bowl_rel']
sb = lambda r: 2 * r / (1 + r)
B(f"**Resampling.** We resampled whole matches 500 times and recomputed every rating. Figure 8 shows each player's 10th-to-90th-percentile rank. {num((bat.p_top10.head(10) >= 0.55).sum())} of our batters and {num((bowl.p_top10.head(10) >= 0.55).sum()).lower()} of our bowlers keep a top-10 place in more than 55% of resamples, and the last places sit close to the players just outside.")
F('f6_rank_intervals', "Figure 8. Final rank (dot) and the resampled 10th-to-90th-percentile rank range (bar) for the top 20 in each list. The dashed line is the top-10 cut.")
IV = json.load(open(f"{OUT}/intervals_summary.json"))
_bi = lambda role: (min(r['beats_11th'] for r in IV[role][:10]), max(r['beats_11th'] for r in IV[role][:10]))
_ah = lambda role: sum(r['ahead'] >= 0.6 for r in IV[role][:10])
B(f"**Rating intervals.** Figure 9 and the tables below give each player's rating with its 90% interval over the same 500 resamples, the gap to the next player, and how often each player stays ahead of the next one and of the player ranked 11th. Neighbours' intervals overlap heavily, and only {num(_ah('bat')).lower()} {'batter' if _ah('bat') == 1 else 'batters'} and {num(_ah('bowl')).lower()} {'bowler' if _ah('bowl') == 1 else 'bowlers'} in the top 10 stay ahead of the next player in 60% or more of resamples, so the order within the ten is loose. The line between the ten and the rest is firmer: each of our batters rates above the 11th-placed batter in {pc(_bi('bat')[0])} to {pc(_bi('bat')[1])} of resamples, and each of our bowlers above the 11th-placed bowler in {pc(_bi('bowl')[0])} to {pc(_bi('bowl')[1])}.")
F('f12_rating_intervals', "Figure 9. Final rating (dot) and its 90% interval over 500 match resamples, averaged over the three win-probability models, for the top 15 in each list. Pink marks the top 10; the dashed line is the top-10 cut.", 16)
for _role, _lab in (('bat', 'Batter'), ('bowl', 'Bowler')):
    _r = IV[_role][:10]
    T(pd.DataFrame({'#': [r['rank'] for r in _r], _lab: [r['name'] for r in _r], 'Rating (90% interval)': [f"{r['rating']:.2f} ({r['lo']:.2f} to {r['hi']:.2f})" for r in _r],
                    'Gap to next (90% interval)': [f"{r['gap']:.2f} ({r['gap_lo']:+.2f} to {r['gap_hi']:+.2f})" for r in _r], 'Ahead of next': [pc(r['ahead']) for r in _r],
                    'Above the 11th': [pc(r['beats_11th']) for r in _r]}), widths=[0.8, 4.2, 3.8, 3.8, 2.0, 2.2], size=8)
B(f"**Split halves.** We also split the matches at random into two halves 100 times and reran the whole method on each half. The two halves' ratings correlate (0 means no relation, 1 means identical rankings) {br['score']:.2f} for batters and {wr['score']:.2f} for bowlers, which implies a full-sample reliability of about {sb(br['score']):.2f} and {sb(wr['score']):.2f}. Figure 10 breaks this down by ingredient.")
F('f8_reliability', "Figure 10. Split-half correlation of each part of the rating. Run value repeats best; raw uplift barely repeats at all.", 16)
B(f"**Early against late.** Ratings from the first half of the period correlate {wr['chrono_score']:.2f} with ratings from the second half for bowlers ({int(wr['chrono_n'])} players) and {br['chrono_score']:.2f} for batters ({int(br['chrono_n'])} players). Bowlers' pressure performance carries forward in time better than batters'.")
B(f"**Season to season.** Run on each season alone, the rating correlates {br['yy_score']:.2f} from one season to the next for batters and {wr['yy_score']:.2f} for bowlers, among the roughly {br['yy_n']:.0f} batters and {wr['yy_n']:.0f} bowlers per pair of seasons with 40 or more pressure balls in both. Level correlates {br['yy_level_eb']:.2f} and {wr['yy_level_eb']:.2f}; raw uplift correlates {br['yy_upB']:.2f} and {wr['yy_upB']:.2f}, which is zero for practical purposes, as baseball finds for its clutch measures.")
rb = S['robust']
B("**Different data.** We reran the final method on eleven alternative datasets. 'Still top 10' counts how many of our ten names stay in the top 10, out of those still eligible after the change, since removing matches can drop a player below 150 pressure balls.")
T(pd.DataFrame({'Rerun': [r[0] for r in rb], 'Batters still top 10': [f"{r[3]} of {r[2]}" for r in rb], 'Batters: rating corr.': [f"{r[4]:.2f}" for r in rb],
                'Bowlers still top 10': [f"{r[8]} of {r[7]}" for r in rb], 'Bowlers: rating corr.': [f"{r[9]:.2f}" for r in rb]}), widths=[5.2, 3.0, 3.0, 3.0, 3.0], size=8)
P(f"Across the eleven reruns, {keep_b(rb):.0f}% of our still-eligible batters and {keep_w(rb):.0f}% of our bowlers keep a top-10 place on average; the main win-probability model alone keeps {keep_b(rbm):.0f}% and {keep_w(rbm):.0f}%. Single-format reruns test hardest, because they discard about half of each player's evidence.", italic=True, size=8.5)
ms = S['missed']
_mb = ', '.join(f"{n_} ({ordn(r_)}, {b_} pressure balls)" for n_, r_, b_ in ms['bat']) or 'none'; _mw = ', '.join(f"{n_} ({ordn(r_)}, {b_} pressure balls)" for n_, r_, b_ in ms['bowl'])
B(f"**Below the bar.** Lowering the eligibility bar to 100 pressure balls would bring these batters into the top 20: {_mb}. " + (f"Among bowlers it would add {_mw}. " if _mw else "No bowler would join. ") + "Their rank ranges are too wide to place them in a top 10, but they are the names to watch.")

doc.add_heading("7.2 Predicting matches the rating has not seen", 2)
P("This is the strictest test the data allows. In each of 100 random splits we computed every measure on one half of the matches and compared it with the win probability each player actually added on pressure balls in the other half, adjusted for the opponent. Conventional statistics sat the same test.")
F('f9_out_of_sample', "Figure 11. Correlation between each measure in one half of the matches and actual pressure value in the other half.", 16)
B(f"**Shrinkage and anchoring help.** A player's own raw pressure record predicts the unseen half at {br['pred_rawP_to_rawP']:.2f} for batters and {wr['pred_rawP_to_rawP']:.2f} for bowlers. Level, built from the same half, reaches {br['pred_level_to_rawP']:.2f} and {wr['pred_level_to_rawP']:.2f}, and the final rating {br['pred_score_to_rawP']:.2f} and {wr['pred_score_to_rawP']:.2f}.")
B(f"**For bowlers, the rating beats economy rate.** Pressure economy scores {wr['pred_convP_to_rawP']:.2f} and overall economy {wr['pred_convA_to_rawP']:.2f}, against {wr['pred_score_to_rawP']:.2f} for the rating. Consistency supplies most of that gain.")
B(f"**For batters, plain strike rate predicts as well as the rating.** Overall strike rate scores {br['pred_convA_to_rawP']:.2f} and pressure strike rate {br['pred_convP_to_rawP']:.2f}, against {br['pred_score_to_rawP']:.2f}. With 97 players these differences lie within sampling error, and they point one way: batting under pressure is mostly general scoring ability carrying over. The rating adds a fairer account, with dismissals, match situation and opposition priced in.")
def sens_txt(role, tab):
    lists = [set(v) for v in S[f'{role}_sens'].values()]
    base = list(tab.name.head(10)); core = set.intersection(*lists)
    cnt = {n: sum(n in l for l in lists) for n in set.union(*lists)}
    wob = [f"{n} ({cnt[n]} of 8)" for n in base if n not in core]
    outs = [f"{n} ({cnt[n]} of 8)" for n in sorted(cnt, key=lambda n: -cnt[n]) if n not in base]
    vb = lambda xs, s, p_: s if len(xs) == 1 else p_
    return len(core), (andj(wob) + vb(wob, ' drops out', ' drop out')) if wob else 'nobody drops out', (andj(outs) + vb(outs, ' comes in', ' come in')) if outs else 'nobody comes in'


cb, wb_, ob = sens_txt('bat', bat); cw, ww_, ow = sens_txt('bowl', bowl)
B(f"**Other weightings.** We re-ranked under eight weightings of level, uplift and consistency, from all level to equal thirds. {num(cb)} of our 10 batters and {num(cw).lower()} of our 10 bowlers make the top 10 under every one. Among batters, {wb_} under some weightings, and {ob}; among bowlers, {ww_}, and {ow}. The brackets give the number of weightings that keep each player in the top 10.")
nb = set(bat.sort_values('rank_main').name.head(10)) - set(S['bat_top10_noadj']); nw = set(bowl.sort_values('rank_main').name.head(10)) - set(S['bowl_top10_noadj'])
B("**Opponent adjustment.** " + ("Removing it from the main model changes none of the twenty names; it reorders players without deciding who is in." if not nb and not nw else f"Removing it from the main model changes {len(nb)} of 10 batters ({', '.join(sorted(nb)) or 'none'}) and {len(nw)} of 10 bowlers ({', '.join(sorted(nw)) or 'none'})."))

doc.add_heading("7.3 A held-out season and a held-out competition", 2)
EX = json.load(open(f"{OUT}/external_summary.json"))
P(f"Split halves mix every season and competition into both halves. A harder test leaves one out entirely. We rebuilt the whole rating without 2026 and scored it against what players did in 2026, and rebuilt it on IPL matches alone (or internationals alone) and scored it on the other competition. The yardstick is the one in 7.2, counted for players with at least {EX['MINB']} pressure balls in the held-out data; each figure is a correlation with its 90% interval.")
_cv = lambda role: 'Strike rate' if role == 'bat' else 'Economy (lower is better)'
_f = lambda v: f"{v['r']:.2f} ({v['lo']:+.2f} to {v['hi']:+.2f})"
_tests = (('season_2026', 'Built on 2022-2025, tested on 2026'), ('ipl_to_intl', 'Built on IPL, tested on internationals'), ('intl_to_ipl', 'Built on internationals, tested on IPL'))
T(pd.DataFrame({'Test': [l for _, l in _tests],
                'Batters scored': [EX[k]['bat']['n_players'] for k, _ in _tests], 'Batters: our rating': [_f(EX[k]['bat']['rows']['Our rating']) for k, _ in _tests], 'Batters: strike rate': [_f(EX[k]['bat']['rows']['Strike rate']) for k, _ in _tests],
                'Bowlers scored': [EX[k]['bowl']['n_players'] for k, _ in _tests], 'Bowlers: our rating': [_f(EX[k]['bowl']['rows']['Our rating']) for k, _ in _tests], 'Bowlers: economy': [_f(EX[k]['bowl']['rows'][_cv('bowl')]) for k, _ in _tests]}),
  widths=[4.2, 1.5, 2.6, 2.6, 1.5, 2.6, 2.6], size=7.5)
_s = EX['season_2026']
P(f"Across competitions the rating carries over well: built on one, it predicts the other at {min(EX[k][r]['rows']['Our rating']['r'] for k in ('ipl_to_intl', 'intl_to_ipl') for r in ('bat', 'bowl')):.2f} to {max(EX[k][r]['rows']['Our rating']['r'] for k in ('ipl_to_intl', 'intl_to_ipl') for r in ('bat', 'bowl')):.2f}, about as well as strike rate or economy. "
  f"For 2026, the bowling rating predicts at {_s['bowl']['rows']['Our rating']['r']:.2f}, and {_s['bowl']['top10_above_avg']} of the {_s['bowl']['top10_scored']} top-10 bowlers who played enough in 2026 beat the average bowler there ({_s['bowl']['top10_mean']:+.0f} against {_s['bowl']['rest_mean']:+.0f} points per 100 balls); plain economy did slightly better that season ({_s['bowl']['rows'][_cv('bowl')]['r']:.2f}). "
  f"For 2026 batters the test cannot separate anything from noise: {_s['bat']['n_players']} batters with 30 to 100 pressure balls each give a correlation of {_s['bat']['rows']['Our rating']['r']:.2f} for the rating and {_s['bat']['rows']['Strike rate']['r']:.2f} for strike rate, both with intervals that include zero. Our top-10 batters still averaged {_s['bat']['top10_mean']:+.0f} points per 100 pressure balls in 2026 against {_s['bat']['rest_mean']:+.0f} for the rest. One season is too short a window to validate batting under pressure, which is why the rating uses all five.")

doc.add_heading("7.4 Is there such a thing as clutch?", 2)
P(f"We tested whether any player truly raises their game under pressure by shuffling which of each player's balls count as pressure balls, 400 times. If some players rose under pressure, the real gaps between players' pressure and usual performance would be wider than the shuffled ones. They are not (p = {S['checks']['bat_perm']['p_spread']:.2f} for batters and {S['checks']['bowl_perm']['p_spread']:.2f} for bowlers; a p-value is the chance of a result this large from luck alone, and only values below 0.05 suggest a real effect). The same test found planted clutch differences most of the time, so any real clutch skill is smaller than the differences in overall quality between players. The lists therefore rank the players worth most in pressure situations, not the players who rise to them, and uplift carries no separate weight. Appendix A.6 gives the details.")

doc.add_heading("7.5 Where the method is less certain", 2)
B(f"**Noise.** With 150 to 800 pressure balls per player, and a wicket worth many runs in win-probability terms, a full-sample reliability of about {sb(br['score']):.2f} for batters and {sb(wr['score']):.2f} for bowlers is what the data supports. The lists show the players who delivered most reliably under pressure in 2022 to 2026, once luck is discounted as far as the sample allows; they are not a forecast.")
B("**Uplift.** It barely repeats between halves of the matches in either list, which is why it carries no separate weight; it still enters through Level.")
B(f"**The exam.** It prices each outcome at its average worth within one of {S['exam_strata']} kinds of situation and ignores finer differences in match state, which is why it gets half the weight, not all.")
B("**Match importance.** The three knockout weights reflect our judgement of how big each kind of match is. The data shows how much weight is affordable, not which final matters more, and the pack has no World Cup.")
B("**Squeeze thresholds.** Four dots in six, a wicket in six and two boundaries in six are simple rules, fixed before we looked at any player results and never tuned.")
B("**The win-probability model.** It knows the match state and scoring level but not team strength, venue, dew, pitch or which batters are still to come. It is well calibrated on average, but individual final-over states rest on few matches. A future version could add team strength (for example each side's rating from earlier matches), venue and the batters still to come as model inputs; the pack records venue and batting order, so those two are within reach.")
B("**Credit.** Striker and bowler take the whole value of each ball. Fielding, the non-striker's running and captaincy are not separated out, and byes and leg-byes, though small, count for the batter and against the bowler.")
B("**Opposition.** The adjustment corrects for each opponent's quality and broad type (pace or spin, left or right hand), but not for finer match-ups such as left-arm spin to a left-hander, nor for field settings.")
B("**Excluded matches.** Leaving out rain-revised matches removes some tense finishes.")
B("**Eligibility.** We do not rate players below 150 pressure balls, however good their record. A lower bar would add names whose rank ranges are too wide to defend.")
B("**Process against outcome.** The rating judges outcomes. Process measures, such as false-shot and control percentages, ball-tracking expected runs and wickets, and how precisely a bowler hit the intended length, would probably settle faster than outcomes and separate skill from luck more cleanly. The pack records only the type of shot played and the type of delivery bowled, with no edges, misses, ball-tracking or fielding chances; we tested shot and delivery types as predictors and they did not help (Appendix A.5).")

# ============ 8. Reproduce
doc.add_heading("8. Reproducing the results", 1)
P("**Code: github.com/Simhaatt/rr.** The repository holds every script, the exact library versions (Python 3.14 with pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2 and python-docx 1.2.0, listed in requirements.txt) and a single command that runs the whole method. Put the six supplied CSVs in data_pack/Data Pack/, then run:")
P("pip install -r requirements.txt", size=9.5).runs[0].font.name = 'Consolas'
P("python run_all.py", size=9.5).runs[0].font.name = 'Consolas'
P("The full run takes about an hour on a four-core computer with 8 GB of memory. Random seeds are fixed, so it reproduces every number in this report. The eight steps below describe what the scripts do, so the method can also be rebuilt from this report alone.")
T(pd.DataFrame({'Step': ['1. Prepare', '2. Win probability', '3. Leverage and pressure', '4. Ball value', '5. Fair comparison', '6. Shrink and rate', '7. Three models', '8. Check'],
                'What to do': ['Load the six files. Mark wides, oversteps and "other" extras as illegal. Rebuild balls left, runs, wickets and runs needed before every ball. Drop rain-revised, abandoned and incomplete matches and super overs. Compute par for each competition and year (3.2, Section 2).',
                               'Fit the chase and first-innings models in five groups of matches, predicting each group with a model fitted on the other four (3.2).',
                               'For every ball, compute the win probability after each of eight outcomes and average the size of the moves, weighted by outcome frequency; scale so the average ball is 1. Multiply by match importance; the top quarter are pressure balls. Flag squeeze from the previous six legal balls (3.3-3.6).',
                               'Credit each ball\'s change in win probability to the striker and its negative to the bowler. Give each outcome its Pressure Exam mark within its group of situations, and value the ball as half exam mark, half actual change (4.1-4.2).',
                               'Estimate opponent quality and quality against each type, and adjust every ball; compare each ball with the average for the same kind of ball and squeeze state (Section 5).',
                               'Compute each player\'s rates on pressure and ordinary balls, shrink by reliability, form Level and Consistency, standardise them and combine with the weights below (4.3-4.4).',
                               'Repeat steps 2 to 6 with each of the three win-probability models and average the ratings; mark firm and contested places from 500 match resamples (4.5, Section 6).',
                               'Rerun on random halves of the matches, single seasons and the eleven altered datasets (Section 7).']}), widths=[3.6, 13.4])
T(pd.DataFrame({'Parameter': ['Ball value', 'Match importance', 'Squeeze', 'Pressure threshold', 'Leverage cap', 'Eligibility', 'Opponent-quality prior', 'Consistency prior', 'Rating weights', 'Win-probability models', 'Firm place', 'Resampling and splits'],
                'Value': ['0.5 x Pressure Exam mark + 0.5 x actual win probability added; exam groups = pressure or not x innings x phase x wickets down', 'IPL or Asia Cup final x1.5; IPL qualifier or eliminator x1.25; tri-series or A-team knockout x1.125; all other matches x1', 'Batting side: 4+ dots or a wicket in the previous 6 legal balls. Bowler: 2+ boundaries in the previous 6', f"Leverage x match importance >= {S['thr']:.3f} (the 75th percentile of in-match leverage)", '3.0 x average ball', f"{R['MIN_PBALLS']}+ pressure balls and {R['MIN_PMATCH']}+ matches with pressure balls", f"Worth {R['K_OPP']:.0f} balls of average leverage, 8 passes; by type of opponent, 300 balls",
                          f"Share of pressure balls not lost, against the same kind of ball; prior of about {S['bat_KCB']:.0f} (batters) and {S['bowl_KCB']:.0f} (bowlers) balls, estimated from the data", WFORM + '; fitted on held-out matches; each part scaled by the spread of its unshrunk version', 'Main blend, logistic only, boosted trees only; the rating is their average', 'Top 10 under all three models and in at least 60% of resamples', '500 match resamples; 100 random splits']}), widths=[5, 12])

# ============ Appendix
# ============ Appendix A: design tests
doc.add_heading("Appendix A. How we tested the design choices", 1)
doc.add_heading("A.1 The ingredients of the rating", 2)
dt = S['design_tests']['rows']
P("Every candidate faced the same test: 40 random splits of the matches into halves, with the whole method rebuilt on each half. 'Repeatable' is how well the rating from one half agrees with the other; 'predicts' is how well Level from one half matches actual pressure value in the other, always measured on the same yardstick.")
T(pd.DataFrame({'Design': [r[0] for r in dt], 'Batters: repeatable': [f"{r[1]:.2f}" for r in dt], 'Batters: predicts': [f"{r[2]:.2f}" for r in dt], 'Bowlers: repeatable': [f"{r[4]:.2f}" for r in dt], 'Bowlers: predicts': [f"{r[5]:.2f}" for r in dt],
                'Verdict': [{'baseline': 'Starting point', 'partly': 'Better for batters only', 'yes': 'Better', 'adopted': 'Adopted', 'mixed': 'Helps bowlers only', 'no gain': 'No gain: not used', 'rejected': 'Rejected'}[r[7]] for r in dt]}), widths=[6.2, 2.0, 2.0, 2.0, 2.0, 3.0], size=8)
F('f11_design_tests', "Figure 12. The first round of tests as a chart. Pink and navy mark the adopted design; dashed lines mark the previous version.", 16)
up_ = S['design_tests']['uplift']
P(f"The first round adopted the exam blend with the squeeze adjustment, which beat the previous version on both measures for both lists by the widest margin. Two ideas failed. A response curve, fitting each player a line for how their value changes as leverage rises, doubled the repeatability of batters' uplift ({up_['binary'][0]:.2f} to {up_['slope'][0]:.2f}) but lowered bowlers' ({up_['binary'][1]:.2f} to {up_['slope'][1]:.2f}) and made the overall rating worse for both. Counting squeezed balls as pressure balls moved neither measure by more than 0.01.")
r3 = S['design_tests']['round3']
P("A second round started from that design plus the knockout weighting. Here 'rating predicts' tests the final rating rather than Level:", size=9.5)
T(pd.DataFrame({'Design': [r[0] for r in r3], 'Batters: repeatable': [f"{r[1]:.2f}" for r in r3], 'Batters: rating predicts': [f"{r[3]:.2f}" for r in r3], 'Bowlers: repeatable': [f"{r[4]:.2f}" for r in r3],
                'Bowlers: rating predicts': [f"{r[6]:.2f}" for r in r3], 'Verdict': [r[7] for r in r3]}), widths=[5.4, 2.1, 2.3, 2.1, 2.3, 3.2], size=8)
P(f"Counting balls instead of matches gave the largest gain of the project. Together, the match-up adjustment and ball-level consistency raised repeatability from {r3[0][1]:.2f} to {r3[4][1]:.2f} for batters and from {r3[0][4]:.2f} to {r3[4][4]:.2f} for bowlers, and they lifted the bowling rating's prediction of unseen matches from {r3[0][6]:.2f} to {r3[4][6]:.2f}. A venue adjustment, removing each ground's tilt towards bat or ball, made batters' ratings less repeatable ({r3[2][1]:.2f} against {r3[0][1]:.2f}) and bowlers' no better, because par and the opposition adjustment already handle the scoring environment. We left it out.")
doc.add_heading("A.2 The size of the knockout weight", 2)
P("We chose the size of the weight with the same test, always scored against the same unweighted yardstick:")
T(pd.DataFrame({'Knockout weighting': [r[0] for r in kt], 'Major final / IPL playoff / minor knockout': [r[1] for r in kt], 'Batters: repeatable': [f"{r[2]:.2f}" for r in kt], 'Batters: predicts': [f"{r[3]:.2f}" for r in kt],
                'Bowlers: repeatable': [f"{r[4]:.2f}" for r in kt], 'Bowlers: predicts': [f"{r[5]:.2f}" for r in kt]}), widths=[4.4, 4.6, 2.0, 2.0, 2.0, 2.0], size=8)
P(f"At the adopted scale every measure stays within 0.01 of the unweighted rating, inside the noise of the test. Heavier weights make bowlers' ratings steadily less repeatable ({kt[0][4]:.2f} with no weight, {kt[2][4]:.2f} at twice the scale, {kt[4][4]:.2f} at eight times), because more of each rating rests on {ko['ko_matches']} matches. Knockout evidence is also thin per player, a median of {ko['bat_elig_ko_press_median']:.1f} knockout pressure balls for an eligible batter and {ko['bowl_elig_ko_press_median']:.1f} for a bowler, and only players in strong sides reach knockouts at all. A light weight recognises the biggest matches without letting a few teams' fixtures decide the lists.")
doc.add_heading("A.3 The three win-probability models", 2)
T(pd.DataFrame({'Win-probability model': mods, 'How it works': ['Average of logistic regression and constrained boosted trees in the chase; logistic regression in the first innings', 'Logistic regression on cricket-shaped features, both innings', 'Constrained boosted trees, both innings'],
                'Log-loss, 1st inns': [f"{WPLL[m][0]:.3f}" for m in mods], 'Log-loss, chase': [f"{WPLL[m][1]:.3f}" for m in mods],
                'Batters: same top 10': [f"{cs_['bat']['agreement'][m]['overlap']} of 10" for m in mods], 'Bowlers: same top 10': [f"{cs_['bowl']['agreement'][m]['overlap']} of 10" for m in mods]}), widths=[3.2, 6.4, 1.9, 1.7, 1.9, 1.9], size=8)
P("We call this average the consensus. A player now earns a place by rating well under all three models, not because one model happens to suit the situations they play in. The three runs use identical resamples and match splits, so the averaged rating has reliability figures of its own:")
T(pd.DataFrame({'Test': ['Rating agrees between random halves of matches', 'Rating predicts actual pressure value in unseen matches', 'Our top 10 kept across the eleven reruns (average)'],
                'Batters: main model alone': [f"{cs_['bat']['rel_models'][_m]:.2f}", f"{cs_['bat']['pred_models'][_m]:.2f}", f"{keep_b(rbm):.0f}%"],
                'Batters: consensus': [f"{cs_['bat']['rel']:.2f}", f"{cs_['bat']['pred']:.2f}", f"{keep_b(rbc):.0f}%"],
                'Bowlers: main model alone': [f"{cs_['bowl']['rel_models'][_m]:.2f}", f"{cs_['bowl']['pred_models'][_m]:.2f}", f"{keep_w(rbm):.0f}%"],
                'Bowlers: consensus': [f"{cs_['bowl']['rel']:.2f}", f"{cs_['bowl']['pred']:.2f}", f"{keep_w(rbc):.0f}%"]}), widths=[6.0, 2.6, 2.4, 2.6, 2.4], size=8)
_d = lambda r_, k_: (cs_[r_]['rel'] - cs_[r_]['rel_models'][_m]) if k_ == 'rel' else (cs_[r_]['pred'] - cs_[r_]['pred_models'][_m])
_word = lambda v: 'slightly better' if v > 0.005 else ('slightly worse' if v < -0.005 else 'no different')
P(f"Averaging does not buy accuracy. Against the main model alone, the consensus is {_word(_d('bat', 'rel'))} on repeatability and {_word(_d('bat', 'pred'))} on prediction for batters, and {_word(_d('bowl', 'rel'))} and {_word(_d('bowl', 'pred'))} for bowlers; where it falls short, the two weaker models are the reason. All these differences lie within the noise of the tests. What averaging buys is independence from a single modelling choice. The names that all three models place in the top 10 form the firm group of Section 6, and averaging only orders the contested places, exactly where picking one model would have been arbitrary.")
WS = json.load(open(f"{OUT}/weights_summary.json")) if os.path.exists(f"{OUT}/weights_summary.json") else None
_bp = os.path.join(os.path.dirname(OUT), 'kaggle', 'brief_version', 'output', 'consensus.pkl')
BRF = pickle.load(open(_bp, 'rb'))['summary'] if os.path.exists(_bp) else None
if WS:
    doc.add_heading("A.4 Choosing the weights", 2)
    P("We fitted the weights themselves to predict actual pressure value in unseen matches (a regression that finds the non-negative weights best predicting the held-out half, using 40 splits of the main model; the weights were fitted on some splits and judged on others, so they were never scored on the data that chose them):", size=9.5)
    _k = ['Brief 50/25/25', 'Level only', 'Level + consistency 50/0/50', 'Equal thirds', 'Learned on unseen matches (cross-fitted)']
    _lw = lambda r_: f"{WS[r_]['learned'][0] * 100:.0f} / {WS[r_]['learned'][1] * 100:.0f} / {WS[r_]['learned'][2] * 100:.0f}"
    T(pd.DataFrame({'Weights (level / uplift / consistency)': ['The brief\'s 50 / 25 / 25', 'Level only: 100 / 0 / 0', '50 / 0 / 50', 'Equal thirds', f"Fitted: batters {_lw('bat')}; bowlers {_lw('bowl')}"],
                    'Batters: rating predicts': [f"{WS['bat']['results'][k_]:.2f}" for k_ in _k], 'Bowlers: rating predicts': [f"{WS['bowl']['results'][k_]:.2f}" for k_ in _k]}), widths=[9.0, 3.2, 3.2], size=8)
    def _swap(tab, w):
        s_ = (w['level'] * tab.z_level + w['uplift'] * tab.z_uplift + w['cons'] * tab.z_cons); new = list(tab.assign(s_=s_).sort_values('s_', ascending=False).name.head(10)); old = list(tab.sort_values('rank').name.head(10))
        return [n for n in new if n not in old], [n for n in old if n not in new]
    _br = dict(level=0.5, uplift=0.25, cons=0.25)
    (bi, bo), (wi, wo) = _swap(bat, _br), _swap(bowl, _br)
    _chg = lambda i, o: 'the same names in another order' if not i else f"{', '.join(i)} instead of {', '.join(o)}"
    _full = (f" Run in full under all three models, the fitted weights also made the rating more repeatable (split-half {cs_['bat']['rel']:.2f} against {BRF['bat']['rel']:.2f} for batters, {cs_['bowl']['rel']:.2f} against {BRF['bowl']['rel']:.2f} for bowlers) and steadier from season to season ({cs_['bat']['yy']:.2f} against {BRF['bat']['yy']:.2f}, and {cs_['bowl']['yy']:.2f} against {BRF['bowl']['yy']:.2f})." if BRF else "")
    P(f"The fitted weights give uplift nothing, for either list, because it adds no prediction beyond what Level carries. They predict slightly better than the brief's 50/25/25 ({WS['bat']['results']['Learned on unseen matches (cross-fitted)']:.2f} against {WS['bat']['results']['Brief 50/25/25']:.2f} for batters, {WS['bowl']['results']['Learned on unseen matches (cross-fitted)']:.2f} against {WS['bowl']['results']['Brief 50/25/25']:.2f} for bowlers)." + _full + f" We adopted them. The brief asks how players perform compared with their usual level, and Level still answers that: it moves each player away from their own record only as far as their pressure evidence allows. With the brief's weights, the batting top 10 would hold {_chg(bi, bo)}, and the bowling top 10 would hold {_chg(wi, wo)}.")
ins = S['insights']
doc.add_heading("A.5 Ideas we tested and dropped", 2)
_ha_b = ins['bat: away v home, all balls']; _ha_w = ins['bowl: away v home, all balls']
B(f"**Home and away.** Penalising players who perform only at home would need individual home-away gaps to be real. Compared with themselves, batters do {_ha_b['effect']:+.1f} points per 100 balls away (standard error {_ha_b['se']:.1f}) and bowlers {_ha_w['effect']:+.1f} ({_ha_w['se']:.1f}). Individual gaps are wide, about {_ha_b['sd_gap']:.0f} points, but they do not repeat between halves of the matches ({_ha_b['rel']:.2f} and {_ha_w['rel']:.2f}), so penalising them would penalise chance.")
_fs_b = ins['bat: after a form slump v otherwise, all balls']; _fs_w = ins['bowl: after a form slump v otherwise, all balls']
B(f"**Form slumps.** After three innings totalling under 30 runs, the same batter is worth {_fs_b['effect']:+.1f} points per 100 balls (standard error {_fs_b['se']:.1f}); after three spells at 10 an over or worse, the same bowler is worth {_fs_w['effect']:+.1f} ({_fs_w['se']:.1f}). We found no effect, though players who were actually dropped are missing from the comparison.")
ar = S['allround']; at_ = S['design_tests']['allround']
hyp = lambda tab: list(tab.assign(h=tab.core + 0.5 * tab.ar).sort_values('h', ascending=False).name.head(10))
hb, hw = hyp(bat), hyp(bowl)
_gw = [n for n in hw if n not in set(bowl.name.head(10))]; _gb = [n for n in hb if n not in set(bat.name.head(10))]
_ar_eff = ("would change neither list" if not _gw and not _gb else "would bring in " + ', '.join(_gb + _gw))
B(f"**A same-match all-round bonus.** Players added win probability with both bat and ball (at least six balls of each) in {ar['dbl']} of the {ar['both']:,} player-matches in which they did both. We built a bonus for this and left it out. The brief ranks batters and bowlers separately on their own discipline; batting and bowling value in the same match are nearly unrelated (correlation {ar['corr']:.2f}; positive with both in {pc(ar['pos_both'])} of such matches, against {pc(ar['indep'])} if independent); and how often a player manages it does not repeat between halves of the matches ({ar['rel']:.2f}). The bonus rewards a role rather than a pressure skill; it also made the batting rating predict slightly worse ({at_[0][3]:.2f} to {at_[1][3]:.2f}), and it {_ar_eff}.")
rc = S['recency']
B(f"**Recency.** Weighting recent matches more, with half-lives from six months to three years, did not consistently improve predictions of the latest 25% or 40% of matches. For bowlers the unweighted rating predicted the last 40% at {rc['bowl|last 40% of matches']['none']['score']:.2f}, against {rc['bowl|last 40% of matches']['2.0y']['score']:.2f} with a two-year half-life, and for batters the result flipped with the cut-off. The rating treats 2022 to 2026 as one period.")
B("**Playing style.** Adding shot selection, delivery mix and bowling speed to Level made predictions of unseen matches much worse; with 97 players, these features fit noise.")
B("**Fielding.** The ball-by-ball file records no fielder and no dropped chances, so catches and drops cannot be credited.")
doc.add_heading("A.6 Checks from published work on clutch", 2)
ck = S['checks']; aw = S['altwp']; pw = S['more']
P("Baseball analysts have studied 'clutch' for decades: Tom Tango's leverage index is the source of our stakes measure, and the FanGraphs Clutch score is the standard published clutch statistic. That work, and recent leverage studies of IPL chases, sets tests that any clutch claim should pass. We ran each one. Some report a p-value: the chance of a result at least this large if luck alone were at work. A small p-value (below 0.05) suggests a real effect; a large one means luck explains the result.")
B(f"**Does clutch skill exist in this data?** For every eligible player we shuffled the pressure labels among that player's own balls 400 times and recomputed uplift. If some players truly rose under pressure, the real spread of uplift between players would exceed the shuffled spread. It does not: {ck['bat_perm']['sd_obs']:.1f} points per 100 balls against {ck['bat_perm']['sd_null_mean']:.1f} shuffled for batters (p = {ck['bat_perm']['p_spread']:.2f}), and {ck['bowl_perm']['sd_obs']:.1f} against {ck['bowl_perm']['sd_null_mean']:.1f} for bowlers (p = {ck['bowl_perm']['p_spread']:.2f}). {num(ck['bat_perm']['n_sig'])} of {ck['bat_perm']['players']} batters and {ck['bowl_perm']['n_sig']} of {ck['bowl_perm']['players']} bowlers fall outside their own 95% chance range, close to the {ck['bat_perm']['expected_by_chance']:.0f} that chance alone predicts. This data cannot tell any player's pressure uplift from luck.")
_sb = andj([x[0] for x in ck['bat_perm']['sig'] if x[1] > 0]); _sw = andj([x[0] for x in ck['bowl_perm']['sig'] if x[1] > 0])
B(f"**What that means for the lists.** They rank the players worth most in pressure situations, which the data can measure, not the players who raise their game, which it cannot. That is why uplift has no separate weight and keeps a fifth of its measured size or less inside Level. The largest raw uplifts belong to {_sb} among batters, and to {_sw} among bowlers. They are the best candidates for a real effect, but with about {ck['bat_perm']['expected_by_chance']:.0f} such players expected by chance in each list, none can be singled out.")
_pw = lambda r_, s_: pc(pw[f'{r_}_power']['power'][s_])
B(f"**Could the test have found clutch skill?** We planted clutch differences of known size on top of shuffled data and reran the test 1,000 times at each size. The test detects a true spread of 15 points per 100 balls {_pw('bat', '15')} of the time for batters and {_pw('bowl', '15')} for bowlers; 10 points, {_pw('bat', '10')} and {_pw('bowl', '10')}; 5 points, {_pw('bat', '5')} and {_pw('bowl', '5')}. Players differ in overall ability by about {pw['bat_power']['level_sd']:.0f} points for batters and {pw['bowl_power']['level_sd']:.0f} for bowlers. Clutch differences as large as differences in ability would very likely have shown up, so any clutch skill in this pool is clearly smaller than the gaps in overall quality.")
B(f"**The standard Clutch formula.** We computed the FanGraphs definition (win probability added divided by average leverage, minus the ball-by-ball sum of win probability added divided by leverage) for every player. It correlates {ck['bat_fg']['corr_raw_uplift']:.2f} with our raw uplift for batters and {ck['bowl_fg']['corr_raw_uplift']:.2f} for bowlers, so both measure the same thing, and it is just as unstable (split-half {ck['bat_fg']['rel']:.2f} and {ck['bowl_fg']['rel']:.2f}). Ranked on that formula alone, the lists would be led by {ck['bat_fg']['top5'][0][0]} and {ck['bowl_fg']['top5'][0][0]}, whom our rating places {ordn(int(ck['bat_fg']['top5'][0][2]))} and {ordn(int(ck['bowl_fg']['top5'][0][2]))}. An unshrunk clutch score should not rank players.")
B(f"**Drift in the win-probability model.** A miscalibrated model leaks spurious value into some match states. Ours moves win probability by {ck['drift']['overall'] * 100:+.3f} points per ball on average and by at most {ck['drift']['max_abs'] * 100:.2f} points in any two-point band, against a typical swing of {ck['drift']['mean_abs_swing'] * 100:.2f} points per ball. Centring within each kind of ball removes even that, so we needed no separate correction.")
B(f"**Other win-probability models.** Each model's ratings correlate at least {min(v['corr'] for v in CON['summary']['bat']['agreement'].values()):.2f} with the consensus for batters and {min(v['corr'] for v in CON['summary']['bowl']['agreement'].values()):.2f} for bowlers. The models agree on the firm names and differ on the contested places, which is why the rating averages them (4.5).")
B(f"**How much data would be enough?** A rating reliability of 0.5 needs about {ck['bat_need']['score']:.0f} pressure balls for a batter and {ck['bowl_need']['score']:.0f} for a bowler; the median eligible player has {2 * ck['bat_need']['half_balls']:.0f} and {2 * ck['bowl_need']['half_balls']:.0f}. Level alone needs {ck['bat_need']['level_eb']:.0f} and {ck['bowl_need']['level_eb']:.0f}. Uplift would need about {ck['bat_need']['uplift_eb']:,.0f} and {ck['bowl_need']['uplift_eb']:,.0f}, more than any player in the data has faced or bowled.")
doc.add_heading("A.7 How players behave on pressure balls", 2)
beh = {r['name']: r for r in S['more']['behaviour']}
bd = lambda k: f"{100 * beh[k]['diff']:+.1f}"; be = lambda k: f"{100 * beh[k]['se']:.1f}"
B(f"Players behave differently on these balls. Within the same innings and over, the dot-ball rate changes by {bd('Dot ball')} percentage points (standard error {be('Dot ball')}; the standard error is the margin of uncertainty, and an effect about twice its size or more is unlikely to be chance), unorthodox shots such as scoops and switch hits by {bd('Innovative shot (scoop, reverse, switch, uppercut)')} ({be('Innovative shot (scoop, reverse, switch, uppercut)')}), and pace bowlers' full tosses by {bd('Pace: full toss')} ({be('Pace: full toss')}). Batters refuse dot balls and reach for options; fast bowlers miss their length. Boundary and wicket rates do not move ({bd('Boundary')} and {bd('Wicket')}), which is why measures built on outcomes alone find so little 'clutch'.")
doc.add_heading("A.8 Related work", 2)
P("Leverage-weighted win probability added comes from baseball and has been applied to T20 chases. Other work defines a pressure index for the side batting second (Bhattacharjee and Lemmer 2016; Bandyopadhyay and Bhattacharjee 2025) and rates runs above expectation with empirical-Bayes shrinkage (arXiv 2608.18020). The dependence between consecutive balls behind our squeeze test is documented in 'The Calibration-Leverage Tradeoff in Exactly Solvable Win-Probability Models' (arXiv 2608.14696). We found no published use of a standardised outcome exam, a within-player squeeze test, or a pressure rating chosen by held-out prediction.", italic=True, size=8.5)

doc.add_heading("Appendix B. Ranks 1-25", 1).paragraph_format.page_break_before = True
P("Rates are win-probability points per 100 balls of average leverage, with 0 as the average player. '(std.)' columns are the rating's parts as standardised scores (0 = pool average, 1 = one standard deviation). Uplift is shown for reference; it carries no separate weight. 'Rank range' is the 10th-to-90th-percentile rank across 500 resamples.", italic=True, size=8.5)
for nm, t in (('Batters', bat), ('Bowlers', bowl)):
    t = t.head(25)
    P(nm, bold=True)
    T(pd.DataFrame({'#': t['rank'].values, 'Player': t.name.values, 'Pressure balls': t.n_P.map(f0).values, 'Usual rate': (t.T_rN * U).map(sgn).values, 'Pressure rate': (t.T_rP * U).map(sgn).values,
                    'Level (std.)': t.z_level.map(lambda v: f"{v:+.2f}").values, 'Uplift (std.)': t.z_uplift.map(lambda v: f"{v:+.2f}").values, 'Consistency (std.)': t.z_cons.map(lambda v: f"{v:+.2f}").values,
                    'Rating': t.score.map(lambda v: f"{v:.2f}").values, 'Rank range': [f"{a:.0f}-{b:.0f}" for a, b in zip(t.rank_lo, t.rank_hi)]}), widths=[0.8, 4.2, 1.6, 1.5, 1.6, 1.4, 1.4, 1.7, 1.3, 1.6], size=8)

_last = doc.paragraphs[-1]
if not _last.text.strip():                 # the spacer after the final table would spill onto a blank page
    _last.paragraph_format.space_after = Pt(0); _last.paragraph_format.space_before = Pt(0); _last.paragraph_format.line_spacing = Pt(1)
doc.save(f"{DEST}/SupeRR_Selector_R1_Methodology_Report.docx")

with open(f"{DEST}/SUBMISSION_top10_lists.md", 'w', encoding='utf-8') as fh:
    fh.write("# SupeRR Selector Season 2, Round 1: lists to enter on Unstop\n\nNames are spelled exactly as in the supplied data.\n\n| Rank | Top 10 batters | Top 10 bowlers |\n|---|---|---|\n")
    for i in range(10):
        fh.write(f"| {i + 1} | {bat.name.iloc[i]} | {bowl.name.iloc[i]} |\n")
    fh.write("\nThe data pack has two players called Rashid Khan. Ours is player_id 4722 (Afghanistan and Gujarat Titans), not player_id 20172 (Nepal).\n")
    fh.write("\nUpload `SupeRR_Selector_R1_Methodology_Report.docx` (or its PDF) as the methodology report.\n")
print("report written")

"""Runs the whole method from the six supplied CSVs to the report, in order.

    python run_all.py                      # full run (about an hour on a 4-core machine with 8 GB of RAM)
    python run_all.py --quick              # skips the eleven robustness re-runs (about 25 minutes)

Inputs : the six supplied CSVs in  data_pack/Data Pack/  (not included in this repository; see README).
Outputs: output/, output_alt/, output_gbm/ (one per win-probability model) and submission/ (report .docx, lists, CSVs).
Two slow design-search results (static/design_tests.json from 12_harness.py, static/weights_summary.json from 23_weights.py)
are copied in rather than recomputed; run those two scripts yourself to regenerate them.
"""
import os, sys, json, shutil, subprocess, time

ROOT = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(ROOT, 'analysis')
DATA = os.path.join(ROOT, 'data_pack', 'Data Pack')
QUICK = '--quick' in sys.argv
ENV = dict(os.environ, RR_ROOT=ROOT, WEIGHTS='learned')
MODELS = (('Main (blend)', 'blend', 'output'), ('Logistic only', 'lr', 'output_alt'), ('Boosted trees only', 'gbm', 'output_gbm'))

if not os.path.exists(os.path.join(DATA, 'ball_by_ball.csv')):
    sys.exit(f'Put the six supplied CSVs in {DATA} first (see README).')
for d in ('output', 'output_alt', 'output_gbm', 'submission'):
    os.makedirs(os.path.join(ROOT, d, 'figs') if d.startswith('output') else os.path.join(ROOT, d), exist_ok=True)
for f in ('design_tests.json', 'weights_summary.json'):
    shutil.copy(os.path.join(ROOT, 'static', f), os.path.join(ROOT, 'output', f))


def run(script, *args, env=None):
    t0 = time.time()
    p = subprocess.run([sys.executable, '-I', os.path.join(AN, script), *map(str, args)], capture_output=True, text=True, env={**ENV, **(env or {})})
    with open(os.path.join(ROOT, 'run_log.txt'), 'a', encoding='utf-8') as log:
        log.write(f'\n===== {script} {args} {env}\n{p.stdout}\n{p.stderr}\n')
    print(f'{script:18s} {env or ""} -> exit {p.returncode} in {time.time() - t0:.0f}s', flush=True)
    if p.returncode:
        sys.exit(f'{script} failed; see run_log.txt')
    return p.stdout


# 1. Win-probability models (five-fold, out of sample), leverage and ball values, once per model
wpll = {}
for name, mode, d in MODELS:
    o = run('04_pipeline.py', DATA, AN, os.path.join(ROOT, d), env={'WPMODE': mode})
    wpll[name] = [float(l.split(':')[-1]) for l in o.splitlines() if l.startswith('OOF logloss')]
json.dump(wpll, open(os.path.join(ROOT, 'output', 'wp_logloss.json'), 'w'), indent=1)

# 2. Ratings under each model (500 resamples, 100 split halves, season to season)
for _, _, d in MODELS:
    run('05_ratings.py', os.path.join(ROOT, d))

# 3. Robustness re-runs on eleven altered datasets, under each model
if not QUICK:
    RER = [('_ipl', {'ONLY': 'IPL'}), ('_intl', {'ONLY': 'INTL'})] + [(f'_drop{y}', {'DROPYEAR': str(y)}) for y in range(2022, 2027)] + \
          [('_pct0.67', {'PCT': '0.67'}), ('_pct0.80', {'PCT': '0.80'}), ('_min100', {'MIN_PBALLS': '100'}), ('_min200', {'MIN_PBALLS': '200'})]
    for _, _, d in MODELS:
        for sfx, e in RER:
            run('05_ratings.py', os.path.join(ROOT, d), env={**e, 'SFX': sfx, 'NBOOT': '1', 'NSPLIT': '2'})

# 4. Average of the three models, firm and contested places, head-to-head odds, intervals
O = os.path.join(ROOT, 'output')
run('20_consensus.py', O); run('22_contest.py', O); run('25_intervals.py', O)

# 5. Checks and evidence
run('11_squeeze.py', O); run('14_knockout.py', O); run('15_insights.py', O); run('16_allround.py', O)
run('17_checks.py', O, 400); run('18_recency.py', O); run('19_more.py', O)
run('21_altwp.py', O, os.path.join(ROOT, 'output_alt'))
if not QUICK:
    run('24_external.py', O)          # needs the season and competition re-runs from step 3

# 6. Charts and the report
run('06_figures.py', O); run('07_report.py', O, os.path.join(ROOT, 'submission'))
print(open(os.path.join(ROOT, 'submission', 'SUBMISSION_top10_lists.md'), encoding='utf-8').read())

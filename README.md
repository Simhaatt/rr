# Who delivers when the match is on the line?

A pressure rating for T20 batters and bowlers, built for SupeRR Selector Season 2, Round 1 (Rajasthan Royals).
The method report is [`submission/SupeRR_Selector_R1_Methodology_Report.pdf`](submission/SupeRR_Selector_R1_Methodology_Report.pdf).

## The method in brief

1. **Win probability.** For every ball, a model estimates the batting side's chance of winning from balls left, runs, wickets, target and a par score for the competition and year. It is fitted in five groups of matches, so no match predicts itself.
2. **Pressure.** A ball's leverage is how far its outcome can move that chance. The top quarter of balls by leverage, weighted lightly for finals and playoffs, are pressure balls.
3. **Ball value.** Each ball is worth the change in win probability it caused. Half is the actual change, and half is a standard mark for that outcome in that kind of situation, which removes the luck of which ball a boundary fell on.
4. **Fair comparison.** Every ball is adjusted for the quality of the opponent, and compared with the average player in the same situation and squeeze state.
5. **Rating.** Level (worth per pressure ball, shrunk towards the average by sample size) and Consistency (how rarely a pressure ball is lost). The weights were fitted on matches the rating had not seen: batters 0.68 / 0.32, bowlers 0.38 / 0.62.
6. **Three models.** The whole method runs under three win-probability models, and the final rating is their average.

## Reproducing the results

Requirements: Python 3.14 (3.12 or later should work), about 8 GB of memory, and the packages pinned in `requirements.txt`.

1. Put the six supplied CSVs (`ball_by_ball.csv`, `competitions.csv`, `competition_team_player_mapping.csv`, `matches.csv`, `playing_eleven.csv`, `teams.csv`) in `data_pack/Data Pack/`. The competition data is not redistributed here.
2. Install the packages and run everything:

```bash
pip install -r requirements.txt
python run_all.py
```

The full run takes about an hour on a four-core machine. `python run_all.py --quick` skips the eleven robustness re-runs and the held-out season and competition test, and takes about 25 minutes. Random seeds are fixed, so the run reproduces every number in the report.

Outputs: `output/`, `output_alt/` and `output_gbm/` (one folder per win-probability model), and `submission/` (report `.docx`, the two lists and the rating CSVs). The PDF was exported from the `.docx` with Microsoft Word.

Two slow design searches are supplied as results in `static/`, not recomputed by `run_all.py`: `design_tests.json` (from `analysis/12_harness.py`) and `weights_summary.json` (from `analysis/23_weights.py`). Run those scripts to regenerate them.

## Scripts

| Script | What it does |
|---|---|
| `common.py` | Loads the six files, marks legal balls, rebuilds the match state, filters matches and computes par |
| `04_pipeline.py` | Fits a win-probability model (`WPMODE` = blend, lr or gbm) and computes leverage and the value of every possible outcome of every ball |
| `05_ratings.py` | Pressure and squeeze flags, ball value, opponent adjustment, shrinkage, the rating, resampling and split-half tests |
| `20_consensus.py`, `22_contest.py` | Average of the three models, firm and contested places, head-to-head odds |
| `25_intervals.py` | 90% intervals for each rating and for the gap between neighbours |
| `24_external.py` | Held-out tests: built on 2022-2025 and scored on 2026, built on IPL and scored on internationals, and the reverse |
| `11`, `14`-`19`, `21` | Checks: squeeze, knockouts, home and away, all-round, clutch tests, recency, calibration, model agreement |
| `05b`, `05c`, `08`-`10`, `12`, `13`, `23` | Design searches and rejected ideas (see the report's Appendix A) |
| `01`-`03`, `04b`, `04c` | Early data profiling and model comparisons |
| `06_figures.py`, `07_report.py` | Charts and the report |

`explainer/` holds a plain-language walkthrough of the whole project (`index.html`, built by `build_explainer.py`).

## Final lists

| Rank | Batters | Bowlers |
|---|---|---|
| 1 | Vaibhav Sooryavanshi | Sunil Narine |
| 2 | Heinrich Klaasen | Krunal Pandya |
| 3 | Phil Salt | Axar Patel |
| 4 | Abhishek Sharma | Jasprit Bumrah |
| 5 | Tim David | Kuldeep Yadav |
| 6 | Suryakumar Yadav | Rashid Khan |
| 7 | Josh Inglis | Josh Hazlewood |
| 8 | Shreyas Iyer | Ravichandran Ashwin |
| 9 | Virat Kohli | Ravindra Jadeja |
| 10 | Cameron Green | Wanindu Hasaranga |

# Simulator: PPO decision period

The original simulator agent (`../cg_run.py`, method `CongestAgent`) runs the PPO policy every 8 intervals (2 s).
This folder chooses the decision period on validation seeds and evaluates the chosen agent on the test seeds.
`../cg_sim.py` and `../cg_run.py` are used unchanged.

## Scripts
- `train_p.py <method> <period>` trains `METHODS[method]` of `cg_run.py` with the given period (in 250 ms intervals):
  300,000 steps, seed 0, same PPO settings as `cg_run.py`. Writes `models/<method>_p<period>.zip`.
- `eval_p.py <val|test|abl> <out.csv> <model names>` runs the same `run_episode` harness as `cg_run.py`:
  `val` = 18 core settings x seeds 201-205 (90 cases), `test` = 18 core settings x seeds 101-110 (180 cases, same as
  `../cg_out/results/core.csv`), `abl` = the six ablation settings x seeds 101-105 (30 cases).
- `analyze.py` gives means with 95% CI and Holm-corrected paired tests against B1, B2, B3 and the 2 s agent
  (`../cg_out/results/core.csv`; the simulated world is seeded, so the baselines are identical).

## Choice of the period (validation seeds only, criterion: lowest mean p95 delay)
| Period | 0.25 s (1) | 0.5 s (2) | 1 s (4) | 2 s (8) |
|---|---|---|---|---|
| p95 delay (ms) | 85.0 | **78.9** | 89.4 | 109.6 |

Files: `val_a.csv`, `val_b.csv`. The chosen period, 0.5 s, was then evaluated once on the test seeds.

## Results
- `test_p2.csv`: 0.5 s agent on the 180 test cases; `test_paired_p2.csv`: paired tests (Table 7 of the paper).
- `test_p8.csv`: the 2 s agent retrained here. PPO training is not bit-for-bit reproducible across machines, so this
  model is not identical to `../cg_out/models/CongestAgent.zip`; the paper reports the original 2 s model.
- `abl_a.csv`, `abl_b.csv`, `abl_paired_p2.csv`: ablations (Table 8). A1, A2, A3 and A5 are retrained with the 0.5 s
  period; A4 (policy every 250 ms, no fast loop) is `../cg_out/models/A4.zip`.
- `models/`: the trained PPO models.

## How to rerun
From this folder, copy `../cg_sim.py`, `../cg_run.py` and `../calib.json` here and copy `../results` to `clfres`, then:

    python train_p.py CongestAgent 2
    python eval_p.py test test_p2.csv CongestAgent_p2
    python analyze.py

For the ablations, copy `../cg_out/models/A4.zip` to `models/A4_p1.zip` first.

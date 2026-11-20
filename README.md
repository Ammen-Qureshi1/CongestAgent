# CongestAgent: Cause-Aware Congestion Detection and Adaptive Regional Traffic Engineering

Code, data, trained models and results for the paper *CongestAgent: Cause-Aware Congestion Detection and
Adaptive Regional Traffic Engineering*.

CongestAgent labels every link interval (250 ms) as **normal**, **congestion** or **non-congestive degradation**
(random loss, jitter, delay shift) before it reroutes traffic, and splits control into regional agents with a
fast loop (250 ms) and a slow loop. It is evaluated in two environments:

* **Emulation** (Mininet + Open vSwitch + Linux `tc`, real CUBIC and BBR traffic): labelled dataset of 199 runs
  and 268 closed-loop runs (RQ1, RQ2, RQ3).
* **Flow-level simulator**: PPO regional agents against static routing and two central DQN baselines.

## Repository layout

| Folder | What it contains |
|---|---|
| `datagen/` | Emulation testbed and dataset generator (Mininet, Open vSwitch, `tc`, traffic agents, probes, labelling, features) |
| `topologies/` | NSFNET and GÉANT from SNDlib, and the script that builds them |
| `long_features.py` | The 14 long-window (5 s and 10 s) features |
| `docs/` | How to run the testbed (`RUNBOOK.md`), dataset description (`README_DATASET.md`) and datasheet (`DATASHEET.md`) |
| `models/` | Offline models and analysis (51- and 65-feature LightGBM, busy-link gate, robustness checks, CUBIC–BBR fairness) |
| `live/` | Closed-loop controller and experiment harness (`live_run.py`), batch scripts, evaluation, statistics and figures |
| `models/results/` | Trained models (`lgbm_full.txt`, `lgbm_long.txt`, `lgbm_live.txt`, `live_rules.json`, `robust/`), offline results, closed-loop per-run tables and paired tests, console outputs, figures and figure data |
| `CongestAgent/` | Interval datasets (`*.csv.gz`, Git LFS), perception training (`train_perception_colab.py`, `train_gru.py`), perception models for five feature sets (`results/`), flow-level simulator (`cg_sim.py`, `cg_run.py`, `cg_calibrate.py`, `cg_plots.py`, `calib.json`), original simulator results with PPO deciding every 2 s (`cg_out/`) |
| `CongestAgent/simulator_period/` | Simulator results with the PPO decision period chosen on validation seeds (0.5 s), ablations, paired tests and trained models |
| `paper/` | Paper figures (`figures/`) and the script that draws them (`figs.py`) |

## Where each result in the paper comes from

| Paper item | Script | Result file(s) |
|---|---|---|
| Table 2 (dataset splits) | `datagen/build_all.py` | `docs/README_DATASET.md` |
| Table 3 (offline perception) | `models/train_models.py`, `models/train_long.py`, `models/robustness_checks.py`, `live/train_live_model.py`, `CongestAgent/train_perception_colab.py`, `live/check_robust.py` | `models/results/results.json`, `results_long.json`, `robustness_checks.json`, `live_model_report.json`, `console_check_robust.txt`, `CongestAgent/results/*/metrics.json` |
| Tables 4–5, Figs. 2–3 (RQ1, closed loop) | `live/live_batch.py` → `live/live_eval.py`, `live/live_extra.py` | `models/results/live_per_run.csv`, `live_tests.csv`, `live_extra_tests.csv`, `console_live_eval_and_extra.txt` |
| Table 6, Fig. 4 (RQ2, regional vs central) | `live/live_batch_rq2.py` → `live/rq2_eval.py` | `models/results/rq2_per_run.csv`, `rq2_tests.csv`, `console_rq2_eval.txt` |
| Fig. 5 (RQ3, CUBIC–BBR fairness) | `live/live_extra.py`, `models/rq3_fairness.py` | `models/results/live_rq3_per_run.csv`, `rq3_fairness.json` |
| Table 7 (simulator core) | `CongestAgent/cg_run.py`, `CongestAgent/simulator_period/eval_p.py` | `CongestAgent/simulator_period/test_p2.csv`, `test_paired_p2.csv`, `CongestAgent/cg_out/results/core.csv` |
| Table 8 (simulator ablation) | `CongestAgent/simulator_period/eval_p.py` | `CongestAgent/simulator_period/abl_*.csv`, `abl_paired_p2.csv` |
| Figures 1–5 | `paper/figs.py` | `paper/figures/*.png` (data: `models/results/figures/figure_data.csv`) |

## Requirements

* Ubuntu 22.04/24.04 with root access for the testbed: `mininet`, `openvswitch-switch`, `iproute2` (`tc`, netem),
  kernel modules `openvswitch`, `sch_netem`, `tcp_bbr` (see `docs/RUNBOOK.md`).
* Python 3.10+ and the packages in `requirements.txt` (`pip install -r requirements.txt`).
* All tools are free and open source.

## How to reproduce

1. **Dataset**: follow `docs/RUNBOOK.md` (`datagen/smoke_test.py`, then `datagen/batch.py`,
   then `datagen/build_all.py`). The processed interval files used in the paper are already in `CongestAgent/*.csv.gz`.
2. **Perception models**: from `models/`, run `python train_models.py` (51-feature model and ablations),
   `python train_long.py` (65-feature model) and `python robustness_checks.py` (robustness checks);
   `python3 live/train_live_model.py` (model, rule thresholds and busy-link gate used in the live loop);
   `python CongestAgent/train_perception_colab.py --featureset all` (robust model and other feature sets).
3. **Closed-loop experiments** (as root): `sudo python3 live/live_batch.py` (RQ1 and RQ3) and
   `sudo python3 live/live_batch_rq2.py` (88 runs, RQ2). Run folders are written to `live_runs/`.
4. **Evaluation**: `python3 live/live_eval.py`, `python3 live/live_extra.py`, `python3 live/rq2_eval.py`,
   `python3 live/make_figures.py`.
5. **Simulator**: `python CongestAgent/cg_run.py --stage all --results-dir CongestAgent/results --calib
   CongestAgent/calib.json` (original 2 s agent), then the scripts in `CongestAgent/simulator_period/` (see `CongestAgent/simulator_period/README.md`)
   for the 0.5 s agent and the ablations.
6. **Figures**: `cd paper && python figs.py`.

## Notes

* Seeds: dataset generation uses fixed seeds per run; closed-loop runs use seeds 11–13 (never used for training);
  the simulator uses seeds 101–110 for testing and 201–205 for choosing the PPO decision period.
* The busy-link gate threshold (mean utilisation ≥ 0.541) is the 1% quantile of utilisation over training
  congestion rows, computed from training data only.
* In the simulator, the GÉANT topology is a synthetic stand-in with 22 nodes and 36 links (`cg_sim.py`); the
  paper reports simulator results on NSFNET only. The emulation uses the real SNDlib GÉANT topology.
* PPO training is not bit-for-bit reproducible across machines, so retrained agents give slightly different
  numbers; the trained models used in the paper are included.
* Location of the processed dataset: the scripts in `models/` read `~/ca_dataset/dataset`, and `live/train_live_model.py` and
  `live/check_robust.py` read `dataset/` at the repository root (variable `D` at the top of each script).
* The raw run folders (`runs/`, `live_runs/`) are not included because of their size.

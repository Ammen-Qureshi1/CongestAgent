# CongestAgent dataset

Labelled network measurements from an emulated network (Mininet + Open vSwitch + netem/tbf, real Linux
TCP CUBIC and BBR). Each row = **one link direction during one 250 ms interval**.
Full column list and generation details: `DATASHEET.md`.

## Files (folder `CongestAgent/`, stored with Git LFS)
| File | What it is |
|---|---|
| `intervals_train_features.csv.gz` | training split (all columns + engineered features) |
| `intervals_val_features.csv.gz` | validation split (early stopping, tuning) |
| `intervals_test_features.csv.gz` | test split, NSFNET |
| `intervals_test_geant_features.csv.gz` | test split, GÉANT (a network never used for training) |
| `intervals_test_stress_features.csv.gz` | test split, stress cases (all-CUBIC, traffic bursts, link failure) |
| `flows_tcp_all.csv.gz` | every TCP transfer: size, CUBIC/BBR, start, completion time |
| `flow_seconds_all.csv.gz` | bytes per flow per second (for fairness / Jain's index) |
| `flows_udp_all.csv.gz` | UDP flows |

Read with pandas: `pd.read_csv(path, low_memory=False)` (the text column `gt_label_snapshot` otherwise gives a harmless type warning).

## Size
199 runs, 4,242,240 rows. Usable rows = after the 20 s warm-up, certain label, full 2 s window
(`~meta_warmup & gt_label != 'uncertain' & w_complete`):

| Split | Runs | Usable rows | normal | degradation | congestion |
|---|---|---|---|---|---|
| train | 87 | 1,418,419 | 1,232,432 | 164,091 | 21,896 |
| val | 24 | 391,315 | 336,058 | 49,249 | 6,008 |
| test | 48 | 783,111 | 677,131 | 94,378 | 11,602 |
| test_geant | 16 | 448,369 | 391,561 | 50,659 | 6,149 |
| test_stress | 24 | 386,014 | 321,462 | 54,905 | 9,647 |

Splits are **by run** (never random rows). Congestion is rare (about 1.5 %): use class weights and report
macro F1 and per-class F1, not accuracy.

## Labels (`gt_label`)
From the real queue backlog (tbf, sampled every 50 ms):
* `congestion`: queue > 50 % full in this and the 2 previous intervals
* `degradation`: noise (random loss, jitter or delay shift) on for the whole interval, queue < 20 %
* `normal`: no noise, queue < 20 %
* `uncertain`: anything else — **leave out** of training and testing

## Which columns to use
* **Features (model inputs):** only columns starting with `f_`, `p_`, `w_`, `nb_`, `static_` (51 columns).
  `w_complete` is a flag, not a feature (use it to filter).
* **Never use `gt_*` or `meta_*` as inputs.** `gt_*` columns contain the answer (queue size, noise level);
  `meta_*` are run settings and ids.
* Our LightGBM model also adds 14 longer-window features (5 s and 10 s rolling means of path loss, port loss,
  delay spread, excess delay, probe loss; see `long_features.py`). Group rolling windows by
  `meta_run_id, meta_dir_id` and sort by `meta_k`, so nothing leaks between runs or directions.
  **`long_features.py` does this for you** (`add_long()`, then `usable()`): call `add_long` on the full file first and
  filter the rows after, never before.
* The features are **not scaled** (LightGBM does not need it). For a neural network or RL, fit min-max scaling on the
  train split only, and fill the NaNs of `w_loss_when_util_high/low` (empty by design).

## Conditions covered
* Topologies: NSFNET (14 nodes, train/val/test) and GÉANT (22 nodes, test only)
* Load on the busiest link: 30 / 60 / 90 %
* Noise: random loss 0.5 / 1 / 2 %; jitter 5 / 10 / 20 / 40 ms; delay shift +40 / +80 ms (extra link delay with no queue)
* Not used for training: loss 2 %, jitter 40 ms, shift +80 ms, the whole GÉANT network (unseen-condition tests)
* Traffic: mixed CUBIC + BBR (and all-CUBIC stress runs), traffic bursts, link failures

## Known issues
* **Utilisation readings:** runs started between 10:00 and 15:00 on 18 Nov (the harder-noise runs and most
  extra-seed runs) have `f_util` above 1.10 in about 0.3–1.3 % of intervals (other runs: about 0 %).
  The port counters were read slightly late when the machine was busy. Labels, delay, loss and probes are not
  affected. If needed, clip `f_util` at 1.0.
* Measured jitter (probe spread) is about half to two-thirds of the configured jitter
  (5→3.5, 10→6.0, 20→10.8, 40→19.6 ms). Labels use the configured noise.
* `gt_noise_extra_delay_ms` is empty for the first 80 runs (built before the column existed); it means 0.
* QC: 199 runs, 0 FAIL, 104 WARN (utilisation readings only). Data checks: 15 of 18 pass; the 3 that fail are
  the two points above and a weak load check (the stronger check, "more load → more congestion", passes).

# CongestAgent dataset — datasheet

Generator: `datagen/` (see `RUNBOOK.md`). Everything below is produced by the code.

## 1. Purpose
Labelled measurements for **RQ1** (tell real congestion from non-congestive degradation before
rerouting), plus the flow-level data needed for **RQ3** (CUBIC vs BBR fairness) and for calibrating the
flow-level simulator used to train the learned decision agents (**RQ2**).

## 2. Unit of analysis
One **link direction** during one **250 ms interval**. Each direction has its own
queue, so congestion can exist in one direction only.

## 3. Emulated network
| Item | Value | Source |
|---|---|---|
| Main topology | NSFNET, 14 nodes, 21 links (SNDlib `nobel-us`) | `topologies/nsfnet.json` |
| Unseen test topology | GÉANT, 22 nodes, 36 links (SNDlib `geant`) | `topologies/geant.json` |
| Link capacity | 10 Mbps (tbf) | design |
| Queue | 150,000 bytes (~100 full-size packets), tbf `limit` | design |
| Propagation delay | real link length / 200 km per ms, clipped to 2–10 ms | design |
| Per-direction qdiscs | `netem` root (delay + noise) → `tbf` child (rate + queue) | design |
| Switches | Open vSwitch, fail-secure, no controller; fixed shortest-path OpenFlow rules (one per source–destination pair per switch, lexicographically smallest shortest path) | — |
| Hosts | one per node, static ARP; TSO/GSO/GRO off so every queued packet is a real packet | — |

Because the noise lives in `netem` and the queue lives in `tbf`, **the tbf backlog measures queueing
only**, and the kernel counts queue (congestion) drops and netem (noise) drops separately.

## 4. Traffic (per run)
* K random source–destination pairs (30 on NSFNET, 45 on GÉANT), weights lognormal(0, 0.8).
* Pair rates are scaled so the busiest link direction carries **30 / 60 / 90 %** of capacity on average
  (target and realised values are stored in `schedule.json` and `qc.json`).
* Half of each pair's rate is **TCP**: finite transfers, Poisson arrivals, sizes lognormal
  (mean ≈ 400 KB, σ = 1.2, cut to 10 KB–8 MB); each flow is **CUBIC or BBR** (50/50 in `mixed`,
  all CUBIC in `cubic`). Half is **UDP** ON/OFF (exponential, mean ON 4 s), constant rate while ON.
* Congestion is therefore **real**: it appears when bursts of transfers and UDP ON periods overflow a queue.

## 5. Non-congestive degradation (noise)
* Levels: random loss **0.5 %, 1 %, 2 %**, delay jitter **5, 10, 20, 40 ms** (netem, normal
  distribution if the table is installed, otherwise uniform — recorded in `machine_config.json`), or a
  delay shift of **+40 or +80 ms** one-way (extra propagation delay with no queue).
* In a noisy run, a random 30 % of links get noise **episodes** on both directions, alternating
  ON/OFF every 10–30 s. Noise can coincide with heavy load (the "both at once" case).
* The **actual** time each change took effect is logged (`actions_log.json`) and used as ground truth.

## 6. Labels (`gt_label`)
Using the tbf backlog **averaged over the 50 ms samples in the interval**, as a share of the queue limit:

| Label | Rule |
|---|---|
| `congestion` | backlog > 50 % in this and the 2 previous intervals (wins over noise) |
| `degradation` | noise active for the **whole** interval and backlog < 20 % |
| `normal` | **no** noise in the interval and backlog < 20 % |
| `uncertain` | everything else; `gt_label_reason` says why. Kept, never used for training |

`gt_label_snapshot` applies the same rule to the last queue sample of the interval (closest to the original
"read every 250 ms"), for comparison.

## 7. Columns of `intervals*.csv.gz`
**Never use `meta_*` or `gt_*` columns as model inputs.**

| Group | Columns | Meaning |
|---|---|---|
| meta | `meta_run_id, meta_split, meta_topology, meta_load, meta_noise, meta_transport, meta_event, meta_seed` | run settings |
| meta | `meta_dir_id, meta_link_id, meta_src, meta_dst, meta_src_region, meta_dst_region` | which link direction (regions for RQ2) |
| meta | `meta_k, meta_t_s, meta_warmup` | interval index, start time (s), first 20 s flag |
| static | `static_capacity_mbps, static_prop_delay_ms, static_queue_limit_bytes` | known link attributes |
| f (port stats) | `f_util, f_tx_mbps, f_tx_pps, f_rx_pps_far, f_tx_dropped, f_rx_dropped_far` | from the switch port counters (identical to OpenFlow port statistics; checked by the smoke test) |
| f (probes) | `f_delay_mean_ms, f_delay_min_ms, f_delay_max_ms, f_delay_std_ms, f_delay_p50_ms` | one-way delay of the 64-byte probes sent in the interval (20 per second per direction) |
| f (probes) | `f_probe_sent, f_probe_recv, f_probe_lost, f_probe_loss_ratio, f_probe_reordered` | probe counts |
| f (flow stats) | `f_pathloss_wmean, f_pathloss_max, f_path_pairs_active, f_path_pkts_2s` | loss of the source–destination flows crossing this direction, from OpenFlow flow counters at their first and last switch over the last 2 s (pairs with ≥ 20 packets) |
| gt (queue) | `gt_backlog_mean_bytes, gt_backlog_max_bytes, gt_backlog_min_bytes, gt_backlog_snapshot_bytes, gt_backlog_mean_frac, gt_backlog_snapshot_frac, gt_qlen_mean_pkts, gt_queue_samples` | tbf queue, sampled every 50 ms |
| gt (drops) | `gt_tbf_drops, gt_netem_drops_raw, gt_noise_drops` | queue (congestion) drops; noise drops = netem drops minus child drops when the kernel counts them in netem (measured by the smoke test) |
| gt (noise) | `gt_noise_frac, gt_noise_jitter_ms, gt_noise_loss_pct, gt_link_down_frac` | share of the interval with noise / link down, and the noise level |
| gt (label) | `gt_label, gt_label_reason, gt_label_snapshot` | see section 6 |

Engineered features (`window_features.py`, causal 2 s windows):
`p_delay_ms, p_delay_ema, p_loss_ema, w_delay_level, w_delay_d1, w_delay_d2, w_delay_trend, w_delay_var,
w_jitter_mean, w_queue_delay_est, w_util_mean, w_util_max, w_corr_util_delay, w_loss_mean, w_probe_loss,
w_loss_when_util_high, w_loss_when_util_low, w_corr_util_loss, w_complete` and neighbour averages
`nb_*` of the main window features.

## 8. Other files per run
`schedule.json` (every flow, noise episode, event), `actions_log.json`, `meta.json` (versions, routes,
interface map), `qc.json`, `flows_tcp.csv.gz` (size, CC, start, completion time, status),
`flows_udp.csv.gz`, `flow_seconds.csv.gz` (bytes received per flow per second → Jain's index for RQ3),
`raw/` (all measurements as recorded).

## 9. Splits (by run, never random rows)
Two plans; `compact` (default, 80 runs) is a subset of `full` (199 runs) — same run ids and seeds.

| Split | compact | full adds |
|---|---|---|
| train | NSFNET, loads 30/60/90 × noise {none, loss 0.5, loss 1, jitter 5, jitter 10}, seeds 1–2 | seeds 3–5 |
| val | same settings + loss 2 %, seed 6 | — |
| test | same settings + loss 2 %, seed 7 (**loss 2 % never in training**) | seed 8 |
| test_geant | GÉANT, loads 60/90 × {none, loss 1, loss 2, jitter 10}, seed 7 | seed 8 |
| test_stress | NSFNET, load 90 × {loss 1, jitter 10} × {all-CUBIC, short bursts 0.3–1.5 s at 1.5–3× capacity, one link failure 10–20 s}, seed 7 | load 60, seed 8 |

Harder impairments (`full` plan, NSFNET, loads 30/60/90): jitter 20 ms and delay shift +40 ms in train (seeds 1–2),
validation (seed 6) and test (seed 7); jitter 40 ms and delay shift +80 ms in test only (seed 7).
Resulting run counts: train 87, val 24, test 48, test_geant 16, test_stress 24.

## 10. Quality control (`qc.json`, every run)
Sampling coverage (ports, queues, probes), **unexplained probe loss on clean links** (must be ~0; a busy
machine drops packets outside the queues and would corrupt labels), base delay vs configured, utilisation
range, failed noise/link actions, label counts, realised load. Runs with status FAIL are not merged.

## 11. Design notes
* No MAWI trace replay (size and rewriting risk); traffic is synthetic with heavy-tailed TCP sizes.
* GÉANT is SNDlib's 22-node version, not the Topology Zoo version (40 nodes).
* Link delay is measured with one-way probes between neighbouring hosts (same clock), not LLDP.
* netem's random numbers cannot be seeded; every other random choice is seeded.
* Single shortest path per pair during data collection (no ECMP).
* Labels are per link direction (not per undirected link).

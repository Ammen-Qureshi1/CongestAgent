# CongestAgent dataset generator — runbook

All commands are run from the `datagen/` folder as root (`sudo`). All tools are free and open source.

## 0. Installation
```bash
sudo apt update
sudo apt install -y mininet openvswitch-switch iproute2 ethtool python3-numpy python3-pandas python3-networkx
sudo modprobe openvswitch sch_netem tcp_bbr
sudo systemctl start openvswitch-switch
```
If the `apt` Python packages are not available: `sudo pip3 install numpy pandas networkx --break-system-packages`.

## 1. Smoke test (about 10 min)
```bash
cd datagen
sudo python3 smoke_test.py
```
All checks must pass. The test also writes `machine_config.json`, which the generator reads.

## 2. Pilot run (about 3 min)
```bash
sudo python3 batch.py --pilot
```
Writes one short run to `runs/pilot/` for checking the output format before the full batch.

## 3. Batch
```bash
sudo python3 batch.py --plan --size full                     # shows the plan: 199 runs, about 8.2 h
sudo systemd-inhibit --what=sleep:idle:handle-lid-switch python3 batch.py --size full
```
The batch is resumable: if it is stopped, running the same command again skips finished runs. Runs are ordered by
priority in five rounds:

| Round | Runs | Content |
|---|---|---|
| A | 51 | every core setting once for training, validation and test |
| B | 14 | GÉANT (held-out topology) and stress tests |
| C | 15 | a second training seed |
| D | 89 | remaining runs of the full plan |
| H | 30 | harder impairments (jitter 20/40 ms, delay shift 40/80 ms) |

Without `--size full`, only rounds A–C are run (80 runs, about 3.3 h).

The full plan gives 87 training, 24 validation, 48 test (NSFNET), 16 test (GÉANT) and 24 stress-test runs.

## 4. Build the dataset (about 15 min)
```bash
python3 build_all.py
python3 window_features.py ../dataset/intervals_*.csv.gz
```
Output: `dataset/intervals_<split>.csv.gz` (and `_features` versions), flow files and `dataset/summary.json`.

See `DATASHEET.md` for every column, the labels and the splits.

"""Shared constants and helpers for the CongestAgent dataset generator."""
import os
import json
import gzip
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPO_DIR = ROOT / 'topologies'
RUNS_DIR = Path(os.environ.get('CA_RUNS_DIR', ROOT / 'runs'))

# ---- fixed settings -----------------------
CAPACITY_MBPS = 10.0
QUEUE_LIMIT_BYTES = 100 * 1500          # ~100 full-size packets
TBF_BURST_BYTES = 15000                 # 10 packets of token bucket depth
NETEM_LIMIT_PKTS = 10000                # netem's own packet limit (large: never the bottleneck)
INTERVAL_S = 0.25                       # unit of analysis: one link direction x 250 ms
RUN_S = int(os.environ.get('CA_RUN_S', '120'))   # run length (120 s; override only for code tests)
WARMUP_S = 20                           # first 20 s marked as warm-up
PROBE_HZ = 20                           # one-way delay probes per link direction per second
TC_SAMPLE_S = 0.05                      # queue (backlog) sampling period
FLOWSTAT_S = 1.0                        # OpenFlow flow-counter polling period
PATHLOSS_WINDOW_S = 2.0                 # path loss computed over 2 s
LABEL_HIGH = 0.5                        # congestion: backlog > 50% of queue ...
LABEL_CONSEC = 3                        # ... for 3 consecutive intervals
LABEL_LOW = 0.2                         # normal / degradation: backlog < 20% of queue
NOISE_LINK_SHARE = 0.3                  # share of links that get noise episodes in a noisy run
NOISE_EPISODE_S = (10.0, 30.0)          # noise on/off episode length range

SANDBOX = os.environ.get('CA_SANDBOX') == '1'          # code testing only, never for real data
USERSPACE_OVS = SANDBOX or os.environ.get('CA_USERSPACE_OVS') == '1'

TCP_PORT = 5201
UDP_PORT = 5202
PROBE_PORT = 9100


def load_topology(name):
    return json.loads((TOPO_DIR / f'{name}.json').read_text())


def host_ip(node_id):
    return f'10.0.0.{node_id + 1}'


def stable_seed(*parts):
    h = hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def write_csv_gz(path, header, rows):
    with gzip.open(path, 'wt', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=str))

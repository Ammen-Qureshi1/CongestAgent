"""
Measurement collector, runs in the root namespace during a run.

Three samplers, each storing raw timestamped readings (wall clock, ns):
  ports  (every 250 ms): kernel counters of every switch-to-switch interface
          (tx/rx bytes, packets, dropped). These are the same counters Open vSwitch
          returns as OpenFlow port statistics (checked by the smoke test).
  queues (every 50 ms) : `tc -s -j qdisc show` for netem and tbf on every link
          direction: backlog (bytes, packets), drops, sent bytes. GROUND TRUTH ONLY.
  flows  (every 1 s)   : OpenFlow flow counters (n_packets, n_bytes) of every
          source-destination rule on every switch (`ovs-ofctl dump-flows`).
"""
import re
import json
import time
import threading
import subprocess

from common import INTERVAL_S, TC_SAMPLE_S, FLOWSTAT_S, write_csv_gz

STAT_FILES = ('tx_bytes', 'tx_packets', 'rx_bytes', 'rx_packets', 'tx_dropped', 'rx_dropped')
FLOW_RE = re.compile(r'cookie=0x([0-9a-f]+).*?n_packets=(\d+), n_bytes=(\d+)')


class Collector:
    def __init__(self, interfaces, switch_names, t_start, t_end):
        self.intfs = sorted(set(interfaces))
        self.intf_set = set(self.intfs)
        self.switches = switch_names
        self.t_start, self.t_end = t_start, t_end
        self.port_rows, self.tc_rows, self.flow_rows, self.load_rows = [], [], [], []
        self.errors = []
        self.threads = []

    def _ports(self):
        k = 0
        while True:
            target = self.t_start + k * INTERVAL_S
            if target > self.t_end:
                break
            d = target - time.time()
            if d > 0:
                time.sleep(d)
            ts = time.time_ns()
            for intf in self.intfs:
                try:
                    vals = []
                    for f in STAT_FILES:
                        with open(f'/sys/class/net/{intf}/statistics/{f}') as fh:
                            vals.append(int(fh.read()))
                    self.port_rows.append((ts, intf, *vals))
                except OSError as e:
                    self.errors.append(f'ports {intf}: {e}')
            k += 1

    def _queues(self):
        k = 0
        while True:
            target = self.t_start + k * TC_SAMPLE_S
            if target > self.t_end:
                break
            d = target - time.time()
            if d > 0:
                time.sleep(d)
            a = time.time_ns()
            r = subprocess.run(['tc', '-s', '-j', 'qdisc', 'show'], capture_output=True, text=True)
            b = time.time_ns()
            ts = (a + b) // 2
            try:
                for q in json.loads(r.stdout):
                    dev = q.get('dev')
                    if dev in self.intf_set and q.get('kind') in ('netem', 'tbf'):
                        self.tc_rows.append((ts, dev, q['kind'], q.get('bytes', 0), q.get('packets', 0),
                                             q.get('drops', 0), q.get('overlimits', 0),
                                             q.get('backlog', 0), q.get('qlen', 0), (b - a) // 1000))
            except (ValueError, KeyError) as e:
                self.errors.append(f'tc parse: {e}')
            k += 1
            # if a call overran, skip missed slots instead of bursting
            k = max(k, int((time.time() - self.t_start) / TC_SAMPLE_S))

    def _flows(self):
        k = 0
        while True:
            target = self.t_start + k * FLOWSTAT_S
            if target > self.t_end:
                break
            d = target - time.time()
            if d > 0:
                time.sleep(d)
            procs = {s: subprocess.Popen(['ovs-ofctl', '-O', 'OpenFlow13', 'dump-flows', s],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for s in self.switches}
            for s, p in procs.items():
                out, err = p.communicate()
                ts = time.time_ns()
                if p.returncode != 0:
                    self.errors.append(f'flows {s}: {err.strip()[:100]}')
                    continue
                for m in FLOW_RE.finditer(out):
                    self.flow_rows.append((ts, s, int(m.group(1), 16), int(m.group(2)), int(m.group(3))))
            try:
                with open('/proc/loadavg') as fh:
                    self.load_rows.append((time.time_ns(), *fh.read().split()[:3]))
            except OSError:
                pass
            k += 1

    def start(self):
        for fn in (self._ports, self._queues, self._flows):
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            self.threads.append(t)

    def join(self):
        for t in self.threads:
            t.join()

    def save(self, raw):
        write_csv_gz(raw / 'ports.csv.gz', ('ts_ns', 'intf', *STAT_FILES), self.port_rows)
        write_csv_gz(raw / 'queues.csv.gz', ('ts_ns', 'intf', 'kind', 'bytes', 'packets', 'drops', 'overlimits',
                                             'backlog_bytes', 'qlen_packets', 'call_us'), self.tc_rows)
        write_csv_gz(raw / 'flowstats.csv.gz', ('ts_ns', 'switch', 'cookie', 'n_packets', 'n_bytes'), self.flow_rows)
        write_csv_gz(raw / 'loadavg.csv.gz', ('ts_ns', 'load1', 'load5', 'load15'), self.load_rows)
        (raw / 'collector_errors.txt').write_text('\n'.join(self.errors[:500]))

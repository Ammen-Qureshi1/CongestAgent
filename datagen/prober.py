"""
One-way link-delay prober, run inside every Mininet host.

Host h_i sends a small UDP probe (64 B) to each neighbouring host h_j every
1/PROBE_HZ s. Between neighbours the installed route is the direct link, so a
probe from h_i to h_j crosses exactly one switch-to-switch link direction
(i -> j) plus two unshaped access links. All hosts are network namespaces on
the same kernel and share one clock, so receive_time - send_time is the exact
one-way delay. (A real controller would use LLDP/echo timing; this is the
emulation-side equivalent and is documented as such.)

Logs: probes_tx_<id>.csv (dst, seq, send_ns) and probes_rx_<id>.csv
(src, seq, send_ns, recv_ns).
"""
import sys
import json
import time
import socket
import struct
import threading
import gzip
import csv

FMT = '!HHIq'
PAD = 64 - struct.calcsize(FMT)


def main():
    cfg = json.loads(sys.argv[1])
    me, peers, t0, dur, hz, port, out = (cfg['id'], cfg['peers'], cfg['t0'], cfg['duration'],
                                         cfg['hz'], cfg['port'], cfg['out'])
    rx_rows, tx_rows = [], []
    rs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rs.bind(('0.0.0.0', port))
    rs.settimeout(0.2)
    stop_rx = t0 + dur + 2.0

    def rx():
        while time.time() < stop_rx:
            try:
                data, _ = rs.recvfrom(256)
            except socket.timeout:
                continue
            now = time.time_ns()
            src, dst, seq, sent = struct.unpack(FMT, data[:struct.calcsize(FMT)])
            rx_rows.append((src, seq, sent, now))

    th = threading.Thread(target=rx, daemon=True)
    th.start()
    ss = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / hz
    n = max(1, len(peers))
    seq = {p['id']: 0 for p in peers}
    while time.time() < t0:
        time.sleep(0.005)
    k = 0
    while True:
        base = t0 + k * period
        if base >= t0 + dur:
            break
        for idx, p in enumerate(peers):
            target = base + idx * period / n          # stagger probes to different neighbours
            d = target - time.time()
            if d > 0:
                time.sleep(d)
            s = seq[p['id']]
            now = time.time_ns()
            try:
                ss.sendto(struct.pack(FMT, me, p['id'], s, now) + b'\0' * PAD, (p['ip'], port))
                tx_rows.append((p['id'], s, now))
            except OSError:
                tx_rows.append((p['id'], s, -now))     # negative = send error (e.g. link down)
            seq[p['id']] = s + 1
        k += 1
    th.join()
    for name, header, rows in (('tx', ('dst', 'seq', 'send_ns'), tx_rows),
                               ('rx', ('src', 'seq', 'send_ns', 'recv_ns'), rx_rows)):
        with gzip.open(f"{out}/probes_{name}_{me}.csv.gz", 'wt', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)


if __name__ == '__main__':
    main()

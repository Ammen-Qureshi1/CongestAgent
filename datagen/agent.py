"""
Traffic agent, run inside every Mininet host. One process per host does:

  * TCP receiver (port 5201): accepts any number of connections; counts received
    bytes per flow per second and the time of the first/last byte.
  * UDP receiver (port 5202): counts received bytes/packets per flow per second
    and tracks sequence numbers (UDP loss per flow).
  * TCP senders: each scheduled flow opens its own connection with the requested
    congestion control (setsockopt TCP_CONGESTION = cubic or bbr) and sends its
    size as fast as TCP allows (a finite, elastic transfer).
  * UDP senders: constant-rate packets during scheduled ON segments, paced in 5 ms ticks.

All timestamps are wall-clock (shared by all hosts). Output files are written to
the run's raw folder at the end.
"""
import sys
import json
import time
import socket
import struct
import threading
import gzip
import csv
from collections import defaultdict

TCP_HDR = '!Q'               # flow id
UDP_HDR = '!QIq'             # flow id, seq, send_ns
UDP_PKT = 1200
CHUNK = 64 * 1024
MAX_CONCURRENT_TCP = 80


def main():
    cfg = json.loads(sys.argv[1])
    me, t0, dur, out = cfg['id'], cfg['t0'], cfg['duration'], cfg['out']
    sched = json.loads(open(cfg['schedule']).read())
    stop_send = t0 + dur
    stop_all = t0 + dur + 3.0
    lock = threading.Lock()
    rx_tcp = defaultdict(lambda: defaultdict(int))     # flow -> sec -> bytes
    rx_tcp_span = {}                                   # flow -> [first_ns, last_ns, total]
    rx_udp = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # flow -> sec -> [bytes, pkts]
    udp_seq = defaultdict(lambda: [None, 0, 0])        # flow -> [max_seq, received, reordered]
    tx_tcp_rows, tx_udp = [], defaultdict(lambda: defaultdict(lambda: [0, 0]))
    errors = []

    def sec_of(ts):
        return int(ts - t0) if ts >= t0 else -1

    # ---------------- receivers ----------------
    def tcp_conn(c):
        try:
            c.settimeout(5.0)
            hdr = b''
            while len(hdr) < 8:
                part = c.recv(8 - len(hdr))
                if not part:
                    return
                hdr += part
            fid = struct.unpack(TCP_HDR, hdr)[0]
            while time.time() < stop_all:
                try:
                    data = c.recv(CHUNK)
                except socket.timeout:
                    continue
                if not data:
                    break
                now = time.time()
                with lock:
                    rx_tcp[fid][sec_of(now)] += len(data)
                    span = rx_tcp_span.setdefault(fid, [time.time_ns(), 0, 0])
                    span[1] = time.time_ns()
                    span[2] += len(data)
        except OSError as e:
            errors.append(f'tcp_rx {e}')
        finally:
            c.close()

    def tcp_server():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', cfg['tcp_port']))
        s.listen(256)
        s.settimeout(0.5)
        while time.time() < stop_all:
            try:
                c, _ = s.accept()
            except socket.timeout:
                continue
            threading.Thread(target=tcp_conn, args=(c,), daemon=True).start()
        s.close()

    def udp_server():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        s.bind(('0.0.0.0', cfg['udp_port']))
        s.settimeout(0.2)
        hl = struct.calcsize(UDP_HDR)
        while time.time() < stop_all:
            try:
                data, _ = s.recvfrom(2048)
            except socket.timeout:
                continue
            fid, seq, _sent = struct.unpack(UDP_HDR, data[:hl])
            now = time.time()
            with lock:
                b = rx_udp[fid][sec_of(now)]
                b[0] += len(data); b[1] += 1
                st = udp_seq[fid]
                st[1] += 1
                if st[0] is None or seq > st[0]:
                    st[0] = seq
                else:
                    st[2] += 1
        s.close()

    # ---------------- senders ----------------
    active = [0]

    def tcp_flow(f):
        start_ns = time.time_ns()
        sent, status = 0, 'complete'
        if active[0] >= MAX_CONCURRENT_TCP:
            tx_tcp_rows.append((f['id'], f['dst'], f['cc'], f['size'], f['start'], start_ns, 0, 0, 'skipped_concurrency'))
            return
        with lock:
            active[0] += 1
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, f['cc'].encode())
            s.settimeout(10.0)
            s.connect((f['dst_ip'], cfg['tcp_port']))
            s.sendall(struct.pack(TCP_HDR, f['id']))
            buf = b'\0' * CHUNK
            while sent < f['size']:
                if time.time() >= stop_send:
                    status = 'cut_at_run_end'
                    break
                n = min(CHUNK, f['size'] - sent)
                s.sendall(buf[:n])
                sent += n
        except OSError as e:
            status = f'error:{type(e).__name__}'
        finally:
            try:
                s.close()
            except OSError:
                pass
            with lock:
                active[0] -= 1
        tx_tcp_rows.append((f['id'], f['dst'], f['cc'], f['size'], f['start'], start_ns, time.time_ns(), sent, status))

    def tcp_scheduler():
        for f in sorted(sched['tcp_flows'], key=lambda x: x['start']):
            d = t0 + f['start'] - time.time()
            if d > 0:
                time.sleep(d)
            if time.time() >= stop_send:
                break
            threading.Thread(target=tcp_flow, args=(f,), daemon=True).start()

    def udp_flow(f):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        seq = 0
        pad = b'\0' * (UDP_PKT - struct.calcsize(UDP_HDR))
        for (a, b, rate) in f['segments']:
            start, end = t0 + a, min(t0 + b, stop_send)
            d = start - time.time()
            if d > 0:
                time.sleep(d)
            bytes_per_s = rate * 1e6 / 8.0
            sent_bytes, seg_t0 = 0, time.time()
            while True:
                now = time.time()
                if now >= end:
                    break
                due = (now - seg_t0) * bytes_per_s
                while sent_bytes + UDP_PKT <= due:
                    try:
                        s.sendto(struct.pack(UDP_HDR, f['id'], seq, time.time_ns()) + pad, (f['dst_ip'], cfg['udp_port']))
                        tx_udp[f['id']][sec_of(time.time())][0] += UDP_PKT
                        tx_udp[f['id']][sec_of(time.time())][1] += 1
                    except OSError:
                        pass
                    seq += 1
                    sent_bytes += UDP_PKT
                time.sleep(0.005)
        s.close()

    threads = [threading.Thread(target=tcp_server, daemon=True), threading.Thread(target=udp_server, daemon=True)]
    for t in threads:
        t.start()
    while time.time() < t0:
        time.sleep(0.005)
    senders = [threading.Thread(target=tcp_scheduler, daemon=True)]
    senders += [threading.Thread(target=udp_flow, args=(f,), daemon=True) for f in sched['udp_flows']]
    for t in senders:
        t.start()
    while time.time() < stop_all:
        time.sleep(0.2)
    time.sleep(0.3)
    with lock:
        rows = [(f, s, b) for f, d in rx_tcp.items() for s, b in sorted(d.items())]
        spans = [(f, a, b, c) for f, (a, b, c) in rx_tcp_span.items()]
        urows = [(f, s, v[0], v[1]) for f, d in rx_udp.items() for s, v in sorted(d.items())]
        useq = [(f, v[0], v[1], v[2]) for f, v in udp_seq.items()]
        txu = [(f, s, v[0], v[1]) for f, d in tx_udp.items() for s, v in sorted(d.items())]
        txt = list(tx_tcp_rows)
    files = {
        f'rx_tcp_sec_{me}': (('flow_id', 'sec', 'bytes'), rows),
        f'rx_tcp_span_{me}': (('flow_id', 'first_ns', 'last_ns', 'bytes'), spans),
        f'rx_udp_sec_{me}': (('flow_id', 'sec', 'bytes', 'packets'), urows),
        f'rx_udp_seq_{me}': (('flow_id', 'max_seq', 'received', 'reordered'), useq),
        f'tx_udp_sec_{me}': (('flow_id', 'sec', 'bytes', 'packets'), txu),
        f'tx_tcp_{me}': (('flow_id', 'dst', 'cc', 'size', 'planned_start_s', 'start_ns', 'end_ns', 'bytes_sent', 'status'), txt),
    }
    for name, (header, data) in files.items():
        with gzip.open(f'{out}/{name}.csv.gz', 'wt', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(data)
    with open(f'{out}/agent_errors_{me}.txt', 'w') as fh:
        fh.write('\n'.join(errors[:200]))


if __name__ == '__main__':
    main()

"""
10-minute physical check of the testbed on THIS machine. Run before any data collection:
    sudo python3 smoke_test.py
It must end with RESULT: ALL PASS. It also writes ../machine_config.json, which the
generator reads (netem distribution support and how netem counts child drops).

Checks
  1  tools: tc netem/tbf, normal jitter table, BBR + CUBIC, OVS kernel module, Python libs
  2  NSFNET builds; routes deliver between neighbours and across the network
  3  base one-way delay of every link direction = configured propagation delay
  4  OVERLOAD on one link (15 Mbps UDP into 10 Mbps): link saturates, the tbf queue fills,
     drops happen in tbf; records how netem counts the child's drops
  5  NOISE LOSS (2%, 5 Mbps UDP): extra drops appear in netem only, queue stays low
  6  JITTER (10 ms): probe delay spread rises, queue stays low; noise OFF restores it
  7  port counters read from the kernel = OpenFlow port statistics from OVS
  8  TCP CUBIC and BBR transfers complete inside the emulated hosts
  9  machine idle loss: no probe loss on a clean, lightly loaded network
"""
import os
import sys
import json
import glob
import time
import shutil
import tempfile
import subprocess
import statistics
from pathlib import Path

import pandas as pd

from common import ROOT, SANDBOX, USERSPACE_OVS, QUEUE_LIMIT_BYTES, CAPACITY_MBPS, load_topology, host_ip
from netbuild import Network, sh

HERE = Path(__file__).resolve().parent
results, machine = [], {}


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def tc_stats(dev):
    out = {}
    for q in json.loads(subprocess.run(['tc', '-s', '-j', 'qdisc', 'show', 'dev', dev], capture_output=True, text=True).stdout):
        out[q['kind']] = q
    return out


def probes(net, seconds, tmp):
    for f in glob.glob(f'{tmp}/probes_*'):
        os.remove(f)
    t0 = time.time() + 1.5
    ps = []
    topo = net.topo
    for i, h in net.hosts.items():
        nb = sorted({l['b'] for l in topo['links'] if l['a'] == i} | {l['a'] for l in topo['links'] if l['b'] == i})
        cfg = {'id': i, 'peers': [{'id': j, 'ip': host_ip(j)} for j in nb], 't0': t0, 'duration': seconds,
               'hz': 20, 'port': 9100, 'out': tmp}
        ps.append(h.popen(['python3', str(HERE / 'prober.py'), json.dumps(cfg)]))
    for p in ps:
        p.wait(timeout=seconds + 20)
    tx = pd.concat([pd.read_csv(f).assign(src=int(Path(f).name.split('_')[-1].split('.')[0])) for f in glob.glob(f'{tmp}/probes_tx_*')])
    rx = pd.concat([pd.read_csv(f).assign(dst=int(Path(f).name.split('_')[-1].split('.')[0])) for f in glob.glob(f'{tmp}/probes_rx_*')])
    m = tx.merge(rx[['src', 'dst', 'seq', 'recv_ns']], on=['src', 'dst', 'seq'], how='left')
    m['delay'] = (m.recv_ns - m.send_ns) / 1e6
    return m


def udp_flow(net, a, b, mbps, seconds, tmp, tcp=None):
    """Send one UDP flow (and optional TCP flows) with the real traffic agent; return receiver stats."""
    t0 = time.time() + 1.5
    sched = {'tcp_flows': tcp or [], 'udp_flows': ([{'id': 1, 'dst': b, 'dst_ip': host_ip(b),
                                                     'segments': [[0, seconds, mbps]]}] if mbps else [])}
    procs = []
    for i, h in net.hosts.items():
        sp = Path(tmp) / f'sched_{i}.json'
        sp.write_text(json.dumps(sched if i == a else {'tcp_flows': [], 'udp_flows': []}))
        cfg = {'id': i, 't0': t0, 'duration': seconds, 'out': tmp, 'schedule': str(sp), 'tcp_port': 5201, 'udp_port': 5202}
        if i in (a, b):
            procs.append(h.popen(['python3', str(HERE / 'agent.py'), json.dumps(cfg)]))
    for p in procs:
        p.wait(timeout=seconds + 30)
    return tmp


def main():
    print(f'mode: {"SANDBOX (code test only)" if SANDBOX else "normal"}; OVS datapath: '
          f'{"userspace" if USERSPACE_OVS else "kernel"}', flush=True)
    # ---------------- 1 tools ----------------
    r = sh('tc qdisc add dev lo root netem delay 1ms && tc qdisc del dev lo root', check=False)
    netem_ok = r.returncode == 0
    if SANDBOX:
        print('SKIP  netem (sandbox)')
    else:
        check('netem available', netem_ok, 'yes' if netem_ok else f'NO: {r.stderr.strip()} -> sudo modprobe sch_netem')
    r = sh('tc qdisc add dev lo root tbf rate 10mbit burst 15000 limit 150000 && tc qdisc del dev lo root', check=False)
    check('tbf available', r.returncode == 0, 'yes' if r.returncode == 0 else r.stderr.strip())
    # test the table by USING it (its folder differs between distributions, e.g. Ubuntu multiarch paths)
    r = sh('tc qdisc add dev lo root netem delay 10ms 5ms distribution normal && tc qdisc del dev lo root', check=False)
    dist = r.returncode == 0
    machine['netem_normal_distribution'] = dist
    if SANDBOX:
        print('SKIP  netem normal jitter table (sandbox)')
    else:
        check("netem 'normal' jitter distribution usable", dist, 'yes' if dist else f'NO: {r.stderr.strip()}')
    cc = open('/proc/sys/net/ipv4/tcp_available_congestion_control').read().split()
    if 'bbr' not in cc:
        sh('modprobe tcp_bbr', check=False)
        cc = open('/proc/sys/net/ipv4/tcp_available_congestion_control').read().split()
    check('TCP BBR and CUBIC available', 'bbr' in cc and 'cubic' in cc, ' '.join(cc) + ('' if 'bbr' in cc else ' -> sudo modprobe tcp_bbr'))
    if not USERSPACE_OVS:
        r = sh('lsmod | grep -q "^openvswitch"', check=False)
        check('OVS kernel module loaded', r.returncode == 0, 'yes' if r.returncode == 0 else
              'NO -> sudo modprobe openvswitch (or run everything with CA_USERSPACE_OVS=1)')
    for mod in ('numpy', 'pandas', 'networkx', 'mininet'):
        try:
            __import__(mod)
            check(f'python module {mod}', True, 'ok')
        except ImportError:
            check(f'python module {mod}', False, f'missing -> pip install {mod}')
    if not all(results):
        return finish()

    tmp = tempfile.mkdtemp(prefix='ca_smoke_')
    topo = load_topology('nsfnet')
    net = Network(topo, normal_dist=dist)
    try:
        net.build()
        # ---------------- 2 routing ----------------
        bad = []
        pairs = [(l['a'], l['b']) for l in topo['links']] + [(0, 13), (1, 9), (5, 12), (12, 3), (7, 10)]
        for a, b in pairs:
            out = net.hosts[a].cmd(f'ping -c 2 -i 0.2 -W 1 {host_ip(b)}')
            if ' 0% packet loss' not in out:
                bad.append((a, b))
        check('routes deliver (all links + 5 multi-hop pairs)', not bad, f'{len(pairs) - len(bad)}/{len(pairs)} ok {bad}')
        # ---------------- 3 base delay + 9 idle loss ----------------
        m = probes(net, 6, tmp)
        got = m.groupby(['src', 'dst']).delay.min()
        exp = {}
        for l in topo['links']:
            exp[(l['a'], l['b'])] = exp[(l['b'], l['a'])] = 0.0 if SANDBOX else l['delay_ms']
        diff = pd.Series({k: got.get(k, float('nan')) - v for k, v in exp.items()})
        check('base one-way delay = configured (all 42 directions)', diff.between(-0.2, 1.5).all(),
              f'min extra {diff.min():.2f} ms, max extra {diff.max():.2f} ms')
        idle_loss = m.recv_ns.isna().mean()
        check('no probe loss on an idle network', idle_loss < 0.002, f'{idle_loss:.3%} of {len(m)} probes lost')
        # ---------------- 4 overload ----------------
        d0 = net.dirs[0]
        dev, a, b = d0['egress_intf'], d0['src'], d0['dst']
        s0 = tc_stats(dev)
        tx0 = int(open(f'/sys/class/net/{dev}/statistics/tx_bytes').read())
        t = time.time()
        backlog = []
        p = subprocess.Popen(['sleep', '0'])
        import threading
        stop = [False]

        def sample():
            while not stop[0]:
                backlog.append(tc_stats(dev).get('tbf', {}).get('backlog', 0))
                time.sleep(0.05)
        th = threading.Thread(target=sample, daemon=True)
        th.start()
        udp_flow(net, a, b, 15.0, 6, tmp)
        stop[0] = True
        th.join()
        el = time.time() - t
        s1 = tc_stats(dev)
        tx1 = int(open(f'/sys/class/net/{dev}/statistics/tx_bytes').read())
        tbf_d = s1['tbf']['drops'] - s0['tbf']['drops']
        util = (tx1 - tx0) * 8 / (6 * CAPACITY_MBPS * 1e6)
        bmax = max(backlog) if backlog else 0
        check('overload: link saturates', 0.9 <= util <= 1.06, f'utilisation over the 6 s flow {util:.2f}')
        check('overload: tbf queue fills', bmax >= 0.8 * QUEUE_LIMIT_BYTES, f'max backlog {bmax} B of {QUEUE_LIMIT_BYTES} B')
        check('overload: drops happen in tbf', tbf_d > 0, f'tbf drops {tbf_d}')
        if not SANDBOX:
            net_d = s1['netem']['drops'] - s0['netem']['drops']
            machine['netem_counts_child_drops'] = bool(abs(net_d - tbf_d) <= max(3, 0.05 * tbf_d))
            check('overload: netem drop counter is consistent', net_d in (0,) or machine['netem_counts_child_drops'],
                  f'netem drops {net_d} vs tbf drops {tbf_d} -> netem counts child drops: {machine["netem_counts_child_drops"]}')
        time.sleep(2)
        # ---------------- 5 noise loss ----------------
        if not SANDBOX:
            net.set_noise(d0['link_id'], 0.0, 2.0)
            s0 = tc_stats(dev)
            udp_flow(net, a, b, 5.0, 8, tmp)
            s1 = tc_stats(dev)
            net.set_noise(d0['link_id'], 0.0, 0.0)
            sent = s1['netem']['packets'] - s0['netem']['packets']
            tbf_d = s1['tbf']['drops'] - s0['tbf']['drops']
            net_d = s1['netem']['drops'] - s0['netem']['drops']
            noise_d = net_d - (tbf_d if machine.get('netem_counts_child_drops', True) else 0)
            ratio = noise_d / max(1, sent + noise_d)
            check('noise loss 2%: drops are netem (noise) drops', 0.012 <= ratio <= 0.03 and tbf_d == 0,
                  f'noise drop ratio {ratio:.2%}, tbf (queue) drops {tbf_d}')
            # ---------------- 6 jitter ----------------
            net.set_noise(d0['link_id'], 10.0, 0.0)
            mj = probes(net, 5, tmp)
            net.set_noise(d0['link_id'], 0.0, 0.0)
            mo = probes(net, 4, tmp)
            sj = mj[(mj.src == a) & (mj.dst == b)].delay.std()
            so = mo[(mo.src == a) & (mo.dst == b)].delay.std()
            # netem delay 5 ms, jitter 10 ms (normal, SD 10 ms) -> delays below 0 are cut at 0, so the
            # measured SD is lower than 10 ms; >= 5 ms shows the jitter is really applied
            check('jitter 10 ms visible in probes', sj >= 5.0, f'one-way delay SD {sj:.2f} ms with jitter (normal distribution)')
            check('noise OFF restores the link', so < 1.0, f'one-way delay SD {so:.2f} ms after noise off')
            st = tc_stats(dev)
            check('tbf queue still attached after noise changes', 'tbf' in st and 'netem' in st,
                  f'qdiscs on {dev}: {sorted(st)}')
        # ---------------- 7 counters = OpenFlow port stats ----------------
        port = net.iface[(a, b)]['port']
        out = subprocess.run(['ovs-ofctl', '-O', 'OpenFlow13', 'dump-ports', net.switches[a].name, str(port)],
                             capture_output=True, text=True).stdout
        sysfs = int(open(f'/sys/class/net/{dev}/statistics/tx_bytes').read())
        import re
        mm = re.search(r'tx pkts=\d+, bytes=(\d+)', out)
        of = int(mm.group(1)) if mm else -1
        check('kernel counters = OpenFlow port stats', of >= 0 and abs(of - sysfs) <= max(2000, 0.01 * sysfs),
              f'OpenFlow tx_bytes {of}, kernel tx_bytes {sysfs}')
        # ---------------- 8 TCP CUBIC + BBR ----------------
        far = 13 if a != 13 else 12
        flows = [{'id': 11, 'dst': far, 'dst_ip': host_ip(far), 'start': 0.0, 'size': 1_000_000, 'cc': 'cubic'},
                 {'id': 12, 'dst': far, 'dst_ip': host_ip(far), 'start': 0.2, 'size': 1_000_000, 'cc': 'bbr'}]
        for f in glob.glob(f'{tmp}/tx_tcp_*') + glob.glob(f'{tmp}/rx_tcp_span_*'):
            os.remove(f)
        t0 = time.time() + 1.5
        procs = []
        for i in (a, far):
            sp = Path(tmp) / f'sched_{i}.json'
            sp.write_text(json.dumps({'tcp_flows': flows if i == a else [], 'udp_flows': []}))
            cfg = {'id': i, 't0': t0, 'duration': 8, 'out': tmp, 'schedule': str(sp), 'tcp_port': 5201, 'udp_port': 5202}
            procs.append(net.hosts[i].popen(['python3', str(HERE / 'agent.py'), json.dumps(cfg)]))
        for p in procs:
            p.wait(timeout=40)
        txt = pd.read_csv(f'{tmp}/tx_tcp_{a}.csv.gz')
        span = pd.read_csv(f'{tmp}/rx_tcp_span_{far}.csv.gz')
        ok = set(txt[txt.status == 'complete'].cc) == {'cubic', 'bbr'} and (span.bytes >= 1_000_000).sum() == 2
        check('TCP CUBIC and BBR transfers complete', ok, f'sender status {dict(zip(txt.cc, txt.status))}, '
              f'received {span.bytes.tolist()}')
    finally:
        net.stop()
        shutil.rmtree(tmp, ignore_errors=True)
    return finish()


def finish():
    machine.update({'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'userspace_ovs': USERSPACE_OVS,
                    'sandbox': SANDBOX, 'all_pass': all(results)})
    if not SANDBOX:
        (ROOT / 'machine_config.json').write_text(json.dumps(machine, indent=1))
    print('\nRESULT:', 'ALL PASS' if all(results) else 'SOME CHECKS FAILED - send this whole output')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()

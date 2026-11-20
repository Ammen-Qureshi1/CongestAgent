"""
Run ONE dataset run end to end:
  build network -> write schedules -> start probers + traffic agents in every host
  -> collector + noise/event scheduler -> wait -> stop -> save raw files.
Building the per-interval dataset is done afterwards by build_intervals.py.

Usage (as root):
  python3 run_one.py nsfnet 60 loss1 mixed none 1 [--split train] [--build]
"""
import sys
import json
import time
import shutil
import platform
import threading
import subprocess
from pathlib import Path

from common import (RUNS_DIR, RUN_S, PROBE_HZ, PROBE_PORT, TCP_PORT, UDP_PORT, SANDBOX, USERSPACE_OVS,
                    load_topology, host_ip, write_json)
from netbuild import Network
from scenario import make_schedule, run_id

HERE = Path(__file__).resolve().parent
START_DELAY_S = 4.0       # time for all host processes to start before t0


def versions():
    def out(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else ''
    return {'kernel': platform.release(), 'python': platform.python_version(),
            'ovs': out('ovs-vsctl --version'), 'mininet': out('mn --version'), 'iproute2': out('tc -V'),
            'cc_available': out('cat /proc/sys/net/ipv4/tcp_available_congestion_control'),
            'cpu': out("grep -m1 'model name' /proc/cpuinfo"), 'nproc': out('nproc')}


def machine_cfg():
    p = HERE.parent / 'machine_config.json'
    return json.loads(p.read_text()) if p.exists() else {}


def run(cfg, split, build=False, runs_dir=RUNS_DIR):
    rid = run_id(cfg)
    out = Path(runs_dir) / rid
    if (out / 'DONE').exists():
        print(f'{rid}: already done')
        return out
    if out.exists():
        shutil.rmtree(out)          # an interrupted, incomplete run is redone from scratch
    raw = out / 'raw'
    raw.mkdir(parents=True)
    mc = machine_cfg()
    if not SANDBOX and not mc.get('all_pass'):
        raise SystemExit('Refusing to run: smoke_test.py has not passed on this machine '
                         '(run: sudo python3 smoke_test.py, it must end with ALL PASS).')
    topo = load_topology(cfg['topology'])
    net = Network(topo, normal_dist=mc.get('netem_normal_distribution', True))
    t_build = time.time()
    procs, log = [], []
    try:
        net.build()
        desc = net.describe()
        sched = make_schedule(cfg, topo, net.paths, net.dirs)
        write_json(out / 'schedule.json', sched)
        nb = {i: sorted({l['b'] for l in topo['links'] if l['a'] == i} | {l['a'] for l in topo['links'] if l['b'] == i})
              for i in net.hosts}
        t0 = time.time() + START_DELAY_S
        for i, h in net.hosts.items():
            hs = {'tcp_flows': [f for f in sched['tcp_flows'] if f['src'] == i],
                  'udp_flows': [f for f in sched['udp_flows'] if f['src'] == i]}
            sp = raw / f'sched_{i}.json'
            sp.write_text(json.dumps(hs))
            pcfg = {'id': i, 'peers': [{'id': j, 'ip': host_ip(j)} for j in nb[i]], 't0': t0, 'duration': RUN_S,
                    'hz': PROBE_HZ, 'port': PROBE_PORT, 'out': str(raw)}
            acfg = {'id': i, 't0': t0, 'duration': RUN_S, 'out': str(raw), 'schedule': str(sp),
                    'tcp_port': TCP_PORT, 'udp_port': UDP_PORT}
            procs.append(h.popen(['python3', str(HERE / 'prober.py'), json.dumps(pcfg)],
                                 stdout=subprocess.DEVNULL, stderr=open(raw / f'prober_err_{i}.txt', 'w')))
            procs.append(h.popen(['python3', str(HERE / 'agent.py'), json.dumps(acfg)],
                                 stdout=subprocess.DEVNULL, stderr=open(raw / f'agent_err_{i}.txt', 'w')))
        from collector import Collector
        intfs = [d['egress_intf'] for d in net.dirs] + [d['ingress_intf'] for d in net.dirs]
        col = Collector(intfs, [net.switches[i].name for i in sorted(net.switches)], t0 - 1.0, t0 + RUN_S + 1.0)
        col.start()

        # ---- noise / event scheduler (actual execution times are the ground truth) ----
        actions = []
        for e in sched['noise']:
            actions.append((e['start'], 'noise_on', e))
            actions.append((e['end'], 'noise_off', e))
        for e in sched['events']:
            if e['type'] == 'link_down':
                actions.append((e['start'], 'link_down', e))
                actions.append((e['end'], 'link_up', e))
        actions.sort(key=lambda a: a[0])

        def scheduler():
            for at, kind, e in actions:
                if at > RUN_S + 0.5:
                    continue
                d = t0 + at - time.time()
                if d > 0:
                    time.sleep(d)
                a = time.time_ns()
                if kind == 'noise_on':
                    rc = net.set_noise(e['link_id'], e['jitter_ms'], e['loss_pct'], e.get('extra_delay_ms', 0.0))
                elif kind == 'noise_off':
                    rc = net.set_noise(e['link_id'], 0.0, 0.0)
                elif kind == 'link_down':
                    rc = net.set_link_state(e['link_id'], False)
                else:
                    rc = net.set_link_state(e['link_id'], True)
                b = time.time_ns()
                # the change takes effect between a and b; b is used as the switch time
                log.append({'kind': kind, 'link_id': e['link_id'], 'planned_s': at, 'start_ns': a, 'ts_ns': b,
                            'rc': rc, 'jitter_ms': e.get('jitter_ms', 0.0), 'loss_pct': e.get('loss_pct', 0.0),
                            'extra_delay_ms': e.get('extra_delay_ms', 0.0)})

        st = threading.Thread(target=scheduler, daemon=True)
        st.start()
        print(f'{rid}: running {RUN_S} s ({len(sched["tcp_flows"])} TCP flows, {len(sched["udp_flows"])} UDP flows, '
              f'{len(sched["noise"])} noise episodes)', flush=True)
        time.sleep(max(0, t0 + RUN_S + 4.0 - time.time()))
        st.join(timeout=5)
        col.join()
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        col.save(raw)
        write_json(out / 'actions_log.json', log)
        write_json(out / 'meta.json', {
            'run_id': rid, 'config': cfg, 'split': split, 't0_ns': int(t0 * 1e9), 'run_s': RUN_S,
            'sandbox': SANDBOX, 'userspace_ovs': USERSPACE_OVS, 'machine_config': mc,
            'versions': versions(), 'network': desc, 'build_and_run_wall_s': round(time.time() - t_build, 1),
            'topology_name': topo['name'], 'topology_source': topo['source']})
        (out / 'DONE').write_text('ok')
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        net.stop()
    if build:
        from build_intervals import build_run
        build_run(out)
    return out


if __name__ == '__main__':
    a = sys.argv[1:]
    cfg = {'topology': a[0], 'load': int(a[1]), 'noise': a[2], 'transport': a[3], 'event': a[4], 'seed': int(a[5])}
    split = a[a.index('--split') + 1] if '--split' in a else 'adhoc'
    run(cfg, split, build='--build' in a)

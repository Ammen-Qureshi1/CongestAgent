"""
Turn one run configuration into exact, reproducible schedules.

Configuration factors:
  topology   : nsfnet | geant
  load       : 30 | 60 | 90  (% of capacity on the busiest link, time-average)
  noise      : none | loss0.5 | loss1 | loss2 | jitter5 | jitter10 | jitter20 | jitter40 | shift40 | shift80
  transport  : mixed (half CUBIC, half BBR TCP flows) | cubic
  event      : none | bursts | linkfail
  seed       : integer

Traffic model (offered load is controlled and documented):
  * K random source-destination host pairs, weights ~ lognormal(0, 0.8);
  * each pair's mean rate is scaled so that the busiest link direction (under the
    fixed shortest-path routes) carries load% of 10 Mbps on average;
  * half of each pair's rate is TCP: finite transfers arriving as a Poisson process,
    sizes lognormal (mean ~400 KB, sigma 1.2, cut to 10 KB..8 MB), CUBIC or BBR per flow;
  * half is UDP: ON/OFF (exponential periods, mean ON 4 s), constant rate while ON.
  Bursts of TCP transfers and UDP ON periods create real, short-lived queue build-up;
  how often the queue overflows depends on the load level.

Noise (non-congestive degradation): a random 30% of links get noise episodes
(both directions) alternating ON/OFF, each 10-30 s, at the run's noise level.
"""
import math
import numpy as np

from common import (CAPACITY_MBPS, RUN_S, NOISE_LINK_SHARE, NOISE_EPISODE_S, stable_seed, host_ip)

NOISE_LEVELS = {'none': (0.0, 0.0, 0.0), 'loss0.5': (0.0, 0.5, 0.0), 'loss1': (0.0, 1.0, 0.0), 'loss2': (0.0, 2.0, 0.0),
                'jitter5': (5.0, 0.0, 0.0), 'jitter10': (10.0, 0.0, 0.0),
                # harder noise: heavy jitter and delay shifts (extra one-way delay, no queue)
                'jitter20': (20.0, 0.0, 0.0), 'jitter40': (40.0, 0.0, 0.0), 'shift40': (0.0, 0.0, 40.0), 'shift80': (0.0, 0.0, 80.0)}
PAIRS = {'nsfnet': 30, 'geant': 45}
TCP_SHARE = 0.5
SIZE_MEAN_B, SIZE_SIGMA = 400e3, 1.2
SIZE_MIN_B, SIZE_MAX_B = 10e3, 8e6
UDP_ON_MEAN_S = 4.0
UDP_MAX_ON_RATE = 0.9 * CAPACITY_MBPS


def run_id(c):
    return f"{c['topology']}_L{c['load']}_{c['noise']}_{c['transport']}_{c['event']}_s{c['seed']}"


def _size_sampler(rng):
    mu = math.log(SIZE_MEAN_B) - SIZE_SIGMA ** 2 / 2
    return lambda n: np.clip(rng.lognormal(mu, SIZE_SIGMA, n), SIZE_MIN_B, SIZE_MAX_B)


LOAD_TOLERANCE = 0.05      # realised busiest-link load must be within +-5% (relative) of the target
MAX_DRAWS = 300


def make_schedule(cfg, topo, paths, dirs):
    """Draw the schedule; redraw (new sub-seed) until the realised background load on the busiest
    link direction is within +-5% of the target. Heavy-tailed transfer sizes otherwise make a run's
    real load drift (pilot: 78% for a 90% target). Draw number and realised load are recorded."""
    best = None
    target = cfg['load'] / 100.0
    for draw in range(MAX_DRAWS):
        s = _draw(cfg, topo, paths, dirs, draw)
        err = abs(s['realised_max_background_util'] - target) / target
        if best is None or err < best[0]:
            best = (err, s)
        if err <= LOAD_TOLERANCE:
            break
    s = best[1]
    s['load_relative_error'] = round(best[0], 4)
    return s


def _draw(cfg, topo, paths, dirs, draw):
    rng = np.random.default_rng(stable_seed('ca-v4', run_id(cfg), draw))
    nodes = [n['id'] for n in topo['nodes']]
    dir_index = {(d['src'], d['dst']): d['dir_id'] for d in dirs}

    # ---- traffic matrix and scaling to the target load ----
    all_pairs = [(a, b) for a in nodes for b in nodes if a != b]
    k = PAIRS[cfg['topology']]
    idx = rng.choice(len(all_pairs), size=k, replace=False)
    pairs = [all_pairs[i] for i in idx]
    w = rng.lognormal(0.0, 0.8, size=k)
    dir_load = np.zeros(len(dirs))
    for (a, b), wi in zip(pairs, w):
        p = paths[(a, b)]
        for u, v in zip(p[:-1], p[1:]):
            dir_load[dir_index[(u, v)]] += wi
    scale = (cfg['load'] / 100.0) * CAPACITY_MBPS / dir_load.max()
    rates = w * scale                                     # Mbps, mean offered per pair

    sizes = _size_sampler(rng)
    mean_size = float(sizes(200000).mean())               # mean of the cut distribution
    tcp, udp, fid = [], [], 1
    for (a, b), r in zip(pairs, rates):
        # TCP: Poisson arrivals of finite transfers
        lam = (TCP_SHARE * r * 1e6 / 8.0) / mean_size     # flows per second
        t = rng.exponential(1.0 / lam)
        while t < RUN_S - 1.0:
            cc = 'cubic' if cfg['transport'] == 'cubic' else ('bbr' if rng.random() < 0.5 else 'cubic')
            tcp.append({'id': fid, 'src': a, 'dst': b, 'dst_ip': host_ip(b), 'start': round(float(t), 4),
                        'size': int(sizes(1)[0]), 'cc': cc})
            fid += 1
            t += rng.exponential(1.0 / lam)
        # UDP: ON/OFF with the right mean rate
        mean_udp = (1 - TCP_SHARE) * r
        on_rate = min(2.0 * mean_udp, UDP_MAX_ON_RATE)
        duty = min(1.0, mean_udp / on_rate)
        off_mean = UDP_ON_MEAN_S * (1 - duty) / duty if duty < 1 else 0.0
        segs, t = [], float(rng.uniform(0, UDP_ON_MEAN_S + off_mean))
        while t < RUN_S:
            on = rng.exponential(UDP_ON_MEAN_S)
            segs.append([round(t, 3), round(min(RUN_S, t + on), 3), round(float(on_rate), 4)])
            t += on + (rng.exponential(off_mean) if off_mean > 0 else 0.0)
        udp.append({'id': fid, 'src': a, 'dst': b, 'dst_ip': host_ip(b), 'segments': segs, 'kind': 'background'})
        fid += 1

    # ---- noise episodes (non-congestive degradation) ----
    jitter, loss, extra = NOISE_LEVELS[cfg['noise']]
    noise = []
    if cfg['noise'] != 'none':
        n_noisy = max(1, round(NOISE_LINK_SHARE * len(topo['links'])))
        for lid in sorted(rng.choice(len(topo['links']), size=n_noisy, replace=False).tolist()):
            t, on = 5.0, bool(rng.random() < 0.5)
            while t < RUN_S:
                d = float(rng.uniform(*NOISE_EPISODE_S))
                if on:
                    noise.append({'link_id': lid, 'start': round(t, 3), 'end': round(min(RUN_S + 1, t + d), 3),
                                  'jitter_ms': jitter, 'loss_pct': loss, 'extra_delay_ms': extra})
                t += d
                on = not on

    # ---- events ----
    events = []
    if cfg['event'] == 'bursts':
        for _ in range(10):
            a, b = all_pairs[int(rng.integers(len(all_pairs)))]
            st = float(rng.uniform(25, RUN_S - 5))
            du = float(rng.uniform(0.3, 1.5))
            rate = float(rng.uniform(1.5, 3.0)) * CAPACITY_MBPS
            udp.append({'id': fid, 'src': a, 'dst': b, 'dst_ip': host_ip(b),
                        'segments': [[round(st, 3), round(st + du, 3), round(rate, 3)]], 'kind': 'burst'})
            events.append({'type': 'burst', 'flow_id': fid, 'src': a, 'dst': b, 'start': round(st, 3),
                           'end': round(st + du, 3), 'rate_mbps': round(rate, 3)})
            fid += 1
    elif cfg['event'] == 'linkfail':
        lid = int(rng.integers(len(topo['links'])))
        st = float(rng.uniform(40, 60))
        events.append({'type': 'link_down', 'link_id': lid, 'start': round(st, 3),
                       'end': round(st + float(rng.uniform(10, 20)), 3)})

    # realised offered load per direction (mean over the run): background only, and with bursts
    def offered(include_bursts):
        o = np.zeros(len(dirs))
        for f in tcp:
            p = paths[(f['src'], f['dst'])]
            for u, v in zip(p[:-1], p[1:]):
                o[dir_index[(u, v)]] += f['size'] * 8 / 1e6 / RUN_S
        for f in udp:
            if f['kind'] == 'burst' and not include_bursts:
                continue
            p = paths[(f['src'], f['dst'])]
            mb = sum((b - a) * r for a, b, r in f['segments'])
            for u, v in zip(p[:-1], p[1:]):
                o[dir_index[(u, v)]] += mb / RUN_S
        return o
    bg, offered = offered(False), offered(True)
    return {'config': cfg, 'run_id': run_id(cfg), 'draw': draw, 'pairs': [list(p) for p in pairs],
            'pair_rate_mbps': [round(float(x), 4) for x in rates],
            'mean_tcp_size_bytes': round(mean_size, 1),
            'tcp_flows': tcp, 'udp_flows': udp, 'noise': noise, 'events': events,
            'offered_load_per_dir_util': [round(float(x) / CAPACITY_MBPS, 4) for x in offered],
            'target_max_util': cfg['load'] / 100.0,
            'realised_max_offered_util': round(float(offered.max()) / CAPACITY_MBPS, 4),
            'realised_max_background_util': round(float(bg.max()) / CAPACITY_MBPS, 4)}

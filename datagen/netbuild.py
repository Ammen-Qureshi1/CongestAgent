"""
Build the emulated network for one run.

* one OVS switch and one host per topology node; static ARP;
* switches in fail-secure mode with NO controller: forwarding is fixed
  shortest-path routing installed as OpenFlow rules (one rule per
  source-destination host pair per switch, each with its own packet counters);
* every switch-to-switch interface (each link direction) gets
      netem (root, handle 1:)  -> propagation delay (+ noise when active)
      tbf   (child, handle 10:) -> 10 Mbps bottleneck queue, limit ~100 packets
  so the tbf backlog reflects queueing only (by design);
* segmentation offloads (TSO/GSO/GRO) are disabled so every packet in a queue is a
  real, MTU-sized packet and byte/packet counters are exact.
"""
import subprocess
import itertools
import networkx as nx
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import Link
from mininet.log import setLogLevel

from common import (CAPACITY_MBPS, QUEUE_LIMIT_BYTES, TBF_BURST_BYTES, NETEM_LIMIT_PKTS,
                    SANDBOX, USERSPACE_OVS, host_ip)

setLogLevel('warning')


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f'command failed ({r.returncode}): {cmd}\n{r.stderr.strip()}')
    return r


def graph_of(topo):
    g = nx.Graph()
    g.add_nodes_from(n['id'] for n in topo['nodes'])
    for l in topo['links']:
        g.add_edge(l['a'], l['b'], link=l['id'])
    return g


def shortest_paths(topo):
    """Deterministic single shortest path (hop count) per ordered pair:
    among all shortest paths, the lexicographically smallest node sequence."""
    g = graph_of(topo)
    paths = {}
    for a, b in itertools.permutations(g.nodes, 2):
        paths[(a, b)] = list(min(tuple(p) for p in nx.all_shortest_paths(g, a, b)))
    return paths


def direction_table(topo):
    """Link directions: dir_id = 2*link_id (a->b) or 2*link_id+1 (b->a)."""
    dirs = []
    for l in topo['links']:
        dirs.append({'dir_id': 2 * l['id'], 'link_id': l['id'], 'src': l['a'], 'dst': l['b']})
        dirs.append({'dir_id': 2 * l['id'] + 1, 'link_id': l['id'], 'src': l['b'], 'dst': l['a']})
    return dirs


def netem_args(delay_ms, jitter_ms=0.0, loss_pct=0.0, normal_dist=True):
    a = f'delay {delay_ms}ms'
    if jitter_ms > 0:
        a += f' {jitter_ms}ms' + (' distribution normal' if normal_dist else '')
    if loss_pct > 0:          # omitted when 0: tc then sets no loss (avoids 'loss 0%' parsing differences)
        a += f' loss {loss_pct}%'
    return a + f' limit {NETEM_LIMIT_PKTS}'


class Network:
    def __init__(self, topo, normal_dist=True):
        self.topo = topo
        self.normal_dist = normal_dist
        self.paths = shortest_paths(topo)
        self.dirs = direction_table(topo)
        self.link_by_id = {l['id']: l for l in topo['links']}
        self.net = None

    def build(self):
        sh('mn -c', check=False)
        net = Mininet(controller=None, switch=OVSSwitch, link=Link, autoSetMacs=True,
                      autoStaticArp=True, build=False)
        self.hosts, self.switches = {}, {}
        sw_kw = {'failMode': 'secure', 'protocols': 'OpenFlow13'}
        if USERSPACE_OVS:
            sw_kw['datapath'] = 'user'
        for n in self.topo['nodes']:
            i = n['id']
            self.switches[i] = net.addSwitch(f's{i + 1}', dpid=f'{i + 1:016x}', **sw_kw)
            self.hosts[i] = net.addHost(f'h{i + 1}', ip=host_ip(i) + '/24')
        self.host_port = {}
        for i in self.hosts:
            lk = net.addLink(self.hosts[i], self.switches[i])
            self.host_port[i] = self.switches[i].ports[lk.intf2]
        self.iface = {}   # (node, neighbour) -> {'intf': name, 'port': n}
        for l in self.topo['links']:
            a, b = l['a'], l['b']
            lk = net.addLink(self.switches[a], self.switches[b])
            self.iface[(a, b)] = {'intf': lk.intf1.name, 'port': self.switches[a].ports[lk.intf1]}
            self.iface[(b, a)] = {'intf': lk.intf2.name, 'port': self.switches[b].ports[lk.intf2]}
        net.build()
        net.start()
        self.net = net
        for d in self.dirs:
            d['egress_intf'] = self.iface[(d['src'], d['dst'])]['intf']      # queue of this direction
            d['ingress_intf'] = self.iface[(d['dst'], d['src'])]['intf']     # far end receives here
        self._offloads()
        self._qdiscs()
        self._flows()
        return self

    def _offloads(self):
        feats = 'tso off gso off gro off' + (' tx off rx off sg off' if SANDBOX else '')
        for s in self.switches.values():
            for x in s.intfList():
                if x.name != 'lo':
                    sh(f'ethtool -K {x.name} {feats}', check=False)
        for h in self.hosts.values():
            h.cmd(f'ethtool -K {h.defaultIntf().name} {feats}')

    def _qdiscs(self):
        rate = f'{CAPACITY_MBPS:g}mbit'
        for d in self.dirs:
            dev = d['egress_intf']
            delay = self.link_by_id[d['link_id']]['delay_ms']
            if SANDBOX:   # no netem in the test sandbox: queue only
                sh(f'tc qdisc replace dev {dev} root handle 10: tbf rate {rate} '
                   f'burst {TBF_BURST_BYTES} limit {QUEUE_LIMIT_BYTES}')
            else:
                sh(f'tc qdisc replace dev {dev} root handle 1: netem {netem_args(delay, normal_dist=self.normal_dist)}')
                sh(f'tc qdisc add dev {dev} parent 1:1 handle 10: tbf rate {rate} '
                   f'burst {TBF_BURST_BYTES} limit {QUEUE_LIMIT_BYTES}')

    def set_noise(self, link_id, jitter_ms=0.0, loss_pct=0.0, extra_delay_ms=0.0):
        """Apply (or clear, with zeros) non-congestive noise on BOTH directions of a link."""
        l = self.link_by_id[link_id]
        out = []
        for d in (self.dirs[2 * link_id], self.dirs[2 * link_id + 1]):
            if SANDBOX:
                out.append(0)   # noise cannot be emulated without netem; schedule is still logged
                continue
            r = sh(f'tc qdisc change dev {d["egress_intf"]} handle 1: netem '
                   f'{netem_args(l["delay_ms"] + extra_delay_ms, jitter_ms, loss_pct, self.normal_dist)}', check=False)
            out.append(r.returncode)
        return out

    def set_link_state(self, link_id, up):
        l = self.link_by_id[link_id]
        rcs = []
        for node, nb in ((l['a'], l['b']), (l['b'], l['a'])):
            rcs.append(sh(f'ip link set dev {self.iface[(node, nb)]["intf"]} {"up" if up else "down"}',
                          check=False).returncode)
        return rcs

    def pair_cookie(self, a, b):
        return a * len(self.hosts) + b + 1

    def _flows(self):
        per_switch = {i: [] for i in self.switches}
        for (a, b), path in self.paths.items():
            c = self.pair_cookie(a, b)
            for k, u in enumerate(path):
                out = self.host_port[u] if u == b else self.iface[(u, path[k + 1])]['port']
                per_switch[u].append(f'cookie={c},table=0,priority=100,ip,nw_src={host_ip(a)},'
                                     f'nw_dst={host_ip(b)},actions=output:{out}')
        for i, rules in per_switch.items():
            name = self.switches[i].name
            path = f'/tmp/ca_flows_{name}.txt'
            with open(path, 'w') as f:
                f.write('\n'.join(rules) + '\n')
            sh(f'ovs-ofctl -O OpenFlow13 del-flows {name}')
            sh(f'ovs-ofctl -O OpenFlow13 add-flows {name} {path}')
        self.n_rules = sum(len(v) for v in per_switch.values())

    def describe(self):
        """Static description of this network instance, stored with every run."""
        return {'directions': [{k: d[k] for k in ('dir_id', 'link_id', 'src', 'dst', 'egress_intf', 'ingress_intf')}
                               for d in self.dirs],
                'paths': {f'{a}-{b}': p for (a, b), p in self.paths.items()},
                'pair_cookie': {f'{a}-{b}': self.pair_cookie(a, b) for (a, b) in self.paths},
                'switch_names': {i: s.name for i, s in self.switches.items()},
                'host_names': {i: h.name for i, h in self.hosts.items()},
                'n_flow_rules': self.n_rules}

    def stop(self):
        if self.net is not None:
            self.net.stop()
        sh('mn -c', check=False)

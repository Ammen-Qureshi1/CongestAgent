"""
Build the two topology files used by the dataset generator from SNDlib sources
(mirrored in TopoHub, https://github.com/piotrjurkiewicz/topohub,
commit 7a6cc7f6bdddad87ea3e63c4d73dcb3ca14f98cb):

  nsfnet.json : SNDlib "nobel-us"  -> 14 nodes, 21 links (NSFNET sites)   [train + test]
  geant.json  : SNDlib "geant"     -> 22 nodes, 36 links                  [held-out test only]

Link settings (link settings): 10 Mbps, queue of 100 full-size packets,
propagation delay 2-10 ms. The delay is taken from the real link length
(SNDlib 'dist', km) at 200 km/ms (light in fibre) and clipped to [2, 10] ms.

Regions (for the regional decision agents, RQ2) are assigned by longitude
into 3 groups of roughly equal size.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPACITY_MBPS = 10.0
QUEUE_PACKETS = 100
KM_PER_MS = 200.0
DELAY_MIN_MS, DELAY_MAX_MS = 2.0, 10.0
N_REGIONS = 3


def convert(src, name, role):
    d = json.loads((HERE / 'sources' / src).read_text())
    nodes = sorted(d['nodes'], key=lambda n: n['id'])
    ids = [n['id'] for n in nodes]
    assert ids == list(range(len(ids))), 'node ids must be 0..N-1'
    # regions by longitude (west -> east), equal-size groups
    order = sorted(ids, key=lambda i: nodes[i]['pos'][0])
    region = {}
    for rank, i in enumerate(order):
        region[i] = min(N_REGIONS - 1, rank * N_REGIONS // len(ids))
    links, seen = [], set()
    for e in d['edges']:
        a, b = sorted((e['source'], e['target']))
        if (a, b) in seen or a == b:
            continue
        seen.add((a, b))
        raw = e['dist'] / KM_PER_MS
        links.append({'id': len(links), 'a': a, 'b': b, 'dist_km': round(e['dist'], 2),
                      'delay_ms_geo': round(raw, 3),
                      'delay_ms': round(min(DELAY_MAX_MS, max(DELAY_MIN_MS, raw)), 1),
                      'capacity_mbps': CAPACITY_MBPS, 'queue_packets': QUEUE_PACKETS})
    out = {'name': name, 'role': role, 'source': f'SNDlib {src} via TopoHub commit 7a6cc7f6',
           'n_nodes': len(ids), 'n_links': len(links),
           'nodes': [{'id': n['id'], 'name': n['name'], 'lon': n['pos'][0], 'lat': n['pos'][1],
                      'region': region[n['id']]} for n in nodes],
           'links': links}
    (HERE / f'{name}.json').write_text(json.dumps(out, indent=1))
    clipped = sum(1 for l in links if l['delay_ms'] != round(l['delay_ms_geo'], 1))
    print(f'{name}: {out["n_nodes"]} nodes, {out["n_links"]} links, '
          f'{clipped} link delays clipped to [{DELAY_MIN_MS}, {DELAY_MAX_MS}] ms, regions '
          f'{[sum(1 for v in region.values() if v == r) for r in range(N_REGIONS)]}')


if __name__ == '__main__':
    convert('sndlib_nobel-us.json', 'nsfnet', 'train_and_test')
    convert('sndlib_geant.json', 'geant', 'test_only')

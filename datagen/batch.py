"""
The full dataset plan (fixed before any run) and an unattended, resumable runner.

  sudo python3 batch.py --plan            # print the plan and time estimate
  sudo python3 batch.py --pilot           # one 120 s pilot run (built + QC), not part of the dataset
  sudo python3 batch.py                   # compact plan (80 runs, ~3.3 h), resumable
  sudo python3 batch.py --size full       # full plan (199 runs, ~8.2 h); finished runs are skipped
  sudo python3 batch.py --group core      # only one group

Splits are by RUN (never random rows). Seeds 1-5 train, 6 validation, 7-8 test.
2% loss, 40 ms jitter and 80 ms delay shift appear only in validation/test (held out); GEANT and the
stress variations are test-only.
"""
import sys
import time
import json
import random
import argparse
import datetime
import traceback

from common import RUNS_DIR, SANDBOX
from scenario import run_id

LOADS = (30, 60, 90)
NOISES = ('none', 'loss0.5', 'loss1', 'loss2', 'jitter5', 'jitter10')
TRAIN, VAL, TEST = (1, 2, 3, 4, 5), (6,), (7, 8)
EST_S = {'nsfnet': 145, 'geant': 185}   # measured: ~132 s per NSFNET run incl. setup


def plan(size='compact'):
    """size='compact' (80 runs, ~3.3 h) is a subset of size='full' (199 runs, ~8.2 h): same run ids and seeds, so running
    'full' later only ADDS runs. Order = priority: if the batch stops early, what exists is balanced.
      round A (essential): every core setting with 1 train + 1 val + 1 test seed
      round B: GEANT (unseen topology) and stress tests
      round C: second train seed
      (full only) round D: train seeds 3-5, second test seed, more GEANT/stress seeds
    """
    runs, seen = [], set()

    def add(group, split, rnd, **c):
        cfg = {'topology': c['topology'], 'load': c['load'], 'noise': c['noise'],
               'transport': c.get('transport', 'mixed'), 'event': c.get('event', 'none'), 'seed': c['seed']}
        rid = run_id(cfg)
        if rid not in seen:
            seen.add(rid)
            runs.append({'group': group, 'split': split, 'round': rnd, 'cfg': cfg, 'run_id': rid})

    def core(seeds_splits, rnd):
        items = []
        for load in LOADS:
            for nz in NOISES:
                for s, sp in seeds_splits:
                    if nz == 'loss2' and sp == 'train':
                        continue          # 2% loss is held out from training
                    items.append((sp, dict(topology='nsfnet', load=load, noise=nz, seed=s)))
        random.Random(2026 + len(runs)).shuffle(items)
        for sp, c in items:
            add('core', sp, rnd, **c)

    def geant(seeds, rnd):
        for load in (60, 90):
            for nz in ('none', 'loss1', 'loss2', 'jitter10'):
                for s in seeds:
                    add('geant', 'test_geant', rnd, topology='geant', load=load, noise=nz, seed=s)

    def stress(loads, seeds, rnd):
        for load in loads:
            for nz in ('loss1', 'jitter10'):
                for s in seeds:
                    add('stress', 'test_stress', rnd, topology='nsfnet', load=load, noise=nz, transport='cubic', seed=s)
                    add('stress', 'test_stress', rnd, topology='nsfnet', load=load, noise=nz, event='bursts', seed=s)
                    add('stress', 'test_stress', rnd, topology='nsfnet', load=load, noise=nz, event='linkfail', seed=s)

    def hard(rnd):
        # harder noise: jitter20 and shift40 in train/val/test; jitter40 and shift80 held out (test only)
        items = []
        for load in LOADS:
            for nz in ('jitter20', 'shift40'):
                for s, sp in ((1, 'train'), (2, 'train'), (6, 'val'), (7, 'test')):
                    items.append((sp, dict(topology='nsfnet', load=load, noise=nz, seed=s)))
            for nz in ('jitter40', 'shift80'):
                items.append(('test', dict(topology='nsfnet', load=load, noise=nz, seed=7)))
        random.Random(4242).shuffle(items)
        for sp, c in items:
            add('hard', sp, rnd, **c)

    core([(1, 'train'), (6, 'val'), (7, 'test')], 'A')
    geant([7], 'B')
    stress([90], [7], 'B')
    core([(2, 'train')], 'C')
    if size in ('hard', 'full'):
        hard('H')
    if size == 'full':
        core([(3, 'train'), (4, 'train'), (5, 'train'), (8, 'test')], 'D')
        geant([8], 'D')
        stress([60], [7], 'D')
        stress([60, 90], [8], 'D')
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--group', choices=('core', 'geant', 'stress', 'hard'))
    ap.add_argument('--limit', type=int)
    ap.add_argument('--size', choices=('compact', 'hard', 'full'), default='compact')
    a = ap.parse_args()
    runs = plan(a.size)
    if a.group:
        runs = [r for r in runs if r['group'] == a.group]
    if a.plan:
        from collections import Counter
        print('runs per group/split:', dict(Counter((r['group'], r['split']) for r in runs)))
        for rnd in sorted({r['round'] for r in runs}):
            rr = [r for r in runs if r['round'] == rnd]
            print(f"  round {rnd}: {len(rr)} runs, {sum(EST_S[r['cfg']['topology']] for r in rr) / 3600:.1f} h")
        est = sum(EST_S[r['cfg']['topology']] for r in runs) / 3600
        print(f'total {len(runs)} runs, estimated {est:.1f} h')
        return
    from run_one import run
    if a.pilot:
        out = run({'topology': 'nsfnet', 'load': 90, 'noise': 'loss1', 'transport': 'mixed', 'event': 'none',
                   'seed': 999}, 'pilot', build=True, runs_dir=RUNS_DIR / 'pilot')
        print(f'pilot written to {out} -> check it before the full batch')
        return
    if SANDBOX:
        sys.exit('Refusing: sandbox mode is for code tests only.')
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f'plan_{a.size}.json').write_text(json.dumps(plan(a.size), indent=1))
    todo = [r for r in runs if not (RUNS_DIR / r['run_id'] / 'DONE').exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f'{len(runs) - len(todo)} done, {len(todo)} to do', flush=True)
    t_start = time.time()
    for n, r in enumerate(todo, 1):
        t = time.time()
        try:
            run(r['cfg'], r['split'], build=False)
        except Exception as e:
            with open(RUNS_DIR / 'failures.log', 'a') as fh:
                fh.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {r['run_id']} {type(e).__name__}: {e}\n"
                         f"{traceback.format_exc()}\n")
            print(f'  FAILED {r["run_id"]}: {e} (logged; it will be retried next time)', flush=True)
        took = time.time() - t
        left = (time.time() - t_start) / n * (len(todo) - n) / 3600
        print(f'[{n}/{len(todo)}] {r["run_id"]} ({r["split"]}) {took:.0f} s | about {left:.1f} h left', flush=True)
    print('BATCH FINISHED. Next: python3 build_all.py')


if __name__ == '__main__':
    main()

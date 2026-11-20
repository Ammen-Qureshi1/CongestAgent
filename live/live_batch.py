"""
Live (closed-loop) experiments for RQ1 (and RQ3 fairness under control). Resumable like batch.py.
Every configuration is run with all controllers back to back (same traffic + noise schedule = paired).
  sudo python3 live_batch.py --plan
  sudo python3 live_batch.py            # runs everything not done yet, seed by seed
"""
import sys, time, json, random, argparse, datetime, traceback
from live_run import run, LIVE_DIR, POLICIES
from scenario import run_id

LOADS = (60, 90)
NOISES = ('none', 'loss2', 'jitter40', 'shift40', 'shift80')
SEEDS = (11, 12, 13)          # new seeds: never used for training or testing the model
EST_S = 150


def plan():
    out = []
    for s in SEEDS:
        cfgs = [dict(topology='nsfnet', load=l, noise=n, transport='mixed', event='none', seed=s) for l in LOADS for n in NOISES]
        random.Random(s).shuffle(cfgs)
        for c in cfgs:
            for p in POLICIES:
                out.append((c, p))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', action='store_true'); ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    runs = plan()
    todo = [(c, p) for c, p in runs if not (LIVE_DIR / f'{run_id(c)}__{p}' / 'DONE').exists()]
    if a.plan:
        print(f'{len(runs)} live runs ({len(runs) // len(SEEDS)} per seed, seeds {SEEDS}); {len(runs) - len(todo)} done, '
              f'{len(todo)} to do, about {len(todo) * EST_S / 3600:.1f} h')
        return
    if a.limit:
        todo = todo[:a.limit]
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'{len(runs) - len(todo)} done, {len(todo)} to do', flush=True)
    t0 = time.time()
    for n, (c, p) in enumerate(todo, 1):
        try:
            run(c, p)
        except Exception as e:
            with open(LIVE_DIR / 'failures.log', 'a') as fh:
                fh.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {run_id(c)}__{p} {type(e).__name__}: {e}\n{traceback.format_exc()}\n")
            print(f'  FAILED {run_id(c)}__{p}: {e} (logged; retried next time)', flush=True)
        left = (time.time() - t0) / n * (len(todo) - n) / 3600
        print(f'[{n}/{len(todo)}] about {left:.1f} h left', flush=True)
    print('LIVE BATCH FINISHED. Next: python3 live_eval.py')


if __name__ == '__main__':
    main()

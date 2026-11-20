"""
Build every finished run and merge them into split files.

  python3 build_all.py            # builds runs not yet built (in parallel), then merges

Output in dataset/:
  intervals_<split>.csv.gz   rows of all runs of that split whose QC is PASS or WARN
  flows_tcp_all.csv.gz, flow_seconds_all.csv.gz  (flow-level data, all runs)
  summary.json               run list, QC status per run, label counts per split
FAIL runs are never merged; they are listed in summary.json with their problems.
"""
import sys
import json
import multiprocessing as mp
from pathlib import Path
import pandas as pd

from common import RUNS_DIR, ROOT, read_json
from build_intervals import build_run


def _build(p):
    try:
        _, qc = build_run(p, quiet=True)
        return str(p), qc['status'], qc['problems'] + qc['warnings']
    except Exception as e:
        return str(p), 'BUILD_ERROR', [f'{type(e).__name__}: {e}']


def main():
    runs = sorted(p for p in Path(RUNS_DIR).iterdir() if (p / 'DONE').exists() and p.name != 'pilot')
    todo = [p for p in runs if not (p / 'BUILT').exists()]
    print(f'{len(runs)} finished runs, {len(todo)} to build', flush=True)
    with mp.Pool(max(1, min(4, mp.cpu_count() - 1))) as pool:
        for i, (p, st, msg) in enumerate(pool.imap_unordered(_build, todo), 1):
            print(f'  [{i}/{len(todo)}] {Path(p).name}: {st} {msg if st != "PASS" else ""}', flush=True)
    out = ROOT / 'dataset'
    out.mkdir(exist_ok=True)
    summary = {'runs': [], 'splits': {}}
    by_split = {}
    for p in runs:
        if not (p / 'qc.json').exists():
            summary['runs'].append({'run_id': p.name, 'status': 'BUILD_ERROR'})
            continue
        qc, meta = read_json(p / 'qc.json'), read_json(p / 'meta.json')
        summary['runs'].append({'run_id': p.name, 'split': meta['split'], 'status': qc['status'],
                                'problems': qc['problems'], 'warnings': qc['warnings']})
        if qc['status'] in ('PASS', 'WARN'):
            by_split.setdefault(meta['split'], []).append(p)
    for split, ps in sorted(by_split.items()):
        df = pd.concat([pd.read_csv(p / 'intervals.csv.gz') for p in ps], ignore_index=True)
        df.to_csv(out / f'intervals_{split}.csv.gz', index=False, compression='gzip', float_format='%.6g')
        post = df[~df.meta_warmup]
        summary['splits'][split] = {'runs': len(ps), 'rows': len(df), 'rows_after_warmup': len(post),
                                    'labels_after_warmup': post.gt_label.value_counts().to_dict()}
        print(f'{split}: {len(ps)} runs, {len(df)} rows, labels {summary["splits"][split]["labels_after_warmup"]}')
    for name in ('flows_tcp', 'flow_seconds', 'flows_udp'):
        parts = [pd.read_csv(p / f'{name}.csv.gz').assign(run_id=p.name) for p in runs if (p / f'{name}.csv.gz').exists()]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out / f'{name}_all.csv.gz', index=False, compression='gzip')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    bad = [r for r in summary['runs'] if r['status'] not in ('PASS', 'WARN')]
    print(f'done -> {out}. Runs not merged: {len(bad)}')


if __name__ == '__main__':
    main()

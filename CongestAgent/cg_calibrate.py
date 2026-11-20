"""cg_calibrate.py - calibrate the flow-level simulator against the Mininet dataset, and report differences.

Usage:
    python cg_calibrate.py --data /content/work --out calib.json [--topo-dir ...]
Needs in --data:  intervals_test_features.csv.gz (or train), flows_tcp_all.csv.gz
Writes calib.json (used by cg_run.py via --calib) and calibration_report.md (sim vs Mininet table, for Step 11).
"""
import argparse, json, os
import numpy as np, pandas as pd
from cg_sim import Scenario, Net, PerceptionEmu, get_topology, load_calib, CALIB_DEFAULT

LOADS = (0.3, 0.6, 0.9)


def dataset_stats(path, chunk=300_000):
    """Per load level, noise-free NSFNET runs: share of congestion / normal / uncertain, mean util, mean backlog."""
    cols = ["meta_load", "meta_noise", "meta_topology", "meta_event", "meta_warmup", "gt_label", "f_util",
            "gt_backlog_mean_frac"]
    acc = {}
    for ch in pd.read_csv(path, usecols=lambda c: c in cols, chunksize=chunk, low_memory=False):
        ch = ch[~ch["meta_warmup"].astype(bool)]
        if "meta_noise" in ch:
            ch = ch[ch["meta_noise"].astype(str) == "none"]
        if "meta_event" in ch:
            ch = ch[ch["meta_event"].astype(str).isin(["none", "nan"])]
        if "meta_topology" in ch:
            ch = ch[ch["meta_topology"].astype(str).str.lower().str.contains("nsf")]
        load = ch["meta_load"].astype(float)
        ch = ch.assign(load=np.where(load > 1.5, load / 100.0, load).round(2))
        ch["util"] = ch["f_util"].clip(upper=1.0)
        for l, g in ch.groupby("load"):
            a = acc.setdefault(l, dict(n=0, cong=0, norm=0, unc=0, util=0.0, bl=0.0))
            a["n"] += len(g)
            a["cong"] += int((g.gt_label == "congestion").sum())
            a["norm"] += int((g.gt_label == "normal").sum())
            a["unc"] += int((g.gt_label == "uncertain").sum())
            a["util"] += float(g.util.sum())
            a["bl"] += float(g.gt_backlog_mean_frac.sum())
    return {l: dict(share_congestion=a["cong"] / a["n"], share_normal=a["norm"] / a["n"],
                    share_uncertain=a["unc"] / a["n"], mean_util=a["util"] / a["n"], mean_backlog=a["bl"] / a["n"],
                    rows=a["n"]) for l, a in acc.items() if a["n"]}


def sim_stats(calib, topo, seeds=8):
    out = {}
    for l in LOADS:
        r = []
        for s in range(seeds):
            net = Net(topo, Scenario("nsfnet", l, "none", "mixed", "none", 500 + s, 120.0), calib,
                      perception=PerceptionEmu(None, "none", np.random.default_rng(0)), fast_loop=False, ecmp=True)
            for _ in range(net.T):
                net.step()
            m = net.metrics()
            tot = m["share_congestion"] + m["share_degradation"]
            r.append((m["share_congestion"], m["delay_mean_ms"], m["loss_pct"], m["throughput_mbps"]))
        r = np.mean(r, 0)
        out[l] = dict(share_congestion=r[0], delay_mean_ms=r[1], loss_pct=r[2], throughput_mbps=r[3])
    return out


def flows_stats(path):
    df = pd.read_csv(path)
    print("flows_tcp_all columns:", list(df.columns))
    low = {c.lower(): c for c in df.columns}
    pick = lambda *keys: next((low[c] for c in low for k in keys if k in c), None)
    size, cc, dur, status = pick("size", "bytes"), pick("cc", "congestion", "algo"), \
        pick("duration", "fct", "complet", "dur"), pick("status")
    if not (size and cc and dur):
        print("[warn] could not identify size / cc / duration columns; skipping flow calibration")
        return None
    d = df
    if status:
        d = d[d[status].astype(str).str.lower().isin(["ok", "done", "completed", "complete", "success", "0"])]
    d = d[(d[dur] > 0) & (d[size] > 0)]
    d = d.assign(thr=d[size] * 8 / d[dur] / 1e6)
    g = d.groupby(d[cc].astype(str).str.lower()).thr.agg(["mean", "median", "count"])
    sz = np.log(d[size].values.astype(float))
    return dict(by_cc=g, size_mean_kb=float(d[size].mean() / 1e3), size_sigma=float(sz.std()), table=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--topo-dir", default=None)
    ap.add_argument("--intervals", default="intervals_test_features.csv.gz")
    a = ap.parse_args()
    topo = get_topology("nsfnet", a.topo_dir)
    ds = dataset_stats(os.path.join(a.data, a.intervals))
    print("Mininet (noise-free NSFNET):", json.dumps(ds, indent=1))
    best, best_err, calib = None, 1e9, load_calib()
    for sg in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        for on in (2.0, 4.0, 6.0):
            c = dict(calib, tcp_sigma=sg, udp_mean_on=on)
            ss = sim_stats(c, topo, seeds=4)
            err = sum(abs(np.log((ss[l]["share_congestion"] + 0.003) / (ds[l]["share_congestion"] + 0.003)))
                      for l in LOADS if l in ds)
            print(f"tcp_sigma={sg} udp_on={on}: error {err:.3f}  sim congestion share " +
                  ", ".join(f"{l}:{ss[l]['share_congestion']:.3f}" for l in LOADS), flush=True)
            if err < best_err:
                best, best_err = c, err
    calib = best
    fl_path = os.path.join(a.data, "flows_tcp_all.csv.gz")
    fs = flows_stats(fl_path) if os.path.exists(fl_path) else None
    if fs is not None:
        print(fs["table"])
        cub = next((v for k, v in fs["by_cc"]["mean"].items() if "cubic" in k), None)
        bbr = next((v for k, v in fs["by_cc"]["mean"].items() if "bbr" in k), None)
        if cub and bbr:
            r = bbr / cub                                   # measured BBR / CUBIC throughput ratio
            d = 0.1                                         # typical drop fraction under overload
            calib["m_cubic"] = float(np.clip((1 - (1 - calib["m_bbr"] * d) / r) / d, 0.8, 3.0))
            print(f"BBR/CUBIC throughput ratio in Mininet flows: {r:.2f} -> m_cubic={calib['m_cubic']:.2f}")
    json.dump(calib, open(a.out, "w"), indent=2)
    ss = sim_stats(calib, topo, seeds=8)
    rep = ["# Simulator vs Mininet (noise-free NSFNET, post warm-up)\n", f"Fitted parameters: {calib}\n",
           "| load | share congestion (Mininet) | share congestion (sim) | mean util (Mininet) | mean backlog (Mininet) | "
           "uncertain share (Mininet) |", "|---|---|---|---|---|---|"]
    for l in LOADS:
        if l in ds:
            rep.append(f"| {int(l*100)}% | {ds[l]['share_congestion']:.4f} | {ss[l]['share_congestion']:.4f} | "
                       f"{ds[l]['mean_util']:.3f} | {ds[l]['mean_backlog']:.3f} | {ds[l]['share_uncertain']:.3f} |")
    rep.append("\nThe simulator has no `uncertain` class and its link utilisation is not read per port, so only the "
               "congestion share is fitted. Remaining differences are limits of the flow-level model.")
    open("calibration_report.md", "w").write("\n".join(rep))
    print("\n".join(rep))
    print(f"\nwrote {a.out} and calibration_report.md")


if __name__ == "__main__":
    main()

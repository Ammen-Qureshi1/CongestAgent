"""CongestAgent perception model (LightGBM), defaults sized for about 12 GB RAM and 2 CPU cores:
  * keeps ALL normal rows (--normal-keep 1.0)
  * reads 200,000 rows at a time
  * uses every CPU core
  * can run all three feature sets one after another with --featureset all

Quick test:
    python train_perception_colab.py --data . --out results_quick --featureset full --quick
Full run (all feature sets):
    python train_perception_colab.py --data . --out results --featureset all

Feature sets: full (base + long-window) | nolong (base only) | symptom (delay and loss only)
              robust (full minus raw delay-level features; added after the delay-shift errors seen on the test split)
"""
import argparse, gc, json, os, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, classification_report, confusion_matrix

CLASSES = ["normal", "degradation", "congestion"]
CLS2ID = {c: i for i, c in enumerate(CLASSES)}
GROUP = ["meta_run_id", "meta_dir_id"]
FILES = {"train": "intervals_train_features.csv.gz",
         "val": "intervals_val_features.csv.gz",
         "test": "intervals_test_features.csv.gz",
         "test_geant": "intervals_test_geant_features.csv.gz",
         "test_stress": "intervals_test_stress_features.csv.gz"}
PREFIXES = ("f_", "p_", "w_", "nb_", "static_")
SYMPTOM_KEYS = ("delay", "loss", "probe_lost")
NEED_META = ["meta_run_id", "meta_dir_id", "meta_k", "meta_warmup", "gt_label", "w_complete"]
EXTRA_META = ["meta_event", "meta_transport", "meta_noise"]   # only for the stress breakdown
CARRY = 5000   # rows kept from the previous piece so rolling windows stay correct across pieces


def add_long(df):
    """Rolling means over 5 s (20 intervals) and 10 s (40 intervals), per run and per link direction.
    Computed piece by piece; the windows follow long_features.py."""
    df = df.sort_values(GROUP + ["meta_k"]).reset_index(drop=True)
    tx = df["f_tx_pps"].replace(0, np.nan)
    b = pd.DataFrame({
        "pathloss": df["f_pathloss_wmean"],
        "portloss": ((df["f_tx_pps"] - df["f_rx_pps_far"]) / tx).clip(0, 1),
        "delay_spread": df["f_delay_std_ms"],
        "excess_delay": df["f_delay_mean_ms"] - df["static_prop_delay_ms"],
        "probe_loss": df["f_probe_loss_ratio"],
        "util": df["f_util"],
        "delay_range": df["f_delay_max_ms"] - df["f_delay_min_ms"],
    }).astype("float32")
    keys = [df[g] for g in GROUP]
    new = {}
    for name in b.columns:
        grp = b[name].groupby(keys, sort=False)
        for sec, n in (("5s", 20), ("10s", 40)):
            new[f"long_{name}_{sec}"] = grp.transform(
                lambda s: s.rolling(n, min_periods=1).mean()).astype("float32")
    return pd.concat([df, pd.DataFrame(new)], axis=1), list(new)


def choose(base, long_cols, fs):
    if fs == "notrend":  # robust minus trend / correlation features (ablation A2)
        return [c for c in choose(base, long_cols, "robust") if "trend" not in c and "corr" not in c]
    if fs == "robust":   # no raw delay LEVEL features (a +80 ms shift fools them); keeps util, spread, trend, loss
        drop = ("delay_mean", "delay_min", "delay_max", "delay_p50", "p_delay", "w_delay_level",
                "queue_delay", "excess_delay")
        return [c for c in base + long_cols if not any(k in c for k in drop)]
    if fs == "symptom":
        return [c for c in base if any(k in c for k in SYMPTOM_KEYS) and "util" not in c and "corr" not in c]
    return base + (long_cols if fs == "full" else [])


def stream(path, fs, quick, chunk, handle):
    """Read one file in pieces. Calls handle(usable_rows, base_cols, long_cols) for each piece."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    base = [c for c in header if c.startswith(PREFIXES) and c != "w_complete"]
    meta = NEED_META + [c for c in EXTRA_META if c in header]
    reader = pd.read_csv(path, usecols=base + meta, dtype={c: "float32" for c in base},
                         chunksize=chunk, low_memory=False)
    carry, n_rows = None, 0
    for i, ch in enumerate(reader):
        if quick and i >= 2:
            break
        ch["_new"] = True
        df = ch if carry is None else pd.concat([carry, ch], ignore_index=True)
        carry = df.tail(CARRY).copy()
        carry["_new"] = False
        df["f_util"] = df["f_util"].clip(upper=1.0)     # late port-counter reads (see docs/README_DATASET.md)
        long_cols = []
        if fs in ("full", "robust", "notrend"):
            df, long_cols = add_long(df)
        keep = (df["_new"] & ~df["meta_warmup"].astype(bool) & (df["gt_label"] != "uncertain")
                & df["w_complete"].astype(bool))
        d = df.loc[keep].copy()
        d["y"] = d["gt_label"].map(CLS2ID).astype("int8")
        n_rows += len(d)
        handle(d, base, long_cols)
        del df, d, ch
        gc.collect()
    return n_rows


def collect_training(name, folder, fs, quick, chunk, keep_normal, rng):
    store, state = [], {}

    def h(d, base, long_cols):
        feats = choose(base, long_cols, fs)
        state["feats"] = feats
        if keep_normal >= 1.0:
            m = np.ones(len(d), dtype=bool)
        else:
            m = ((d["y"] != 0) | (rng.random(len(d)) < keep_normal)).values
        x = d.loc[m, feats + ["y"]].copy()
        x["w0"] = np.where(x["y"] == 0, 1.0 / keep_normal, 1.0).astype("float32")
        store.append(x)

    t = time.time()
    n = stream(os.path.join(folder, FILES[name]), fs, quick, chunk, h)
    df = pd.concat(store, ignore_index=True)
    feats = state["feats"]
    tot = df["w0"].sum()
    cw = {}
    for c in range(3):
        s = df.loc[df["y"] == c, "w0"].sum()
        cw[c] = tot / (3 * s) if s > 0 else 1.0      # "balanced" class weights
    w = (df["w0"] * df["y"].map(cw)).astype("float32")
    print(f"{name}: {n:,} usable rows, kept {len(df):,} for training ({time.time()-t:.0f}s)", flush=True)
    return df[feats], df["y"].astype(int), w, feats


def report(name, y, p):
    r = classification_report(y, p, labels=[0, 1, 2], target_names=CLASSES, output_dict=True, zero_division=0)
    out = {"macro_f1": float(f1_score(y, p, average="macro", labels=[0, 1, 2])),
           "per_class_f1": {c: r[c]["f1-score"] for c in CLASSES},
           "per_class_precision": {c: r[c]["precision"] for c in CLASSES},
           "per_class_recall": {c: r[c]["recall"] for c in CLASSES},
           "confusion_matrix(rows=true, cols=pred)": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
           "n": int(len(y))}
    print(f"== {name}: macro F1 {out['macro_f1']:.4f} | " +
          " | ".join(f"{c} F1 {out['per_class_f1'][c]:.3f}" for c in CLASSES), flush=True)
    return out


def evaluate(name, model, feats, folder, fs, quick, chunk, threads, results):
    ys, ps, extra = [], [], []

    def h(d, base, long_cols):
        p = model.predict(d[feats], num_iteration=model.best_iteration, num_threads=threads).argmax(1)
        ys.append(d["y"].values.astype("int8"))
        ps.append(p.astype("int8"))
        if name == "test_stress":
            e = d[[c for c in EXTRA_META if c in d.columns]].astype(str)
            extra.append(e.assign(y=d["y"].values, p=p))

    stream(os.path.join(folder, FILES[name]), fs, quick, chunk, h)
    y, p = np.concatenate(ys), np.concatenate(ps)
    results[name] = report(name, y, p)
    if extra:                                   # breakdown by scenario
        e = pd.concat(extra, ignore_index=True)
        for col in [c for c in EXTRA_META if c in e.columns]:
            results[f"stress_by_{col}"] = {
                str(v): float(f1_score(g["y"], g["p"], average="macro", labels=[0, 1, 2]))
                for v, g in e.groupby(col)}


def run_one(a, fs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    print(f"\n######## featureset = {fs} ########", flush=True)
    Xtr, ytr, wtr, feats = collect_training("train", a.data, fs, a.quick, a.chunk, a.normal_keep, rng)
    Xva, yva, wva, _ = collect_training("val", a.data, fs, a.quick, a.chunk, a.normal_keep, rng)
    print(f"{len(feats)} features", flush=True)
    dtr = lgb.Dataset(Xtr, ytr, weight=wtr, params={"max_bin": 255}, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, weight=wva, reference=dtr, free_raw_data=True)
    del Xtr, Xva
    gc.collect()

    params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, max_bin=255, seed=a.seed, verbose=-1, num_threads=a.threads)
    model = lgb.train(params, dtr, num_boost_round=30 if a.quick else 2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(25)])
    model.save_model(os.path.join(out_dir, "model.txt"))
    del dtr, dva
    gc.collect()

    results = {"featureset": fs, "n_features": len(feats), "best_iteration": model.best_iteration}
    for k in ["val", "test", "test_geant", "test_stress"]:
        evaluate(k, model, feats, a.data, fs, a.quick, a.chunk, a.threads, results)
        json.dump(results, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)   # saved after each file
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    imp.to_csv(os.path.join(out_dir, "feature_importance.csv"))
    json.dump(feats, open(os.path.join(out_dir, "features.json"), "w"))
    print("top features:\n", imp.head(15), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="perception_out")
    ap.add_argument("--featureset", choices=["full", "nolong", "symptom", "robust", "notrend", "all"], default="full")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--chunk", type=int, default=200_000, help="rows read at a time")
    ap.add_argument("--normal-keep", type=float, default=1.0, help="share of normal rows kept (1.0 = all)")
    ap.add_argument("--quick", action="store_true", help="tiny test run")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sets = ["full", "nolong", "symptom"] if a.featureset == "all" else [a.featureset]
    summary = {}
    for fs in sets:
        out_dir = os.path.join(a.out, fs) if a.featureset == "all" else a.out
        r = run_one(a, fs, out_dir)
        summary[fs] = {k: r[k]["macro_f1"] for k in ["val", "test", "test_geant", "test_stress"]}
        gc.collect()
    if len(sets) > 1:
        json.dump(summary, open(os.path.join(a.out, "summary_macro_f1.json"), "w"), indent=2)
        print("\nSUMMARY (macro F1):")
        for fs, s in summary.items():
            print(fs, {k: round(v, 4) for k, v in s.items()})
    print("done", flush=True)


if __name__ == "__main__":
    main()

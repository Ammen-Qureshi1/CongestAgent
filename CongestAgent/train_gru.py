"""train_gru.py - small GRU perception model + comparison with LightGBM (Step 5a of the proposal).

The proposal says: "compare gradient-boosted decision trees with a small GRU and keep the better one".
This script trains a GRU on windows of 8 intervals (2 s) of the SAME features as the LightGBM model, tests it
on the same splits, and writes perception_comparison.md with a recommendation.

Needs train_perception_colab.py in the same folder (it reuses its feature code).

Quick test:
    python train_gru.py --data /content/work --out /content/gru_quick --featureset robust --quick
Real run (use a GPU runtime in Colab if you can: Runtime > Change runtime type > T4 GPU; CPU also works):
    python train_gru.py --data /content/work --out /content/drive/MyDrive/CongestAgent/results/gru_robust \
        --featureset robust --compare /content/drive/MyDrive/CongestAgent/results/robust \
                                      /content/drive/MyDrive/CongestAgent/results/full
"""
import argparse, gc, json, os, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import f1_score
import train_perception_colab as T

W = 8   # window length in intervals (2 s), as in the proposal


# ----------------------------------------------------------------------------- data
def chunk_stream(path, fs, quick, chunk):
    """Yield (df sorted by run, link direction, interval; feature names). df keeps `_new` = rows of this piece."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    base = [c for c in header if c.startswith(T.PREFIXES) and c != "w_complete"]
    meta = T.NEED_META + [c for c in T.EXTRA_META if c in header]
    reader = pd.read_csv(path, usecols=base + meta, dtype={c: "float32" for c in base}, chunksize=chunk,
                         low_memory=False)
    carry = None
    for i, ch in enumerate(reader):
        if quick and i >= 2:
            break
        ch["_new"] = True
        df = ch if carry is None else pd.concat([carry, ch], ignore_index=True)
        carry = df.tail(T.CARRY).copy()
        carry["_new"] = False
        df["f_util"] = df["f_util"].clip(upper=1.0)
        long_cols = []
        if fs in ("full", "robust", "notrend"):
            df, long_cols = T.add_long(df)            # also sorts by run, direction, interval
        else:
            df = df.sort_values(T.GROUP + ["meta_k"]).reset_index(drop=True)
        yield df, T.choose(base, long_cols, fs)


def make_windows(df, feats, keep_normal, rng):
    """One window (W x features) per usable interval. Windows never cross runs / link directions."""
    n = len(df)
    X = df[feats].to_numpy(np.float32)
    gid = df.groupby(T.GROUP, sort=False).ngroup().to_numpy()
    k = df["meta_k"].to_numpy()
    ok = np.zeros(n, bool)
    if n >= W:
        ok[W - 1:] = (gid[W - 1:] == gid[:n - W + 1]) & (k[W - 1:] - k[:n - W + 1] == W - 1)
    usable = (df["_new"].to_numpy() & ~df["meta_warmup"].astype(bool).to_numpy() &
              (df["gt_label"] != "uncertain").to_numpy() & df["w_complete"].astype(bool).to_numpy())
    sel = np.where(ok & usable)[0]
    y = df["gt_label"].map(T.CLS2ID).to_numpy()[sel].astype(np.int64)
    w0 = np.ones(len(sel), np.float32)
    if keep_normal < 1.0:
        keep = (y != 0) | (rng.random(len(sel)) < keep_normal)
        sel, y, w0 = sel[keep], y[keep], np.where(y[keep] == 0, 1.0 / keep_normal, 1.0).astype(np.float32)
    idx = sel[:, None] + np.arange(-(W - 1), 1)[None, :]
    extra = df.iloc[sel][[c for c in T.EXTRA_META if c in df.columns]].astype(str).reset_index(drop=True)
    return X[idx], y, w0, extra


def normalise(Xw, mean, std):
    Xw -= mean
    Xw /= std
    np.nan_to_num(Xw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(Xw, -10, 10, out=Xw)
    return Xw


# ----------------------------------------------------------------------------- model
class GRUNet(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1])


def predict(model, Xw, dev, bs=8192):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xw), bs):
            out.append(model(torch.from_numpy(Xw[i:i + bs]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([], int)


def collect(path, fs, quick, chunk, keep_normal, rng):
    Xs, ys, ws, feats = [], [], [], None
    for df, feats in chunk_stream(path, fs, quick, chunk):
        X, y, w0, _ = make_windows(df, feats, keep_normal, rng)
        Xs.append(X), ys.append(y), ws.append(w0)
        del df
    X, y, w0 = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    return X, y, w0, feats


def class_weights(y, w0):
    tot = w0.sum()
    cw = np.array([tot / (3 * w0[y == c].sum()) if (y == c).any() else 1.0 for c in range(3)], np.float32)
    return w0 * cw[y]


# ----------------------------------------------------------------------------- evaluation helpers
def evaluate(name, model, feats, mean, std, a, dev, results):
    ys, ps, extras = [], [], []
    for df, _ in chunk_stream(os.path.join(a.data, T.FILES[name]), a.featureset, a.quick, a.chunk):
        X, y, _, extra = make_windows(df, feats, 1.0, None)
        if len(y):
            ps.append(predict(model, normalise(X, mean, std), dev))
            ys.append(y)
            if name == "test_stress":
                extras.append(extra.assign(y=y, p=ps[-1]))
    y, p = np.concatenate(ys), np.concatenate(ps)
    results[name] = T.report(name, y, p)
    if extras:
        e = pd.concat(extras, ignore_index=True)
        for col in [c for c in T.EXTRA_META if c in e.columns]:
            results[f"stress_by_{col}"] = {str(v): float(f1_score(g["y"], g["p"], average="macro", labels=[0, 1, 2]))
                                           for v, g in e.groupby(col)}


def rates(m):
    """Needless-reroute risk and missed congestion (%), summed over the held-out splits."""
    cm = sum(np.array(m[s]["confusion_matrix(rows=true, cols=pred)"], float)
             for s in ("test", "test_geant", "test_stress"))
    return 100 * cm[1][2] / cm[1].sum(), 100 * (cm[2].sum() - cm[2][2]) / cm[2].sum()


def compare(out_dir, my_name, my_results, others):
    rows = []
    allm = {my_name: my_results}
    for d in others:
        p = os.path.join(d, "metrics.json")
        if os.path.exists(p):
            allm[os.path.basename(d.rstrip("/"))] = json.load(open(p))
    for n, m in allm.items():
        fa, mc = rates(m)
        f1 = {s: m[s]["macro_f1"] for s in ("val", "test", "test_geant", "test_stress")}
        rows.append(dict(model=n, **{f"F1 {k}": round(v, 4) for k, v in f1.items()},
                         mean_F1_heldout=round(np.mean([f1["test"], f1["test_geant"], f1["test_stress"]]), 4),
                         degradation_called_congestion_pct=round(fa, 3), congestion_missed_pct=round(mc, 3)))
    df = pd.DataFrame(rows)
    gru = df[df.model == my_name].iloc[0]
    lgb = df[df.model != my_name]
    msg = "No LightGBM results were given with --compare, so no recommendation was made."
    if len(lgb):
        best = lgb.sort_values("mean_F1_heldout", ascending=False).iloc[0]
        diff = gru.mean_F1_heldout - best.mean_F1_heldout
        safer = gru.degradation_called_congestion_pct < best.degradation_called_congestion_pct
        if diff > 0.005 and safer:
            msg = f"KEEP THE GRU: higher mean held-out macro F1 (+{diff:.3f}) and fewer needless-reroute errors."
        elif diff < -0.005 or not safer and diff <= 0.005:
            msg = (f"KEEP LIGHTGBM ({best.model}): the GRU is not clearly better "
                   f"(mean held-out macro F1 difference {diff:+.3f}), and LightGBM is cheaper to run.")
        else:
            msg = (f"MIXED: GRU F1 difference {diff:+.3f}, needless-reroute errors "
                   f"{gru.degradation_called_congestion_pct:.2f}% vs {best.degradation_called_congestion_pct:.2f}%. "
                   "Prefer the model with fewer needless-reroute errors, or keep LightGBM for lower cost.")
    cols = list(df.columns)
    md = "# Perception model comparison (Step 5a)\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        md += "| " + " | ".join(str(v) for v in r.values) + " |\n"
    md += f"\n**Decision rule:** keep the model with the higher mean macro F1 on the three held-out splits; if the " \
          f"difference is below 0.005, prefer fewer needless-reroute errors / lower cost.\n\n**Result:** {msg}\n"
    open(os.path.join(out_dir, "perception_comparison.md"), "w").write(md)
    print("\n" + md)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="gru_out")
    ap.add_argument("--featureset", choices=["full", "nolong", "symptom", "robust", "notrend"], default="robust")
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--normal-keep", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--compare", nargs="*", default=[], help="folders with LightGBM metrics.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)

    t0 = time.time()
    Xtr, ytr, w0tr, feats = collect(os.path.join(a.data, T.FILES["train"]), a.featureset, a.quick, a.chunk,
                                    a.normal_keep, rng)
    Xva, yva, w0va, _ = collect(os.path.join(a.data, T.FILES["val"]), a.featureset, a.quick, a.chunk,
                                a.normal_keep, rng)
    print(f"windows: train {len(ytr):,} val {len(yva):,}, {len(feats)} features ({time.time()-t0:.0f}s)", flush=True)
    last = Xtr[:, -1, :]
    mean = np.nanmean(last, 0).astype(np.float32)
    std = np.nanstd(last, 0).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalise(Xtr, mean, std)
    normalise(Xva, mean, std)
    wtr, wva = class_weights(ytr, w0tr), class_weights(yva, w0va)

    model = GRUNet(len(feats), a.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss(reduction="none")
    Xva_t, yva_t, wva_t = (torch.from_numpy(Xva).to(dev), torch.from_numpy(yva).to(dev), torch.from_numpy(wva).to(dev))
    best, best_state, bad, bs = 1e9, None, 0, 2048
    for ep in range(2 if a.quick else a.epochs):
        model.train()
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x, y, w = (torch.from_numpy(Xtr[b]).to(dev), torch.from_numpy(ytr[b]).to(dev),
                       torch.from_numpy(wtr[b]).to(dev))
            opt.zero_grad()
            (lossf(model(x), y) * w).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum((lossf(model(Xva_t[i:i + 8192]), yva_t[i:i + 8192]) * wva_t[i:i + 8192]).sum().item()
                     for i in range(0, len(yva), 8192)) / wva_t.sum().item()
        print(f"epoch {ep+1}: val loss {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_state)
    torch.save(dict(state=model.state_dict(), feats=feats, mean=mean, std=std, hidden=a.hidden, window=W),
               os.path.join(a.out, "model.pt"))
    del Xtr, Xva, Xva_t
    gc.collect()

    results = dict(featureset=a.featureset, model="gru", n_features=len(feats), window=W)
    for k in ["val", "test", "test_geant", "test_stress"]:
        evaluate(k, model, feats, mean, std, a, dev, results)
        json.dump(results, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    json.dump(feats, open(os.path.join(a.out, "features.json"), "w"))
    compare(a.out, "gru_" + a.featureset, results, a.compare)
    print("done", flush=True)


if __name__ == "__main__":
    main()

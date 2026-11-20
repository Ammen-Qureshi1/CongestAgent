"""cg_plots.py - all figures for the report (bars with 95% confidence intervals).

Usage:
    python cg_plots.py --workdir /content/drive/MyDrive/CongestAgent/cg_out \
                       --results-dir /content/drive/MyDrive/CongestAgent/results
Reads  <workdir>/results/*.csv  (core, variations, ablation, stress_delay1/2)  and  <results-dir>/*/metrics.json
Writes <workdir>/figures/*.png
"""
import argparse, glob, json, math, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

COL = {"CongestAgent": "#1f77b4", "B1": "#7f7f7f", "B2": "#ff7f0e", "B3": "#d62728",
       "A1": "#9467bd", "A2": "#8c564b", "A3": "#e377c2", "A4": "#17becf", "A5": "#bcbd22"}
LABEL = {"delay_p95_ms": "p95 delay (ms)", "delay_mean_ms": "mean delay (ms)", "loss_pct": "packet loss (%)",
         "throughput_mbps": "throughput (Mbps)", "jain": "Jain's index (CUBIC vs BBR)",
         "false_reroute_rate": "false-reroute rate", "route_changes_per_min": "route changes / min",
         "decision_latency_ms": "decision latency (ms per interval)",
         "msgs_per_s_central": "messages/s at the central controller",
         "msgs_per_s_max_regional": "messages/s at the busiest controller", "cpu_ms_per_sim_s": "CPU ms per simulated s"}
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def bars(ax, df, metric, groups, group_col="method", x_col=None, x_order=None, log=False):
    """Grouped bars: x = x_col values, one bar per method, error bars = 95% CI."""
    xs = x_order if x_order is not None else [None]
    w = 0.8 / len(groups)
    for j, g in enumerate(groups):
        m, h = [], []
        for xv in xs:
            d = df[df[group_col] == g]
            if x_col is not None:
                d = d[d[x_col] == xv]
            a, b = ci(d[metric].values) if len(d) else (np.nan, 0)
            m.append(a), h.append(b)
        pos = np.arange(len(xs)) + (j - (len(groups) - 1) / 2) * w
        ax.bar(pos, m, w * 0.92, yerr=h, color=COL.get(g, None), label=g, capsize=2, error_kw=dict(lw=0.8))
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([str(x) if x is not None else "" for x in xs], rotation=0)
    ax.set_ylabel(LABEL.get(metric, metric))
    if log:
        ax.set_yscale("log")


def save(fig, path):
    if not getattr(fig, "_keep_layout", False) and not fig.legends:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def read(p):
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="cg_out")
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args()
    rd, fd = os.path.join(a.workdir, "results"), os.path.join(a.workdir, "figures")
    os.makedirs(fd, exist_ok=True)
    core, var, abl = read(f"{rd}/core.csv"), read(f"{rd}/variations.csv"), read(f"{rd}/ablation.csv")
    M = ["CongestAgent", "B1", "B2", "B3"]

    if core is not None:
        fig, axs = plt.subplots(2, 3, figsize=(11, 6))
        for ax, m in zip(axs.ravel(), ["false_reroute_rate", "route_changes_per_min", "delay_p95_ms",
                                       "throughput_mbps", "loss_pct", "jain"]):
            bars(ax, core, m, M)
            ax.set_xticks([])
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
        fig.suptitle("Core experiment: 18 NSFNET configurations (mean, 95% CI)")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        save(fig, f"{fd}/fig1_core_overview.png")

        loads = sorted(core.load.unique())
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
        for ax, m in zip(axs, ["delay_p95_ms", "false_reroute_rate", "throughput_mbps"]):
            bars(ax, core, m, M, x_col="load", x_order=loads)
            ax.set_xticklabels([f"{int(l*100)}%" for l in loads])
            ax.set_xlabel("offered load")
        axs[0].legend(frameon=False)
        save(fig, f"{fd}/fig2_core_by_load.png")

        order = [n for n in ["none", "loss0.5", "loss1", "loss2", "jitter5", "jitter10"] if n in set(core.noise)]
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
        for ax, m in zip(axs, ["false_reroute_rate", "delay_p95_ms"]):
            bars(ax, core, m, M, x_col="noise", x_order=order)
            ax.set_xlabel("non-congestive noise")
        axs[0].legend(frameon=False)
        save(fig, f"{fd}/fig3_core_by_noise.png")

        fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
        for ax, m in zip(axs, ["decision_latency_ms", "msgs_per_s_central", "msgs_per_s_max_regional",
                               "cpu_ms_per_sim_s"]):
            bars(ax, core, m, M)
            ax.set_xticks([])
        axs[0].legend(frameon=False)
        fig.suptitle("Control cost (RQ2)")
        save(fig, f"{fd}/fig4_control_cost.png")

    if var is not None:
        var["variation"] = var.apply(lambda r: "geant" if r.topo == "geant" else
                                     (r.transport if r.transport == "cubic" else
                                      (r.event if r.event != "none" else r.noise)), axis=1)
        order = ["cubic", "burst", "failure", "geant", "shift40", "shift80", "jitter20", "jitter40"]
        order = [o for o in order if o in set(var.variation)]
        fig, axs = plt.subplots(2, 1, figsize=(11, 6))
        for ax, m in zip(axs, ["delay_p95_ms", "false_reroute_rate"]):
            bars(ax, var, m, M, x_col="variation", x_order=order)
        axs[0].legend(frameon=False, ncol=4)
        fig.suptitle("One-factor variations around load 90% (unseen: shift80, jitter40, GEANT)")
        save(fig, f"{fd}/fig5_variations.png")

    if abl is not None and core is not None:
        ca = core[(core.method == "CongestAgent") & core.scen_key.isin(abl.scen_key.unique()) &
                  core.seed.isin(abl.seed.unique())]
        ab = pd.concat([abl, ca], ignore_index=True)
        G = ["CongestAgent", "A1", "A2", "A3", "A4", "A5"]
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        for ax, m in zip(axs.ravel(), ["false_reroute_rate", "delay_p95_ms", "route_changes_per_min",
                                       "throughput_mbps", "msgs_per_s_central", "decision_latency_ms"]):
            bars(ax, ab, m, G)
            ax.set_xticks([])
        axs[0, 0].legend(frameon=False, ncol=2)
        fig.suptitle("Ablation (6 configurations x 5 seeds): A1 no degradation class, A2 no trend/corr features, "
                     "A3 no route penalty, A4 no fast loop, A5 central agent")
        save(fig, f"{fd}/fig6_ablation.png")

    ds = [d for d in (read(f"{rd}/stress_delay1.csv"), read(f"{rd}/stress_delay2.csv")) if d is not None]
    if ds and core is not None:
        s = pd.concat(ds, ignore_index=True)
        base = core[(core.method == "CongestAgent") & core.scen_key.isin(s.scen_key.unique()) &
                    core.seed.isin(s.seed.unique())].assign(tag="summary_delay0")
        al = pd.concat([base, s], ignore_index=True)
        names = {"summary_delay0": "0 (fresh)", "summary_delay1": "1 (0.25 s)", "summary_delay2": "2 (0.5 s)"}
        al["delay"] = al.tag.map(names)
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
        for ax, m in zip(axs, ["delay_p95_ms", "false_reroute_rate", "throughput_mbps"]):
            bars(ax, al, m, ["CongestAgent"], x_col="delay", x_order=list(names.values()))
            ax.set_xlabel("neighbour-summary delay (intervals)")
        save(fig, f"{fd}/fig7_delayed_summaries.png")

    if a.results_dir:
        names = [os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob(f"{a.results_dir}/*/metrics.json"))]
        data = {n: json.load(open(f"{a.results_dir}/{n}/metrics.json")) for n in names}
        splits = ["val", "test", "test_geant", "test_stress"]
        data = {n: m for n, m in data.items() if all(s in m for s in splits)}
        if data:
            fig, ax = plt.subplots(figsize=(10, 3.6))
            w = 0.8 / len(data)
            for j, (n, m) in enumerate(data.items()):
                ax.bar(np.arange(4) + (j - (len(data) - 1) / 2) * w, [m[s]["macro_f1"] for s in splits], w * 0.92, label=n)
            ax.set_xticks(range(4))
            ax.set_xticklabels(splits)
            ax.set_ylim(0.6, 1.0)
            ax.set_ylabel("macro F1")
            ax.legend(frameon=False, ncol=3)
            ax.set_title("Perception models (LightGBM feature sets and GRU)")
            save(fig, f"{fd}/fig8_perception_f1.png")
            pick = "robust" if "robust" in data else list(data)[0]
            cm = sum(np.array(data[pick][s]["confusion_matrix(rows=true, cols=pred)"], float)
                     for s in ["test", "test_geant", "test_stress"])
            pc = 100 * cm / cm.sum(1, keepdims=True)
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            ax.imshow(pc, cmap="Blues", vmin=0, vmax=100)
            lab = ["normal", "degradation", "congestion"]
            ax.set_xticks(range(3)), ax.set_xticklabels(lab, rotation=20), ax.set_yticks(range(3)), ax.set_yticklabels(lab)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{pc[i, j]:.2f}%", ha="center", va="center",
                            color="white" if pc[i, j] > 50 else "black")
            ax.set_xlabel("predicted"), ax.set_ylabel("true"), ax.grid(False)
            ax.set_title(f"{pick}: held-out splits (row %)")
            save(fig, f"{fd}/fig9_confusion_{pick}.png")
    print("all figures written to", fd)


if __name__ == "__main__":
    main()

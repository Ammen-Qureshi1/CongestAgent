"""Evaluate CongestAgent variants with a given decision period on a seed list. Same harness as cg_run.run_episode."""
import sys, warnings, os; warnings.filterwarnings("ignore")
import torch; torch.set_num_threads(1)
import pandas as pd, numpy as np
import cg_run as R
from cg_sim import load_calib, get_topology, Scenario
from stable_baselines3 import PPO
tag, out = sys.argv[1], sys.argv[2]          # tag: val | test | abl
names = sys.argv[3].split(",")               # e.g. CongestAgent_p4,A1_p4
cms, calib = R.load_cms("clfres"), load_calib("calib.json")
topo = get_topology("nsfnet")
clf_ms = 1.916                               # classifier latency per interval measured by cg_run.py (measure_classifier_ms)
if tag == "val":
    sc = [Scenario("nsfnet", l, n, "mixed", "none", s, 120.0) for l in (0.3, 0.6, 0.9) for n in R.CORE_NOISE for s in range(201, 206)]
elif tag == "test":
    sc = [Scenario("nsfnet", l, n, "mixed", "none", s, 120.0) for l in (0.3, 0.6, 0.9) for n in R.CORE_NOISE for s in range(101, 111)]
else:
    sc = [Scenario("nsfnet", l, n, "mixed", "none", s, 120.0) for l, n in R.ABL_CONFIGS for s in range(101, 106)]
rows = []
for nm in names:
    base, per = nm.rsplit("_p", 1)
    R.METHODS[nm] = dict(R.METHODS[base], period=int(per))
    model = PPO.load(os.path.join("models", nm), device="cpu")
    for s in sc:
        m = R.run_episode(nm, s, model, topo, calib, cms, clf_ms)
        m.update(method=nm, scen_key=s.key(), load=s.load, noise=s.noise, seed=s.seed, tag=tag)
        rows.append(m)
    print(nm, "done", flush=True)
pd.DataFrame(rows).to_csv(out, index=False)

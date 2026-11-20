"""Train one cg_run.py method with a given PPO decision period: python train_p.py <method> <period in intervals>."""
import sys, warnings; warnings.filterwarnings("ignore")
import torch; torch.set_num_threads(1)
import cg_run as R
from cg_sim import load_calib
base, per = sys.argv[1], int(sys.argv[2])
name = f"{base}_p{per}"
R.METHODS[name] = dict(R.METHODS[base], period=per)
R.train_one(name, 300_000, "models", R.load_cms("clfres"), load_calib("calib.json"), None, seed=0)

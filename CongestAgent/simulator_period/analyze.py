"""Means with 95% CI and Holm-corrected paired tests: 0.5 s agent (test_p2.csv) vs B1, B2, B3 and the 2 s agent."""
import pandas as pd, numpy as np, sys
sys.argv=["x"]; import cg_run as R
core=pd.read_csv("../cg_out/results/core.csv")
new=pd.read_csv("test_p2.csv"); p8=pd.read_csv("test_p8.csv")
# check retrained p8 vs original
o=core[core.method=="CongestAgent"].set_index(["scen_key","seed"]).sort_index(); q=p8.set_index(["scen_key","seed"]).sort_index()
print("retrained p8 vs original p95 max abs diff:", float((o.delay_p95_ms-q.delay_p95_ms).abs().max()))
new=new.assign(method="CA_p2")
df=pd.concat([core[core.method.isin(["B1","B2","B3","CongestAgent"])],new],ignore_index=True)
M=["delay_mean_ms","delay_p95_ms","loss_pct","throughput_mbps","jain","false_reroute_rate","route_changes_per_min","decision_latency_ms","msgs_per_s_central","msgs_per_s_max_regional","cpu_ms_per_sim_s"]
t=R.summary_table(df,M,["method"]); print(t.to_string())
c=R.compare(df,"CA_p2",["B1","B2","B3","CongestAgent"],M)
c.to_csv("test_paired_p2.csv",index=False)
print(c[["other","metric","ref_mean","other_mean","test","p_holm","effect","significant_0.05"]].round(4).to_string())
